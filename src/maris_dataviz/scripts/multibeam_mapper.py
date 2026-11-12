#!/usr/bin/env python3
"""Incremental bathymetry from multibeam swaths.

Each swath (PointCloud2 in the sonar frame) is transformed to the world
frame at its timestamp and binned on a regular grid (running mean depth per
cell with outlier rejection). The grid is split in tiles; tiles that
changed are re-published as hill-shaded, depth-coloured TRIANGLE_LIST
markers. The live swath fan (sonar -> soundings) is also drawn.
"""

import collections
import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from maris_dataviz.common import BATHY_LUT, apply_lut

DT = {7: np.float32, 8: np.float64}


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cloud_xyz(msg):
    f = {fl.name: fl for fl in msg.fields}
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, np.uint8).reshape(n, msg.point_step)
    cols = []
    for name in ('x', 'y', 'z'):
        fl = f[name]
        dt = np.dtype(DT[fl.datatype]).newbyteorder('>' if msg.is_bigendian else '<')
        cols.append(buf[:, fl.offset:fl.offset + dt.itemsize].copy().view(dt).ravel())
    return np.stack(cols, 1).astype(np.float64)


class Tile(object):
    def __init__(self, n):
        self.sum = np.zeros((n, n))
        self.cnt = np.zeros((n, n))


class MultibeamMapper(object):
    def __init__(self):
        self.world = rospy.get_param('~world_frame', 'world')
        self.cell = float(rospy.get_param('~cell_size', 0.25))
        self.tile_cells = int(rospy.get_param('~tile_cells', 64))
        self.max_range = float(rospy.get_param('~max_range', 40.0))
        self.min_count = int(rospy.get_param('~min_count', 2))
        self.outlier = float(rospy.get_param('~outlier_threshold', 0.8))
        self.period = float(rospy.get_param('~publish_period', 1.0))
        self.decimate = int(rospy.get_param('~swath_decimation', 1))
        self.sun = np.array(rospy.get_param('~sun_direction', [-0.5, 0.5, 0.7]), dtype=np.float64)
        self.sun /= np.linalg.norm(self.sun)

        self.tf = tf2_ros.Buffer(cache_time=rospy.Duration(60))
        tf2_ros.TransformListener(self.tf)
        self.tiles = {}
        self.dirty = set()
        self.zrange = None
        self.lock = threading.Lock()
        self.n_swaths = 0
        self.last_fan = rospy.Time(0)

        self.pending = collections.deque(maxlen=4000)

        self.pub = rospy.Publisher('~bathymetry', MarkerArray, queue_size=2)
        self.pub_fan = rospy.Publisher('~swath_fan', Marker, queue_size=1)
        rospy.Subscriber('multibeam', PointCloud2, self.on_cloud, queue_size=1000, buff_size=2 ** 24)
        threading.Thread(target=self.process_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop, daemon=True).start()

    # ------------------------------------------------------------------
    def on_cloud(self, msg):
        self.n_swaths += 1
        if self.decimate > 1 and self.n_swaths % self.decimate:
            return
        self.pending.append(msg)

    def process_loop(self):
        """Process queued swaths whose timestamp is already covered by TF.

        Lookups never block: swaths newer than the latest vehicle pose wait
        in the queue for the next iteration, so the mapper keeps up with
        high ping rates and fast playback."""
        import time
        while not rospy.is_shutdown():
            time.sleep(0.05)
            if not self.pending:
                continue
            frame = self.pending[0].header.frame_id
            try:
                latest = self.tf.lookup_transform(self.world, frame, rospy.Time(0)).header.stamp
            except Exception:  # noqa  (TF not available yet)
                continue
            batch, last = [], None
            while self.pending and self.pending[0].header.stamp <= latest:
                msg = self.pending.popleft()
                try:
                    tr = self.tf.lookup_transform(self.world, msg.header.frame_id, msg.header.stamp)
                except tf2_ros.ExtrapolationException:
                    continue  # older than the TF buffer
                except Exception as e:  # noqa
                    rospy.logwarn_throttle(10, 'multibeam_mapper: TF %s', e)
                    continue
                p = cloud_xyz(msg)
                r = np.linalg.norm(p, axis=1)
                p = p[np.isfinite(r) & (r > 0.3) & (r < self.max_range)]
                if len(p) == 0:
                    continue
                R = quat_to_rot(tr.transform.rotation)
                t = tr.transform.translation
                batch.append(p.dot(R.T) + np.array([t.x, t.y, t.z]))
                last = (msg, p)
            if batch:
                with self.lock:
                    self.accumulate(np.concatenate(batch))
            if last is not None:
                self.publish_fan(*last)

    def accumulate(self, w):
        n = self.tile_cells
        ij = np.floor(w[:, :2] / self.cell).astype(np.int64)
        tk = np.floor_divide(ij, n)
        loc = ij - tk * n
        keys = tk[:, 0] * 1000003 + tk[:, 1]
        for k in np.unique(keys):
            sel = keys == k
            key = (int(tk[sel][0, 0]), int(tk[sel][0, 1]))
            tile = self.tiles.get(key)
            if tile is None:
                tile = self.tiles[key] = Tile(n)
            li, lj, z = loc[sel, 0], loc[sel, 1], w[sel, 2]
            # Reject soundings far from the current cell mean.
            have = tile.cnt[li, lj] >= 3
            mean = np.where(have, tile.sum[li, lj] / np.maximum(tile.cnt[li, lj], 1), z)
            ok = np.abs(z - mean) < self.outlier
            np.add.at(tile.sum, (li[ok], lj[ok]), z[ok])
            np.add.at(tile.cnt, (li[ok], lj[ok]), 1)
            self.dirty.add(key)

    def publish_fan(self, msg, p):
        now = msg.header.stamp
        if (now - self.last_fan).to_sec() < 0.1 and now >= self.last_fan:
            return
        self.last_fan = now
        m = Marker()
        m.header.frame_id = msg.header.frame_id
        m.header.stamp = now
        m.ns, m.id = 'swath_fan', 0
        m.type = Marker.TRIANGLE_LIST
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color = ColorRGBA(0.3, 0.9, 1.0, 0.18)
        # Fan: triangles between the sonar head and consecutive soundings.
        q = p[np.argsort(np.arctan2(p[:, 0], -p[:, 2]))][::2]
        o = Point(0, 0, 0)
        for a, b in zip(q[:-1], q[1:]):
            if np.linalg.norm(a - b) < 2.0:
                m.points += [o, Point(*a), Point(*b)]
        self.pub_fan.publish(m)

    # ------------------------------------------------------------------
    def publish_loop(self):
        import time
        while not rospy.is_shutdown():
            time.sleep(self.period)  # wall clock: keeps refreshing at any playback rate
            try:
                self.publish_dirty()
            except Exception as e:  # noqa
                rospy.logwarn_throttle(5, 'multibeam_mapper: %s', e)

    def update_zrange(self):
        zs = []
        for t in self.tiles.values():
            v = t.cnt >= self.min_count
            if v.any():
                zs.append(t.sum[v] / t.cnt[v])
        if not zs:
            return False
        z = np.concatenate(zs)
        lo, hi = np.percentile(z, 2), np.percentile(z, 98)
        if hi - lo < 0.5:
            hi = lo + 0.5
        if self.zrange is None or abs(lo - self.zrange[0]) > 0.15 * (self.zrange[1] - self.zrange[0]) \
                or abs(hi - self.zrange[1]) > 0.15 * (self.zrange[1] - self.zrange[0]):
            self.zrange = (lo, hi)
            return True
        return False

    def publish_dirty(self):
        with self.lock:
            if self.update_zrange():
                self.dirty = set(self.tiles.keys())  # recolour everything
            keys, self.dirty = self.dirty, set()
            snap = {k: (self.tiles[k].sum.copy(), self.tiles[k].cnt.copy()) for k in keys}
            # Include one-cell borders from neighbours so tiles join seamlessly.
            nb = {}
            for (ti, tj) in keys:
                for d in ((1, 0), (0, 1), (1, 1)):
                    k2 = (ti + d[0], tj + d[1])
                    if k2 in self.tiles:
                        nb[(ti, tj, d)] = (self.tiles[k2].sum.copy(), self.tiles[k2].cnt.copy())
        if not snap or self.zrange is None:
            return
        arr = MarkerArray()
        for key, (s, c) in snap.items():
            m = self.tile_marker(key, s, c, nb)
            if m is not None:
                arr.markers.append(m)
        if arr.markers:
            self.pub.publish(arr)

    def tile_marker(self, key, s, c, nb):
        n = self.tile_cells
        ti, tj = key
        # Extended (n+1)x(n+1) height grid using the neighbour borders.
        S = np.zeros((n + 1, n + 1))
        Cn = np.zeros((n + 1, n + 1))
        S[:n, :n], Cn[:n, :n] = s, c
        if (ti, tj, (1, 0)) in nb:
            S[n, :n], Cn[n, :n] = nb[(ti, tj, (1, 0))][0][0, :], nb[(ti, tj, (1, 0))][1][0, :]
        if (ti, tj, (0, 1)) in nb:
            S[:n, n], Cn[:n, n] = nb[(ti, tj, (0, 1))][0][:, 0], nb[(ti, tj, (0, 1))][1][:, 0]
        if (ti, tj, (1, 1)) in nb:
            S[n, n], Cn[n, n] = nb[(ti, tj, (1, 1))][0][0, 0], nb[(ti, tj, (1, 1))][1][0, 0]
        valid = Cn >= self.min_count
        if valid.sum() < 3:
            return None
        Z = np.where(valid, S / np.maximum(Cn, 1), np.nan)
        # Light 3x3 smoothing of valid cells only.
        Zf = np.nan_to_num(Z)
        W = valid.astype(np.float64)
        k = np.ones((3, 3))
        from scipy.ndimage import convolve
        Zs = convolve(Zf, k, mode='nearest') / np.maximum(convolve(W, k, mode='nearest'), 1e-6)
        Z = np.where(valid, Zs, np.nan)

        # Hillshade from grid normals.
        gx, gy = np.gradient(np.nan_to_num(Z, nan=np.nanmean(Z)), self.cell)
        nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        shade = np.clip(0.35 + 0.75 * nrm.dot(self.sun), 0.0, 1.0)
        lo, hi = self.zrange
        rgb = apply_lut(BATHY_LUT, (Z - lo) / (hi - lo)) * shade[..., None]

        X = (ti * n + np.arange(n + 1) + 0.5) * self.cell
        Y = (tj * n + np.arange(n + 1) + 0.5) * self.cell
        # Quads with all four corners valid, split in two CCW triangles.
        q = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
        ii, jj = np.nonzero(q)
        if len(ii) == 0:
            return None
        corners = np.stack([np.stack([ii, jj], 1), np.stack([ii + 1, jj], 1), np.stack([ii + 1, jj + 1], 1),
                            np.stack([ii, jj], 1), np.stack([ii + 1, jj + 1], 1), np.stack([ii, jj + 1], 1)], 1)
        ci = corners.reshape(-1, 2)
        px, py, pz = X[ci[:, 0]], Y[ci[:, 1]], Z[ci[:, 0], ci[:, 1]]
        cols = rgb[ci[:, 0], ci[:, 1]]
        m = Marker()
        m.header.frame_id = self.world
        m.ns = 'bathymetry'
        m.id = (ti + 5000) * 10000 + (tj + 5000)
        m.type = Marker.TRIANGLE_LIST
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color = ColorRGBA(1, 1, 1, 1)
        m.points = [Point(a, b, cz) for a, b, cz in zip(px.tolist(), py.tolist(), pz.tolist())]
        m.colors = [ColorRGBA(r, g, b, 1.0) for r, g, b in cols.tolist()]
        return m


if __name__ == '__main__':
    rospy.init_node('multibeam_mapper')
    MultibeamMapper()
    rospy.spin()
