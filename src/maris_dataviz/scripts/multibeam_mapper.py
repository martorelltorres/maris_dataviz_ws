#!/usr/bin/env python3
"""Incremental bathymetry from multibeam swaths.

Each swath (PointCloud2 in the sonar frame) is transformed to the world
frame at its timestamp and binned on a regular grid (running mean depth per
cell with outlier rejection). Small gaps without soundings are filled from
the surrounding cells. The grid is split in tiles; tiles that changed are
re-published as hill-shaded, depth-coloured TRIANGLE_LIST markers. The live
swath fan (sonar -> soundings) is also drawn.
"""

import threading
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from maris_dataviz.common import BATHY_LUT, BathyGrid, SwathQueue, apply_lut


class MultibeamMapper(object):
    def __init__(self):
        self.world = rospy.get_param('~world_frame', 'world')
        self.period = float(rospy.get_param('~publish_period', 1.0))
        self.decimate = int(rospy.get_param('~swath_decimation', 1))
        self.sun = np.array(rospy.get_param('~sun_direction', [-0.5, 0.5, 0.7]), dtype=np.float64)
        self.sun /= np.linalg.norm(self.sun)
        self.grid = BathyGrid(cell=float(rospy.get_param('~cell_size', 0.25)),
                              tile_cells=int(rospy.get_param('~tile_cells', 64)),
                              outlier=float(rospy.get_param('~outlier_threshold', 0.8)),
                              min_count=int(rospy.get_param('~min_count', 2)),
                              fill_radius=float(rospy.get_param('~fill_radius', 1.0)))

        self.tf = tf2_ros.Buffer(cache_time=rospy.Duration(60))
        tf2_ros.TransformListener(self.tf)
        self.swaths = SwathQueue(self.tf, self.world, float(rospy.get_param('~max_range', 40.0)))
        self.dirty = set()
        self.dirty_lock = threading.Lock()
        self.zrange = None
        self.n_swaths = 0
        self.last_fan = rospy.Time(0)

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
        self.swaths.add(msg)

    def process_loop(self):
        while not rospy.is_shutdown():
            time.sleep(0.05)
            ready = self.swaths.pop_ready()
            if not ready:
                continue
            dirty = self.grid.accumulate(np.concatenate([w for _, _, w in ready]))
            with self.dirty_lock:
                self.dirty |= dirty
            msg, p, _ = ready[-1]
            self.publish_fan(msg, p)

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
        while not rospy.is_shutdown():
            time.sleep(self.period)  # wall clock: keeps refreshing at any playback rate
            try:
                self.publish_dirty()
            except Exception as e:  # noqa
                rospy.logwarn_throttle(5, 'multibeam_mapper: %s', e)

    def update_zrange(self):
        z = self.grid.valid_heights()
        if len(z) == 0:
            return False
        lo, hi = np.percentile(z, 2), np.percentile(z, 98)
        if hi - lo < 0.5:
            hi = lo + 0.5
        if self.zrange is None or abs(lo - self.zrange[0]) > 0.15 * (self.zrange[1] - self.zrange[0]) \
                or abs(hi - self.zrange[1]) > 0.15 * (self.zrange[1] - self.zrange[0]):
            self.zrange = (lo, hi)
            return True
        return False

    def publish_dirty(self):
        with self.dirty_lock:
            keys, self.dirty = self.dirty, set()
        if self.update_zrange():
            keys = set(self.grid.tiles.keys())  # recolour everything
        if not keys or self.zrange is None:
            return
        # Hole filling and smoothing reach into neighbour tiles: refresh them too.
        keys = {(ti + di, tj + dj) for (ti, tj) in keys for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (ti + di, tj + dj) in self.grid.tiles}
        arr = MarkerArray()
        for key in keys:
            m = self.tile_marker(key)
            if m is not None:
                arr.markers.append(m)
        if arr.markers:
            self.pub.publish(arr)

    def tile_marker(self, key):
        n, cell = self.grid.n, self.grid.cell
        ti, tj = key
        # (n+1) x (n+1) vertex grid (shares its last row/column with the
        # neighbours) plus one extra cell around for the gradients.
        Z = self.grid.heights_box(ti * n - 1, tj * n - 1, n + 3, n + 3)
        valid = np.isfinite(Z)
        if valid[1:-1, 1:-1].sum() < 3:
            return None
        gx, gy = np.gradient(np.where(valid, Z, np.nanmean(Z)), cell)
        Z, valid, gx, gy = Z[1:-1, 1:-1], valid[1:-1, 1:-1], gx[1:-1, 1:-1], gy[1:-1, 1:-1]
        # Hillshade from grid normals.
        nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        shade = np.clip(0.35 + 0.75 * nrm.dot(self.sun), 0.0, 1.0)
        lo, hi = self.zrange
        rgb = apply_lut(BATHY_LUT, (Z - lo) / (hi - lo)) * shade[..., None]

        X = (ti * n + np.arange(n + 1) + 0.5) * cell
        Y = (tj * n + np.arange(n + 1) + 0.5) * cell
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
