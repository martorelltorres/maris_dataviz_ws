#!/usr/bin/env python3
"""Project side-scan sonar pings onto the seafloor.

Every port/starboard ping pair is slant-range corrected with the vehicle
altitude (flat-seafloor assumption), resampled on a uniform ground-range
grid, gain-normalised (running across-track mean) and colour-mapped.
Batches of consecutive pings become textured Collada strips placed along
the vehicle track, building a georeferenced SSS mosaic. A scrolling
waterfall image is also published.
"""

import math
import os
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import Image, Range
from visualization_msgs.msg import Marker, MarkerArray

from maris_dataviz.common import (SSS_LUT, ScalarBuffer, apply_lut, decode_any, grid_triangles,
                                  make_cache_dir, write_textured_dae)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SSSProjector(object):
    def __init__(self):
        self.world = rospy.get_param('~world_frame', 'world')
        self.default_range = float(rospy.get_param('~range', 50.0))
        self.bins = int(rospy.get_param('~ground_bins', 512))
        self.batch = int(rospy.get_param('~pings_per_tile', 40))
        self.gamma = float(rospy.get_param('~gamma', 0.7))
        self.gain = float(rospy.get_param("~gain", 0.3))
        self.ema = float(rospy.get_param('~normalization_rate', 0.01))
        self.lift = float(rospy.get_param('~lift', 0.02))
        self.waterfall_rows = int(rospy.get_param('~waterfall_rows', 500))
        # The Scout MkII driver stores port samples far -> near.
        self.port_reversed = bool(rospy.get_param('~port_reversed', True))

        self.cache = make_cache_dir('sss')
        self.tf = tf2_ros.Buffer(cache_time=rospy.Duration(60))
        tf2_ros.TransformListener(self.tf)
        self.alt = ScalarBuffer()
        self.range = {'port': None, 'starboard': None}
        self.half = {}           # stamp -> {'port': img, 'starboard': img}
        self.profile = None      # running across-track mean, (2, bins)
        self.rows, self.poses = [], []
        self.n_tiles = 0
        self.waterfall = np.zeros((self.waterfall_rows, 2 * self.bins, 3), np.uint8)
        self.last_wf_pub = rospy.Time(0)
        self.queue = []
        self.cv = threading.Condition()

        self.pub_tiles = rospy.Publisher('~mosaic', MarkerArray, queue_size=20)
        self.pub_wf = rospy.Publisher('~waterfall', Image, queue_size=1)
        rospy.Subscriber('altitude', Range, lambda r: self.alt.add(r.header.stamp.to_sec(), r.range), queue_size=100)
        for side in ('port', 'starboard'):
            rospy.Subscriber('sss/' + side, Image, self.on_ping, side, queue_size=200)
            rospy.Subscriber('sss/%s/sss_info' % side, rospy.AnyMsg, self.on_info, side, queue_size=10)
        t = threading.Thread(target=self.loop)
        t.daemon = True
        t.start()

    def on_info(self, msg, side):
        try:
            self.range[side] = float(decode_any(msg).range)
        except Exception:  # noqa
            pass

    def on_ping(self, img, side):
        key = img.header.stamp.to_nsec()
        entry = self.half.setdefault(key, {})
        entry[side] = img
        if len(entry) == 2:
            del self.half[key]
            with self.cv:
                self.queue.append(entry)
                self.cv.notify()
        if len(self.half) > 50:  # drop unmatched halves
            for k in sorted(self.half)[:-10]:
                del self.half[k]

    def loop(self):
        """Process ping pairs once TF covers their timestamp (non-blocking)."""
        while not rospy.is_shutdown():
            with self.cv:
                self.cv.wait(0.05)
                if not self.queue:
                    continue
                frame = self.queue[0]['port'].header.frame_id
                try:
                    latest = self.tf.lookup_transform(self.world, frame, rospy.Time(0)).header.stamp
                except Exception:  # noqa  (TF not available yet)
                    continue
                ready = [it for it in self.queue if it['port'].header.stamp <= latest]
                self.queue = [it for it in self.queue if it['port'].header.stamp > latest][-2000:]
            for it in sorted(ready, key=lambda e: e['port'].header.stamp):
                try:
                    self.process(it)
                except Exception as e:  # noqa
                    rospy.logwarn_throttle(5, 'sss_projector: %s', e)

    # ------------------------------------------------------------------
    def ground_resample(self, data, rng, alt):
        """Slant-range samples -> uniform ground-range bins in [0, rng]."""
        n = len(data)
        g = (np.arange(self.bins) + 0.5) * rng / self.bins
        slant = np.sqrt(g * g + alt * alt)
        idx = slant / rng * n - 0.5
        valid = idx < n - 1
        out = np.interp(idx, np.arange(n), data.astype(np.float32))
        out[~valid] = 0.0
        return out, valid

    def process(self, entry):
        stamp = entry['port'].header.stamp
        alt = self.alt.at(stamp.to_sec())
        if alt is None or alt <= 0.0:
            return
        try:
            tr = self.tf.lookup_transform(self.world, entry['port'].header.frame_id, stamp)
        except tf2_ros.ExtrapolationException:
            return  # older than the TF buffer
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn_throttle(10, 'sss_projector: TF %s', e)
            return
        sides, valids = [], []
        for side in ('port', 'starboard'):
            msg = entry[side]
            data = np.frombuffer(msg.data, np.uint8 if msg.encoding.endswith('8UC1') or msg.encoding == 'mono8'
                                 else np.uint16)
            if side == 'port' and self.port_reversed:
                data = data[::-1]
            rng = self.range[side] or self.default_range
            s, v = self.ground_resample(data, rng, alt)
            sides.append(s)
            valids.append(v)
        prof = np.stack(sides)
        valid = np.stack(valids)
        # Empirical gain normalisation with a running across-track mean.
        if self.profile is None:
            self.profile = np.maximum(prof, 0.5)
        else:
            self.profile += self.ema * (np.maximum(prof, 0.5) - self.profile)
        norm = prof / (cv2.GaussianBlur(self.profile, (0, 0), sigmaX=3)[:, :] + 1e-3)
        val = np.clip(self.gain * norm, 0.0, 1.0) ** self.gamma
        val[~valid] = 0.0
        # Row layout: port reversed (far port on the left) | starboard.
        row = np.concatenate([val[0, ::-1], val[1]])
        rgb = (apply_lut(SSS_LUT, row) * 255).astype(np.uint8)
        self.update_waterfall(rgb, stamp)

        t = tr.transform.translation
        floor_z = t.z - alt + self.lift
        self.rows.append(rgb)
        self.poses.append((t.x, t.y, floor_z, yaw_of(tr.transform.rotation),
                           self.range['port'] or self.default_range))
        if len(self.rows) >= self.batch:
            self.emit_tile()
            # Keep the last ping as the first one of the next strip (seamless).
            self.rows, self.poses = self.rows[-1:], self.poses[-1:]

    def emit_tile(self):
        rows = np.stack(self.rows)                     # (P, 2*bins, 3), RGB
        P = len(self.rows)
        rng = self.poses[0][4]
        # Vertex grid: pings (subsampled) x across-track positions.
        r_idx = np.unique(np.concatenate([np.arange(0, P, 4), [P - 1]]))
        n_cols = 33
        g = np.linspace(-rng, rng, n_cols)            # port negative, starboard positive
        verts, uvs = [], []
        for ri in r_idx:
            x, y, z, yaw, _ = self.poses[ri]
            # yaw is the heading of the bow about the z-up world axis, so
            # starboard is the heading rotated -90 deg.
            sx, sy = math.cos(yaw - math.pi / 2), math.sin(yaw - math.pi / 2)
            for gc in g:
                verts.append((x + gc * sx, y + gc * sy, z + 0.002 * (self.n_tiles % 20)))
                uvs.append(((gc + rng) / (2 * rng), 1.0 - (ri + 0.5) / P))
        tris = grid_triangles(len(r_idx), n_cols)
        idx = self.n_tiles
        self.n_tiles += 1
        name = 'sss_%05d' % idx
        cv2.imwrite(os.path.join(self.cache, name + '.png'), cv2.cvtColor(rows, cv2.COLOR_RGB2BGR))
        dae = os.path.join(self.cache, name + '.dae')
        write_textured_dae(dae, np.array(verts), np.array(uvs), tris, name + '.png')
        m = Marker()
        m.header.frame_id = self.world
        m.ns, m.id = 'sss_mosaic', idx
        m.type = Marker.MESH_RESOURCE
        m.mesh_resource = 'file://' + dae
        m.mesh_use_embedded_materials = True
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        self.pub_tiles.publish(MarkerArray(markers=[m]))

    def update_waterfall(self, rgb, stamp):
        self.waterfall = np.roll(self.waterfall, 1, axis=0)
        self.waterfall[0] = rgb
        now = rospy.Time.now()
        if self.pub_wf.get_num_connections() and ((now - self.last_wf_pub).to_sec() > 0.25 or now < self.last_wf_pub):
            self.last_wf_pub = now
            img = Image()
            img.header.stamp = stamp
            img.header.frame_id = 'sparus2/sidescan'
            img.height, img.width = self.waterfall.shape[:2]
            img.encoding = 'rgb8'
            img.step = img.width * 3
            img.data = self.waterfall.tobytes()
            self.pub_wf.publish(img)


if __name__ == '__main__':
    rospy.init_node('sss_projector')
    SSSProjector()
    rospy.spin()
