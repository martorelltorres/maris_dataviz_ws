#!/usr/bin/env python3
"""Multibeam bathymetry textured with the camera images.

Builds its own bathymetry grid from the multibeam swaths (same gridding,
outlier rejection and gap filling as multibeam_mapper) and takes camera
keyframes (one every time the footprint has advanced a fraction of its
length). Each keyframe waits until the multibeam has covered its footprint;
then the grid vertices seen by the camera are projected into the
(undistorted, colour-enhanced) image, giving a textured patch of the real
3D seafloor instead of a flat-floor projection. Patches are written as
textured Collada meshes and published as persistent markers.

Keyframes whose footprint is not covered by the multibeam after ~max_wait
seconds (ROS time) are textured with whatever coverage exists, or dropped
if there is almost none.
"""

import os
import threading
import time

import cv2
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, Range
from visualization_msgs.msg import Marker, MarkerArray

from maris_dataviz.common import (BathyGrid, ImageEnhancer, ScalarBuffer, SwathQueue, make_cache_dir,
                                  quat_to_rot, write_textured_dae)

REDUCE = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
          4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


class Keyframe(object):
    def __init__(self, img, R, C, stamp, box):
        self.img, self.R, self.C, self.stamp, self.box = img, R, C, stamp, box


class TexturedBathymetry(object):
    def __init__(self):
        self.world = rospy.get_param('~world_frame', 'world')
        self.grid = BathyGrid(cell=float(rospy.get_param('~cell_size', 0.2)),
                              tile_cells=int(rospy.get_param('~tile_cells', 64)),
                              outlier=float(rospy.get_param('~outlier_threshold', 0.8)),
                              min_count=int(rospy.get_param('~min_count', 2)),
                              fill_radius=float(rospy.get_param('~fill_radius', 1.0)))
        self.reduce = int(rospy.get_param('~decode_reduction', 4))
        self.tex_width = int(rospy.get_param('~texture_width', 512))
        self.advance = float(rospy.get_param('~keyframe_advance', 0.5))
        self.min_alt = float(rospy.get_param('~min_altitude', 0.5))
        self.max_alt = float(rospy.get_param('~max_altitude', 12.0))
        self.min_cov = float(rospy.get_param('~min_coverage', 0.7))
        self.max_wait = float(rospy.get_param('~max_wait', 30.0))
        self.lift = float(rospy.get_param('~lift', 0.03))
        self.min_cos = np.cos(np.radians(float(rospy.get_param('~max_view_angle', 70.0))))
        self.max_tiles = int(rospy.get_param('~max_tiles', 3000))

        self.cache = make_cache_dir('textured_bathymetry')
        self.tf = tf2_ros.Buffer(cache_time=rospy.Duration(60))
        tf2_ros.TransformListener(self.tf)
        self.swaths = SwathQueue(self.tf, self.world, float(rospy.get_param('~max_range', 40.0)))
        self.alt = ScalarBuffer()
        self.info = None
        self.enh = None
        self.last_key = None
        self.keyframes = []
        self.n_tiles = 0
        self.lock = threading.Lock()
        self.latest_img = None

        self.pub = rospy.Publisher('~mesh', MarkerArray, queue_size=20)
        rospy.Subscriber('multibeam', PointCloud2, self.swaths.add, queue_size=1000, buff_size=2 ** 24)
        rospy.Subscriber('altitude', Range, lambda r: self.alt.add(r.header.stamp.to_sec(), r.range), queue_size=100)
        rospy.Subscriber('camera_info', CameraInfo, self.on_info, queue_size=1)
        rospy.Subscriber('image', CompressedImage, self.on_image, queue_size=1, buff_size=2 ** 24)
        for target in (self.swath_loop, self.image_loop, self.texture_loop):
            threading.Thread(target=target, daemon=True).start()

    def on_info(self, msg):
        if self.info is None:
            self.info = msg

    def on_image(self, msg):
        with self.lock:
            self.latest_img = msg

    # ------------------------------------------------------------ multibeam
    def swath_loop(self):
        while not rospy.is_shutdown():
            time.sleep(0.05)
            ready = self.swaths.pop_ready()
            if ready:
                self.grid.accumulate(np.concatenate([w for _, _, w in ready]))

    # ------------------------------------------------------------ keyframes
    def image_loop(self):
        while not rospy.is_shutdown():
            time.sleep(0.02)
            with self.lock:
                msg, self.latest_img = self.latest_img, None
            if msg is None or self.info is None:
                continue
            try:
                self.add_keyframe(msg)
            except Exception as e:  # noqa
                rospy.logwarn_throttle(5, 'textured_bathymetry: %s', e)

    def add_keyframe(self, msg):
        stamp = msg.header.stamp
        alt = self.alt.at(stamp.to_sec())
        if alt is None or not (self.min_alt < alt < self.max_alt):
            return
        try:
            tr = self.tf.lookup_transform(self.world, msg.header.frame_id, stamp, rospy.Duration(0.3))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            return
        if self.enh is None:
            self.enh = ImageEnhancer(self.info, self.info.width // self.reduce, self.info.height // self.reduce)
        R = quat_to_rot(tr.transform.rotation)
        C = np.array([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])
        # Approximate footprint on a flat floor `alt` below the camera; only
        # used to pick keyframes and the grid window to texture.
        w, h = self.enh.size
        K = self.enh.K
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
        rays = np.stack([(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1],
                         np.ones(4)], -1).dot(R.T)
        if np.any(rays[:, 2] > -0.05):
            return
        foot = C + rays * (-alt / rays[:, 2])[:, None]
        center = foot.mean(0)
        length = np.linalg.norm(foot[0] - foot[3])
        if self.last_key is not None and np.linalg.norm(center[:2] - self.last_key[:2]) < self.advance * length:
            return
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), REDUCE.get(self.reduce, cv2.IMREAD_COLOR))
        if img is None:
            return
        img = self.enh.enhance(self.enh.undistort(img))
        self.last_key = center
        margin = 0.5 + 0.3 * alt  # relief can push the real footprint outwards
        box = (foot[:, 0].min() - margin, foot[:, 0].max() + margin,
               foot[:, 1].min() - margin, foot[:, 1].max() + margin)
        with self.lock:
            self.keyframes.append(Keyframe(img, R, C, stamp, box))

    # ------------------------------------------------------------ texturing
    def texture_loop(self):
        while not rospy.is_shutdown():
            time.sleep(0.5)
            with self.lock:
                todo = list(self.keyframes)
            now = rospy.Time.now()
            for kf in todo:
                try:
                    done = self.texture(kf, (now - kf.stamp).to_sec() > self.max_wait)
                except Exception as e:  # noqa
                    rospy.logwarn_throttle(5, 'textured_bathymetry: %s', e)
                    done = True
                if done:
                    with self.lock:
                        self.keyframes.remove(kf)

    def texture(self, kf, timeout):
        """Texture one keyframe; returns False to retry later."""
        cell = self.grid.cell
        x0, x1, y0, y1 = kf.box
        i0, j0 = int(np.floor(x0 / cell)), int(np.floor(y0 / cell))
        ni, nj = int(np.ceil(x1 / cell)) - i0 + 1, int(np.ceil(y1 / cell)) - j0 + 1
        Z = self.grid.heights_box(i0, j0, ni, nj)
        X = (i0 + np.arange(ni) + 0.5) * cell
        Y = (j0 + np.arange(nj) + 0.5) * cell
        XX, YY = np.meshgrid(X, Y, indexing='ij')
        # Grid vertices -> camera pixels (on a flat floor for unknown cells,
        # only to measure the coverage of the image).
        zf = np.where(np.isfinite(Z), Z, np.nanmedian(Z) if np.isfinite(Z).any() else kf.C[2] - 3.0)
        P = np.stack([XX, YY, zf], -1).reshape(-1, 3)
        pc = (P - kf.C).dot(kf.R)
        K = self.enh.K
        w, h = self.enh.size
        with np.errstate(divide='ignore', invalid='ignore'):
            u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
            v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
        seen = (pc[:, 2] > 0.05) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        valid = np.isfinite(Z).ravel()
        n_seen = seen.sum()
        if n_seen == 0:
            return True
        coverage = (seen & valid).sum() / float(n_seen)
        if coverage < self.min_cov and not timeout:
            return False
        if (seen & valid).sum() < 20:
            return True  # no bathymetry under this image

        ok = (seen & valid).reshape(ni, nj)
        q = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
        ii, jj = np.nonzero(q)
        if len(ii) == 0:
            return True
        idx = lambda a, b: a * nj + b  # noqa: E731
        tris = np.concatenate([np.stack([idx(ii, jj), idx(ii + 1, jj), idx(ii + 1, jj + 1)], 1),
                               np.stack([idx(ii, jj), idx(ii + 1, jj + 1), idx(ii, jj + 1)], 1)])
        # Drop triangles seen at grazing angles (steep walls): their texture
        # would be smeared; the coloured bathymetry shows through instead.
        a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
        nrm = np.cross(b - a, c - a)
        nrm *= np.sign(nrm[:, 2:3]) + (nrm[:, 2:3] == 0)
        view = kf.C - (a + b + c) / 3.0
        cos = np.sum(nrm * view, 1) / (np.linalg.norm(nrm, axis=1) * np.linalg.norm(view, axis=1) + 1e-9)
        tris = tris[cos > self.min_cos]
        if len(tris) == 0:
            return True
        used, tris = np.unique(tris, return_inverse=True)
        tris = tris.reshape(-1, 3)
        verts = P[used].copy()
        tile = self.n_tiles
        self.n_tiles += 1
        # Above the coloured bathymetry; small per-patch step so overlapping
        # patches do not z-fight.
        verts[:, 2] += self.lift + 0.002 * (tile % 50)
        uvs = np.stack([u[used] / (w - 1), 1.0 - v[used] / (h - 1)], 1)

        name = 'patch_%05d' % tile
        th = int(round(kf.img.shape[0] * self.tex_width / float(kf.img.shape[1])))
        cv2.imwrite(os.path.join(self.cache, name + '.jpg'),
                    cv2.resize(kf.img, (self.tex_width, th), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        dae = os.path.join(self.cache, name + '.dae')
        write_textured_dae(dae, verts, uvs, tris, name + '.jpg')
        m = Marker()
        m.header.frame_id = self.world
        m.ns = 'textured_bathymetry'
        m.id = tile % self.max_tiles
        m.type = Marker.MESH_RESOURCE
        m.mesh_resource = 'file://' + dae
        m.mesh_use_embedded_materials = True
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        self.pub.publish(MarkerArray(markers=[m]))
        return True


if __name__ == '__main__':
    rospy.init_node('textured_bathymetry')
    TexturedBathymetry()
    rospy.spin()
