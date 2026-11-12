#!/usr/bin/env python3
"""Project the down-looking camera images onto the seafloor.

Each image is decoded at reduced resolution, undistorted, colour-enhanced
and back-projected through the pinhole model onto a flat seafloor plane
located `altitude` metres below the vehicle. Keyframes (one every time the
footprint has advanced a fraction of its length) are written as textured
Collada meshes and shown as persistent markers, building a 3D photomosaic.
The current footprint and camera frustum are drawn on every frame.
"""

import os
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Range
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from maris_dataviz.common import ScalarBuffer, grid_triangles, make_cache_dir, write_textured_dae

REDUCE = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
          4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class CameraProjector(object):
    def __init__(self):
        self.world = rospy.get_param('~world_frame', 'world')
        self.reduce = int(rospy.get_param('~decode_reduction', 4))
        self.tex_width = int(rospy.get_param('~texture_width', 400))
        self.overlap = float(rospy.get_param('~keyframe_advance', 0.45))
        self.max_alt = float(rospy.get_param('~max_altitude', 12.0))
        self.min_alt = float(rospy.get_param('~min_altitude', 0.5))
        self.lift = float(rospy.get_param('~lift', 0.05))
        self.enhance = bool(rospy.get_param('~enhance', True))
        self.max_tiles = int(rospy.get_param('~max_tiles', 3000))
        self.grid = (7, 9)  # vertex rows, cols of each projected tile

        self.cache = make_cache_dir('camera')
        self.tf = tf2_ros.Buffer(cache_time=rospy.Duration(60))
        tf2_ros.TransformListener(self.tf)
        self.alt = ScalarBuffer()
        self.info = None
        self.maps = None
        self.last_key = None
        self.n_tiles = 0
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.lock = threading.Lock()
        self.pending = None

        self.pub_tiles = rospy.Publisher('~mosaic', MarkerArray, queue_size=20)
        self.pub_live = rospy.Publisher('~footprint', MarkerArray, queue_size=1)
        self.pub_img = rospy.Publisher('~image_enhanced', Image, queue_size=1)
        rospy.Subscriber('altitude', Range, lambda r: self.alt.add(r.header.stamp.to_sec(), r.range), queue_size=100)
        rospy.Subscriber('camera_info', CameraInfo, self.on_info, queue_size=1)
        rospy.Subscriber('image', CompressedImage, self.on_image, queue_size=1, buff_size=2 ** 24)
        self.worker = threading.Thread(target=self.loop)
        self.worker.daemon = True
        self.worker.start()

    def on_info(self, msg):
        if self.info is None:
            self.info = msg

    def on_image(self, msg):
        # Keep only the latest image; processing runs in the worker thread.
        with self.lock:
            self.pending = msg

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            with self.lock:
                msg, self.pending = self.pending, None
            if msg is not None:
                try:
                    self.process(msg)
                except Exception as e:  # noqa
                    rospy.logwarn_throttle(5, 'camera_projector: %s', e)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    # ------------------------------------------------------------------
    def setup_maps(self, w, h):
        info = self.info
        sx, sy = w / float(info.width), h / float(info.height)
        K = np.array(info.K, dtype=np.float64).reshape(3, 3)
        K[0, :] *= sx
        K[1, :] *= sy
        D = np.array(info.D, dtype=np.float64)
        self.K = K
        self.maps = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)
        self.size = (w, h)

    def enhance_img(self, bgr):
        # Gray-world white balance followed by CLAHE on luminance.
        f = bgr.astype(np.float32)
        means = f.reshape(-1, 3).mean(0) + 1e-3
        f *= means.mean() / means
        bgr = np.clip(f, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process(self, msg):
        if self.info is None:
            rospy.logwarn_throttle(10, 'camera_projector: waiting for camera_info')
            return
        stamp = msg.header.stamp
        alt = self.alt.at(stamp.to_sec())
        if alt is None or not (self.min_alt < alt < self.max_alt):
            return
        try:
            tr = self.tf.lookup_transform(self.world, msg.header.frame_id, stamp, rospy.Duration(0.3))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn_throttle(10, 'camera_projector: TF %s', e)
            return
        try:
            base = self.tf.lookup_transform(self.world, 'sparus2/base_link', stamp, rospy.Duration(0.1))
            floor_z = base.transform.translation.z - alt
        except Exception:  # noqa
            floor_z = tr.transform.translation.z - alt

        R = quat_to_rot(tr.transform.rotation)
        C = np.array([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])

        # Footprint grid: pixel -> ray -> plane intersection.
        if self.maps is None:
            w = self.info.width // self.reduce
            h = self.info.height // self.reduce
            self.setup_maps(w, h)
        w, h = self.size
        rows, cols = self.grid
        us = np.linspace(0, w - 1, cols)
        vs = np.linspace(0, h - 1, rows)
        uu, vv = np.meshgrid(us, vs)
        rays = np.stack([(uu - self.K[0, 2]) / self.K[0, 0], (vv - self.K[1, 2]) / self.K[1, 1],
                         np.ones_like(uu)], -1).reshape(-1, 3)
        rays_w = rays.dot(R.T)
        if np.any(rays_w[:, 2] > -0.05):
            return  # camera not looking down enough
        lam = (floor_z - C[2]) / rays_w[:, 2]
        pts = C + rays_w * lam[:, None]
        pts[:, 2] += self.lift
        center = pts.mean(0)
        length = np.linalg.norm(pts[0] - pts[(rows - 1) * cols])  # image height on the floor

        self.publish_live(pts, rows, cols, C, msg.header.frame_id)

        key = self.last_key is None or np.linalg.norm(center[:2] - self.last_key[:2]) > self.overlap * length
        if not key and self.pub_img.get_num_connections() == 0:
            return

        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), REDUCE.get(self.reduce, cv2.IMREAD_COLOR))
        if img is None:
            return
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = cv2.remap(img, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
        if self.enhance:
            img = self.enhance_img(img)
        if self.pub_img.get_num_connections():
            out = Image()
            out.header = msg.header
            out.height, out.width = img.shape[:2]
            out.encoding = 'bgr8'
            out.step = out.width * 3
            out.data = img.tobytes()
            self.pub_img.publish(out)
        if key:
            self.last_key = center
            self.add_tile(img, pts, rows, cols, us / (w - 1), vs / (h - 1))

    def add_tile(self, img, pts, rows, cols, u_norm, v_norm):
        tw = self.tex_width
        th = int(round(img.shape[0] * tw / float(img.shape[1])))
        tex = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        idx = self.n_tiles
        self.n_tiles += 1
        name = 'tile_%05d' % idx
        cv2.imwrite(os.path.join(self.cache, name + '.jpg'), tex, [cv2.IMWRITE_JPEG_QUALITY, 90])
        uu, vv = np.meshgrid(u_norm, v_norm)
        uvs = np.stack([uu.ravel(), 1.0 - vv.ravel()], 1)  # OpenGL v: 0 = bottom
        # Small per-tile lift so overlapping tiles do not z-fight.
        verts = pts.copy()
        verts[:, 2] += 0.002 * (idx % 50)
        dae = os.path.join(self.cache, name + '.dae')
        write_textured_dae(dae, verts, uvs, grid_triangles(rows, cols), name + '.jpg')

        m = Marker()
        m.header.frame_id = self.world
        m.ns = 'camera_mosaic'
        m.id = idx % self.max_tiles
        m.type = Marker.MESH_RESOURCE
        m.mesh_resource = 'file://' + dae
        m.mesh_use_embedded_materials = True
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        arr = MarkerArray(markers=[m])
        self.pub_tiles.publish(arr)

    def publish_live(self, pts, rows, cols, C, cam_frame):
        corners = [0, cols - 1, rows * cols - 1, (rows - 1) * cols]
        P = [Point(*pts[i]) for i in corners]
        outline = Marker()
        outline.header.frame_id = self.world
        outline.ns, outline.id = 'footprint', 0
        outline.type = Marker.LINE_STRIP
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.04
        outline.color = ColorRGBA(1.0, 1.0, 0.3, 0.9)
        outline.points = P + [P[0]]
        frustum = Marker()
        frustum.header.frame_id = self.world
        frustum.ns, frustum.id = 'frustum', 0
        frustum.type = Marker.LINE_LIST
        frustum.pose.orientation.w = 1.0
        frustum.scale.x = 0.02
        frustum.color = ColorRGBA(1.0, 1.0, 0.3, 0.5)
        c = Point(*C)
        for p in P:
            frustum.points += [c, p]
        self.pub_live.publish(MarkerArray(markers=[outline, frustum]))


if __name__ == '__main__':
    rospy.init_node('camera_projector')
    CameraProjector()
    rospy.spin()
