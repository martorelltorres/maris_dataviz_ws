#!/usr/bin/env python3
"""Satellite / aerial orthophoto layer under the mission.

Downloads the XYZ map tiles that cover the mission area (plus a margin),
stitches them into one texture and publishes it as a textured mesh draped
over the seafloor measured during the mission (depth + altitude of the
navigation, smoothed and extrapolated outside the surveyed area), slightly
below it so the camera / side-scan mosaics and the bathymetry stay on top.

Default source: PNOA orthophoto of the Spanish Instituto Geografico
Nacional (WMTS, CC BY 4.0, "PNOA cedido por (c) Instituto Geografico
Nacional"). Any XYZ source can be used through ~tile_url ({z}, {x}, {y}).
Tiles are cached in ~/.ros/maris_dataviz_cache/satellite/tiles, so the
network is only needed the first time an area is shown.
"""

import hashlib
import os
import urllib.request

import cv2
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from maris_dataviz.common import (geodetic_to_ned, grid_triangles, lonlat_to_tile_px, main_bag,
                                  ned_to_world, read_first_messages, seafloor_samples,
                                  tile_px_to_lonlat, write_textured_dae)

NAV_TOPIC = '/sparus2/navigator/navigation'
PNOA_URL = ('https://www.ign.es/wmts/pnoa-ma?request=GetTile&service=WMTS&VERSION=1.0.0'
            '&Layer=OI.OrthoimageCoverage&Style=default&Format=image/jpeg'
            '&TileMatrixSet=GoogleMapsCompatible&TileMatrix={z}&TileRow={y}&TileCol={x}')
CACHE = os.path.expanduser('~/.ros/maris_dataviz_cache/satellite')


class SatelliteLayer(object):
    def __init__(self):
        self.frame = rospy.get_param('~world_frame', 'world')
        self.url = rospy.get_param('~tile_url', PNOA_URL)
        self.margin = float(rospy.get_param('~margin', 100.0))
        self.max_zoom = int(rospy.get_param('~max_zoom', 19))
        self.max_px = int(rospy.get_param('~max_texture', 4096))
        self.offset = float(rospy.get_param('~depth_offset', 0.5))
        self.grid = int(rospy.get_param('~grid', 48))
        self.pub = rospy.Publisher('~layer', MarkerArray, queue_size=1, latch=True)

        bag = main_bag(rospy.get_param('~mission_dir'))
        nav = read_first_messages(bag, [NAV_TOPIC]).get(NAV_TOPIC)
        if nav is None:
            rospy.logerr('satellite_layer: no navigation in %s', bag)
            return
        self.lat0, self.lon0 = nav.origin.latitude, nav.origin.longitude
        floor = seafloor_samples(bag, NAV_TOPIC, rospy.is_shutdown)
        if floor is None:
            return
        if len(floor) < 10:
            rospy.logwarn('satellite_layer: no altitude data, using a flat seafloor')
            d0 = nav.position.depth + nav.altitude if nav.altitude > 0.0 else 10.0
            floor = np.array([[nav.position.north, nav.position.east, d0]])
        self.floor = floor
        self.build()

    # ------------------------------------------------------------- tiles
    def tile(self, z, x, y):
        key = hashlib.md5(self.url.encode()).hexdigest()[:8]
        path = os.path.join(CACHE, 'tiles', key, str(z), str(x), '%d.jpg' % y)
        if not os.path.exists(path):
            req = urllib.request.Request(self.url.format(z=z, x=x, y=y),
                                         headers={'User-Agent': 'maris_dataviz'})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + '.part', 'wb') as f:
                f.write(data)
            os.rename(path + '.part', path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (256, 256):
            os.remove(path)
            raise IOError('invalid tile %d/%d/%d' % (z, x, y))
        return img

    def area(self):
        """Lat/lon bounds of the mission samples plus the margin."""
        n0, e0 = self.floor[:, 0].min() - self.margin, self.floor[:, 1].min() - self.margin
        n1, e1 = self.floor[:, 0].max() + self.margin, self.floor[:, 1].max() + self.margin
        m_lat = 111132.0
        m_lon = 111320.0 * np.cos(np.radians(self.lat0))
        return (self.lat0 + n0 / m_lat, self.lat0 + n1 / m_lat,
                self.lon0 + e0 / m_lon, self.lon0 + e1 / m_lon)

    # ------------------------------------------------------------- build
    def build(self):
        lat_a, lat_b, lon_a, lon_b = self.area()
        for z in range(self.max_zoom, 0, -1):
            x0, y0 = lonlat_to_tile_px(lon_a, lat_b, z)
            x1, y1 = lonlat_to_tile_px(lon_b, lat_a, z)
            tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
            if max(tx1 - tx0 + 1, ty1 - ty0 + 1) * 256 <= self.max_px:
                break
        nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
        rospy.loginfo('satellite_layer: zoom %d, %dx%d tiles (%dx%d px)', z, nx, ny, nx * 256, ny * 256)

        img = np.full((ny * 256, nx * 256, 3), 90, np.uint8)
        failed = 0
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                if rospy.is_shutdown():
                    return
                try:
                    t = self.tile(z, tx, ty)
                except Exception as e:  # noqa
                    failed += 1
                    rospy.logdebug('satellite_layer: tile %d/%d/%d: %s', z, tx, ty, e)
                    continue
                r, c = (ty - ty0) * 256, (tx - tx0) * 256
                img[r:r + 256, c:c + 256] = t
        if failed == nx * ny:
            rospy.logerr('satellite_layer: no tile could be downloaded from %s', self.url.split('?')[0])
            return
        if failed:
            rospy.logwarn('satellite_layer: %d of %d tiles missing', failed, nx * ny)

        out = os.path.join(CACHE, 'mesh_%s' % hashlib.md5(('%s %d %d %d %d %d %.6f %.6f %d %.2f' % (
            self.url, z, tx0, ty0, tx1, ty1, self.lat0, self.lon0, len(self.floor), self.offset)).encode()).hexdigest()[:12])
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, 'ortho.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Vertex grid in Web Mercator pixels; texture coordinates are linear in it.
        g = self.grid
        u = np.linspace(0.0, 1.0, g)
        v = np.linspace(0.0, 1.0, g)
        uu, vv = np.meshgrid(u, v)                      # rows: north -> south
        px = tx0 * 256 + uu * nx * 256
        py = ty0 * 256 + vv * ny * 256
        lon, lat = tile_px_to_lonlat(px, py, z)
        ne = np.array([geodetic_to_ned(la, lo, self.lat0, self.lon0)[:2]
                       for la, lo in zip(lat.ravel(), lon.ravel())])
        depth = self.depth_at(ne) + self.offset
        verts = np.array([ned_to_world(n, e, d) for (n, e), d in zip(ne, depth)])
        uvs = np.stack([uu.ravel(), 1.0 - vv.ravel()], 1)   # OpenGL v: 0 = bottom
        dae = os.path.join(out, 'ortho.dae')
        write_textured_dae(dae, verts, uvs, grid_triangles(g, g), 'ortho.jpg')

        m = Marker()
        m.header.frame_id = self.frame
        m.ns, m.id = 'satellite', 0
        m.type = Marker.MESH_RESOURCE
        m.mesh_resource = 'file://' + dae
        m.mesh_use_embedded_materials = True
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        self.pub.publish(MarkerArray(markers=[m]))
        rospy.loginfo('satellite_layer: %.0f x %.0f m published (%s)',
                      np.ptp(ne[:, 1]), np.ptp(ne[:, 0]), dae)

    def depth_at(self, ne):
        """Smoothed seafloor depth; far from the samples it tends to their
        local mean, so the extrapolation stays flat."""
        from scipy.spatial import cKDTree
        f = self.floor
        k = min(64, len(f))
        d, idx = cKDTree(f[:, :2]).query(ne, k=k)
        d, idx = d.reshape(len(ne), k), idx.reshape(len(ne), k)
        w = 1.0 / (d + 5.0) ** 2
        return np.sum(f[idx, 2] * w, 1) / np.sum(w, 1)


if __name__ == '__main__':
    rospy.init_node('satellite_layer')
    SatelliteLayer()
    rospy.spin()
