"""Shared helpers: bag discovery, geodesy, dynamic message decoding,
colormaps, time-indexed buffers and textured-mesh export for RViz."""

import bisect
import glob
import os
import re

import numpy as np

# --------------------------------------------------------------------------
# Bag discovery
# --------------------------------------------------------------------------

_MAIN_BAG_RE = re.compile(r'^sparus2_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_\d+\.bag$')


def find_mission_bags(mission_dir):
    """Return a dict with the bags of a mission folder.

    Keys: 'main' (list), 'camera', 'multibeam', 'sidescan', 'stereo_camera'.
    Every list is sorted by file name (i.e. by split index / time).
    """
    files = sorted(glob.glob(os.path.join(os.path.expanduser(mission_dir), '*.bag')))
    out = {'main': [], 'camera': [], 'multibeam': [], 'sidescan': [], 'stereo_camera': []}
    for f in files:
        name = os.path.basename(f)
        if _MAIN_BAG_RE.match(name):
            out['main'].append(f)
            continue
        for key in ('stereo_camera', 'camera', 'multibeam', 'sidescan'):
            if name.startswith('sparus2_%s_' % key):
                out[key].append(f)
                break
    return out


def main_bag(mission_dir):
    bags = find_mission_bags(mission_dir)['main']
    if not bags:
        raise RuntimeError('No main navigation bag (sparus2_<date>_N.bag) in %s' % mission_dir)
    return bags[0]


# --------------------------------------------------------------------------
# Geodesy (WGS84, local NED tangent plane)
# --------------------------------------------------------------------------

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def _ecef(lat, lon, h=0.0):
    lat = np.radians(lat)
    lon = np.radians(lon)
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - _E2) + h) * np.sin(lat)])


def geodetic_to_ned(lat, lon, lat0, lon0, depth=0.0):
    """Latitude/longitude (deg) -> (north, east, down) relative to the origin."""
    d = _ecef(lat, lon) - _ecef(lat0, lon0)
    la, lo = np.radians(lat0), np.radians(lon0)
    north = -np.sin(la) * np.cos(lo) * d[0] - np.sin(la) * np.sin(lo) * d[1] + np.cos(la) * d[2]
    east = -np.sin(lo) * d[0] + np.cos(lo) * d[1]
    return float(north), float(east), float(depth)


def ned_to_world(n, e, d):
    """world_ned -> world (world = world_ned rotated 180 deg about X)."""
    return n, -e, -d


# Web Mercator (XYZ map tiles, 256 px)

def lonlat_to_tile_px(lon, lat, zoom):
    """Longitude/latitude (deg) -> global Web Mercator pixel (x, y) at zoom."""
    s = 256.0 * 2 ** zoom
    lat = np.radians(np.clip(lat, -85.05112878, 85.05112878))
    x = (np.asarray(lon) + 180.0) / 360.0 * s
    y = (1.0 - np.log(np.tan(lat) + 1.0 / np.cos(lat)) / np.pi) / 2.0 * s
    return x, y


def tile_px_to_lonlat(x, y, zoom):
    """Global Web Mercator pixel (x, y) at zoom -> longitude/latitude (deg)."""
    s = 256.0 * 2 ** zoom
    lon = np.asarray(x) / s * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * np.asarray(y) / s))))
    return lon, lat


# --------------------------------------------------------------------------
# Dynamic decoding of messages whose packages are not installed
# --------------------------------------------------------------------------

_DYN_CACHE = {}


def decode_any(any_msg):
    """Deserialize a rospy.AnyMsg using the definition carried in the
    connection header (rosbag play forwards it), so custom vendor
    messages can be read without their packages being installed."""
    import genpy.dynamic
    hdr = any_msg._connection_header
    mtype = hdr['type']
    cls = _DYN_CACHE.get(mtype)
    if cls is None:
        cls = genpy.dynamic.generate_dynamic(mtype, hdr['message_definition'])[mtype]
        _DYN_CACHE[mtype] = cls
    msg = cls()
    msg.deserialize(any_msg._buff)
    return msg


# --------------------------------------------------------------------------
# Time-indexed scalar buffer (e.g. altitude) with linear interpolation
# --------------------------------------------------------------------------

class ScalarBuffer(object):
    def __init__(self, max_len=4000):
        self.t = []
        self.v = []
        self.max_len = max_len

    def add(self, t, v):
        if self.t and t <= self.t[-1]:
            if t < self.t[-1] - 2.0:  # time jumped backwards (bag restarted)
                self.t, self.v = [], []
            else:
                return
        self.t.append(t)
        self.v.append(v)
        if len(self.t) > self.max_len:
            del self.t[:len(self.t) - self.max_len]
            del self.v[:len(self.v) - self.max_len]

    def at(self, t, max_gap=2.0):
        if not self.t:
            return None
        i = bisect.bisect_left(self.t, t)
        if i <= 0:
            return self.v[0] if self.t[0] - t < max_gap else None
        if i >= len(self.t):
            return self.v[-1] if t - self.t[-1] < max_gap else None
        t0, t1 = self.t[i - 1], self.t[i]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return self.v[i - 1] * (1.0 - w) + self.v[i] * w


# --------------------------------------------------------------------------
# Colormaps (piecewise-linear LUTs, input in [0, 1])
# --------------------------------------------------------------------------

def _lut(stops):
    stops = np.asarray(stops, dtype=np.float64)
    x = np.linspace(0.0, 1.0, 256)
    pos = np.linspace(0.0, 1.0, len(stops))
    return np.stack([np.interp(x, pos, stops[:, c]) for c in range(3)], axis=1)


# Warm bronze palette, the usual look of side-scan imagery.
SSS_LUT = _lut([[0.00, 0.00, 0.00], [0.20, 0.10, 0.02], [0.50, 0.30, 0.08],
                [0.80, 0.58, 0.25], [0.97, 0.86, 0.60], [1.00, 1.00, 0.92]])
# Bathymetry palette: deep (0) blue/purple -> shallow (1) yellow/white.
BATHY_LUT = _lut([[0.10, 0.05, 0.35], [0.05, 0.25, 0.65], [0.00, 0.55, 0.75],
                  [0.20, 0.75, 0.55], [0.75, 0.85, 0.35], [1.00, 0.90, 0.55],
                  [1.00, 1.00, 0.90]])


def apply_lut(lut, values):
    idx = np.clip((np.asarray(values) * 255.0).astype(np.int32), 0, 255)
    return lut[idx]


# --------------------------------------------------------------------------
# Textured mesh export (Collada) for RViz MESH_RESOURCE markers
# --------------------------------------------------------------------------

def _fmt(arr):
    return ' '.join('%.4f' % v for v in np.asarray(arr).ravel())


def write_textured_dae(path, vertices, uvs, triangles, texture_file, normal=(0.0, 0.0, 1.0)):
    """Write a Collada mesh with a single diffuse texture.

    vertices: (N, 3) float, uvs: (N, 2) in OpenGL convention (v=0 is the
    bottom row of the image), triangles: (M, 3) int counter-clockwise when
    seen from the side the normal points to. texture_file is relative to
    the .dae location.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64).copy()
    # Make every triangle face the requested normal (avoids back-face culling).
    a, b, c = vertices[tris[:, 0]], vertices[tris[:, 1]], vertices[tris[:, 2]]
    flip = np.cross(b - a, c - a).dot(np.asarray(normal)) < 0.0
    tris[flip] = tris[flip][:, ::-1]
    n_v, n_t = len(vertices), len(tris)
    # <p> holds, per corner, the vertex/normal/uv indices (normal is shared).
    p = np.stack([tris, np.zeros_like(tris), tris], axis=2).ravel()
    dae = """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_images><image id="tex"><init_from>{tex}</init_from></image></library_images>
  <library_effects><effect id="fx"><profile_COMMON>
    <newparam sid="surf"><surface type="2D"><init_from>tex</init_from></surface></newparam>
    <newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>
    <technique sid="common"><lambert>
      <emission><color>0.25 0.25 0.25 1</color></emission>
      <ambient><color>1 1 1 1</color></ambient>
      <diffuse><texture texture="samp" texcoord="UV"/></diffuse>
    </lambert></technique>
  </profile_COMMON></effect></library_effects>
  <library_materials><material id="mat"><instance_effect url="#fx"/></material></library_materials>
  <library_geometries><geometry id="g"><mesh>
    <source id="pos"><float_array id="pos-a" count="{nv3}">{pos}</float_array>
      <technique_common><accessor source="#pos-a" count="{nv}" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
      </accessor></technique_common></source>
    <source id="nrm"><float_array id="nrm-a" count="3">{nrm}</float_array>
      <technique_common><accessor source="#nrm-a" count="1" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
      </accessor></technique_common></source>
    <source id="uv"><float_array id="uv-a" count="{nv2}">{uv}</float_array>
      <technique_common><accessor source="#uv-a" count="{nv}" stride="2">
        <param name="S" type="float"/><param name="T" type="float"/>
      </accessor></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
    <triangles material="m" count="{nt}">
      <input semantic="VERTEX" source="#v" offset="0"/>
      <input semantic="NORMAL" source="#nrm" offset="1"/>
      <input semantic="TEXCOORD" source="#uv" offset="2" set="0"/>
      <p>{p}</p>
    </triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="s"><node id="n">
    <instance_geometry url="#g"><bind_material><technique_common>
      <instance_material symbol="m" target="#mat"><bind_vertex_input semantic="UV" input_semantic="TEXCOORD" input_set="0"/></instance_material>
    </technique_common></bind_material></instance_geometry>
  </node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>
""".format(tex=texture_file, nv=n_v, nv3=3 * n_v, nv2=2 * n_v, nt=n_t,
           pos=_fmt(vertices), nrm=_fmt(normal), uv=_fmt(uvs),
           p=' '.join(str(int(i)) for i in p))
    tmp = path + '.part'
    with open(tmp, 'w') as f:
        f.write(dae)
    os.rename(tmp, path)


def grid_triangles(rows, cols):
    """Two CCW triangles per cell of a rows x cols vertex grid laid out
    row-major, assuming rows advance along +v and cols along +u in a
    right-handed (u, v) plane."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing='ij')
    i00 = (r * cols + c).ravel()
    i01 = i00 + 1
    i10 = i00 + cols
    i11 = i10 + 1
    return np.concatenate([np.stack([i00, i01, i11], 1), np.stack([i00, i11, i10], 1)])


def make_cache_dir(name):
    import time
    base = os.path.join(os.path.expanduser('~/.ros/maris_dataviz_cache'), name)
    run = os.path.join(base, time.strftime('%Y%m%d_%H%M%S') + '_%d' % os.getpid())
    os.makedirs(run)
    # Keep only the 3 most recent runs of this node.
    runs = sorted(glob.glob(os.path.join(base, '*')))
    import shutil
    for old in runs[:-3]:
        shutil.rmtree(old, ignore_errors=True)
    return run


# --------------------------------------------------------------------------
# Fast sequential bag reader (no index loading)
# --------------------------------------------------------------------------

def _fields(buf):
    out, i = {}, 0
    import struct
    while i < len(buf):
        n = struct.unpack_from('<I', buf, i)[0]
        k, _, v = buf[i + 4:i + 4 + n].partition(b'=')
        out[k.decode()] = v
        i += 4 + n
    return out


def read_first_messages(path, topics):
    """Return {topic: message} with the first message of each topic.

    Reads a ROS bag v2.0 sequentially from the beginning and stops as soon
    as every topic has been seen. This avoids rosbag.Bag(), which loads the
    whole index (tens of seconds for large bags). Messages are deserialized
    with the definition stored in the bag, so no message package is needed.
    """
    import bz2
    import struct
    import genpy.dynamic
    wanted = set(topics)
    conns, found, classes = {}, {}, {}

    def handle(op, hdr, data):
        if op == 0x07:
            c = _fields(data)
            conns[struct.unpack('<I', hdr['conn'])[0]] = (hdr['topic'].decode(), c)
        elif op == 0x02:
            cid = struct.unpack('<I', hdr['conn'])[0]
            topic, c = conns.get(cid, (None, None))
            if topic in wanted and topic not in found:
                mtype = c['type'].decode()
                cls = classes.get(mtype)
                if cls is None:
                    cls = classes[mtype] = genpy.dynamic.generate_dynamic(
                        mtype, c['message_definition'].decode())[mtype]
                msg = cls()
                msg.deserialize(data)
                found[topic] = msg

    def records(buf):
        i = 0
        while i + 4 <= len(buf):
            hl = struct.unpack_from('<I', buf, i)[0]
            hdr = _fields(buf[i + 4:i + 4 + hl])
            i += 4 + hl
            dl = struct.unpack_from('<I', buf, i)[0]
            yield hdr, buf[i + 4:i + 4 + dl]
            i += 4 + dl

    with open(path, 'rb') as f:
        if f.readline() != b'#ROSBAG V2.0\n':
            raise RuntimeError('%s is not a ROS bag v2.0' % path)
        while len(found) < len(wanted):
            raw = f.read(4)
            if len(raw) < 4:
                break
            hdr = _fields(f.read(struct.unpack('<I', raw)[0]))
            data = f.read(struct.unpack('<I', f.read(4))[0])
            op = hdr['op'][0]
            if op == 0x05:
                comp = hdr.get('compression', b'none')
                if comp == b'bz2':
                    data = bz2.decompress(data)
                elif comp == b'lz4':
                    import roslz4
                    data = roslz4.decompress(data)
                for h, d in records(data):
                    handle(h['op'][0], h, d)
                    if len(found) == len(wanted):
                        break
            elif op in (0x04, 0x06):
                break  # index section reached: end of data
            else:
                handle(op, hdr, data)
    return found


def bag_topics(path):
    """Topics recorded in a bag, read from the connection records stored
    after the index position (end of file); no message is read."""
    import struct
    topics = set()
    with open(path, 'rb') as f:
        if f.readline() != b'#ROSBAG V2.0\n':
            return topics
        hdr = _fields(f.read(struct.unpack('<I', f.read(4))[0]))
        index_pos = struct.unpack('<Q', hdr['index_pos'])[0]
        if index_pos == 0:
            return topics  # unindexed bag
        f.seek(index_pos)
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break
            h = _fields(f.read(struct.unpack('<I', raw)[0]))
            f.seek(struct.unpack('<I', f.read(4))[0], 1)
            if h['op'][0] != 0x07:
                break
            topics.add(h['topic'].decode())
    return topics


def detect_profile(mission_dir):
    """'sss' (side-scan data), 'optical' (camera and/or multibeam data),
    'full' (both) or 'nav' (navigation only)."""
    bags = find_mission_bags(mission_dir)
    has = lambda key, needle: any(any(needle in t for t in bag_topics(b)) for b in bags[key])
    sss = has('sidescan', '/raw_data/port')
    optical = has('multibeam', 'multibeam_scan') or has('stereo_camera', 'image_raw')
    return {(True, True): 'full', (True, False): 'sss', (False, True): 'optical'}.get((sss, optical), 'nav')


def bag_time_info(path, topic_filter=None):
    """(bag_start, first_time) in seconds from the bag index, without
    reading messages. first_time is the start of the first chunk holding a
    message of a topic matched by topic_filter(topic) (None if no match).
    Precision is one chunk (typically well below a few seconds)."""
    import struct
    conns, bag_start, first = {}, None, None
    with open(path, 'rb') as f:
        if f.readline() != b'#ROSBAG V2.0\n':
            return None, None
        hdr = _fields(f.read(struct.unpack('<I', f.read(4))[0]))
        index_pos = struct.unpack('<Q', hdr['index_pos'])[0]
        if index_pos == 0:
            return None, None
        f.seek(index_pos)
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break
            h = _fields(f.read(struct.unpack('<I', raw)[0]))
            data = f.read(struct.unpack('<I', f.read(4))[0])
            op = h['op'][0]
            if op == 0x07:
                conns[struct.unpack('<I', h['conn'])[0]] = h['topic'].decode()
            elif op == 0x06:
                s, ns = struct.unpack('<II', h['start_time'])
                t = s + ns * 1e-9
                bag_start = t if bag_start is None else min(bag_start, t)
                if topic_filter is not None:
                    n = struct.unpack('<I', h['count'])[0]
                    ids = [struct.unpack_from('<I', data, 8 * k)[0] for k in range(n)]
                    if any(topic_filter(conns.get(c, '')) for c in ids):
                        first = t if first is None else min(first, t)
    return bag_start, first


def motion_start_offset(mission_dir, speed=0.3, hold=20.0, window=10.0):
    """Seconds from the start of the mission's *_latlon.csv until the
    vehicle keeps moving faster than `speed` for `hold` seconds.
    Returns None when no CSV is available or the vehicle never moves."""
    import csv
    import datetime
    files = sorted(glob.glob(os.path.join(os.path.expanduser(mission_dir), '*_latlon.csv')))
    if not files:
        return None
    t, lat, lon = [], [], []
    with open(files[0]) as f:
        rows = csv.reader(f)
        next(rows, None)
        for r in rows:
            try:
                t.append(datetime.datetime.fromisoformat(r[0]).timestamp())
                lat.append(float(r[1]))
                lon.append(float(r[2]))
            except (ValueError, IndexError):
                continue
    if len(t) < 10:
        return None
    t, lat, lon = np.array(t), np.array(lat), np.array(lon)
    n = (lat - lat[0]) * 111132.0
    e = (lon - lon[0]) * 111320.0 * np.cos(np.radians(lat[0]))
    j = np.searchsorted(t, t + window)
    ok = j < len(t)
    i = np.nonzero(ok)[0]
    v = np.hypot(n[j[ok]] - n[i], e[j[ok]] - e[i]) / (t[j[ok]] - t[i])
    moving = v > speed
    for k in np.nonzero(moving)[0]:
        end = np.searchsorted(t[i], t[i[k]] + hold)
        if moving[k:end].all() and end < len(i):
            return float(t[i[k]] - t[0])
    return None


# --------------------------------------------------------------------------
# Seafloor samples from the navigation
# --------------------------------------------------------------------------

def seafloor_samples(path, nav_topic='/sparus2/navigator/navigation', should_stop=None):
    """(N, 3) array of (north, east, depth + altitude) along the mission.

    Reads the (throttled, if present) navigation of a main bag; the result
    is cached per bag because indexing a large bag is slow. Returns None if
    should_stop() becomes true while reading.
    """
    st = os.stat(path)
    cache_dir = os.path.expanduser('~/.ros/maris_dataviz_cache/seafloor')
    cache = os.path.join(cache_dir, '%s_%d.npy' % (os.path.basename(path), st.st_size))
    if os.path.exists(cache):
        return np.load(cache)
    import rosbag
    rows = []
    with rosbag.Bag(path) as bag:
        topics = bag.get_type_and_topic_info().topics
        topic, decim = nav_topic + '_throttle', 1
        if topic not in topics:
            topic, decim = nav_topic, 5
        for i, (_, nav, _) in enumerate(bag.read_messages(topics=[topic])):
            if should_stop is not None and should_stop():
                return None
            if i % decim == 0 and 0.0 < nav.altitude < 80.0:
                rows.append((nav.position.north, nav.position.east, nav.position.depth + nav.altitude))
    data = np.array(rows).reshape(-1, 3)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    np.save(cache, data)
    return data


# --------------------------------------------------------------------------
# Rotations, point clouds and camera images
# --------------------------------------------------------------------------

def quat_to_rot(q):
    """geometry_msgs Quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


_PC2_DT = {7: np.float32, 8: np.float64}


def cloud_xyz(msg):
    """(N, 3) float64 x, y, z of a sensor_msgs/PointCloud2."""
    f = {fl.name: fl for fl in msg.fields}
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, np.uint8).reshape(n, msg.point_step)
    cols = []
    for name in ('x', 'y', 'z'):
        fl = f[name]
        dt = np.dtype(_PC2_DT[fl.datatype]).newbyteorder('>' if msg.is_bigendian else '<')
        cols.append(buf[:, fl.offset:fl.offset + dt.itemsize].copy().view(dt).ravel())
    return np.stack(cols, 1).astype(np.float64)


class ImageEnhancer(object):
    """Undistortion (pinhole + distortion from CameraInfo, at any decoded
    resolution) and colour enhancement of underwater images."""

    def __init__(self, info, width, height):
        sx, sy = width / float(info.width), height / float(info.height)
        K = np.array(info.K, dtype=np.float64).reshape(3, 3)
        K[0, :] *= sx
        K[1, :] *= sy
        import cv2
        self.K = K
        self.size = (width, height)
        self.maps = cv2.initUndistortRectifyMap(K, np.array(info.D, dtype=np.float64), None, K,
                                                (width, height), cv2.CV_16SC2)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def undistort(self, bgr):
        import cv2
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            bgr = cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)
        return cv2.remap(bgr, self.maps[0], self.maps[1], cv2.INTER_LINEAR)

    def enhance(self, bgr):
        """Gray-world white balance followed by CLAHE on luminance."""
        import cv2
        f = bgr.astype(np.float32)
        means = f.reshape(-1, 3).mean(0) + 1e-3
        f *= means.mean() / means
        bgr = np.clip(f, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# Multibeam: swath ingestion and bathymetry grid
# --------------------------------------------------------------------------

class SwathQueue(object):
    """Queue of multibeam swaths transformed to the world frame once TF
    covers their timestamp. pop_ready() never blocks: swaths newer than the
    latest vehicle pose stay queued for the next call."""

    def __init__(self, tf_buffer, world, max_range=40.0, maxlen=4000):
        import collections
        self.tf = tf_buffer
        self.world = world
        self.max_range = max_range
        self.pending = collections.deque(maxlen=maxlen)

    def add(self, msg):
        self.pending.append(msg)

    def pop_ready(self):
        """List of (msg, points in sensor frame, points in world frame)."""
        import tf2_ros
        import rospy
        out = []
        if not self.pending:
            return out
        try:
            latest = self.tf.lookup_transform(self.world, self.pending[0].header.frame_id,
                                              rospy.Time(0)).header.stamp
        except Exception:  # noqa  (TF not available yet)
            return out
        while self.pending and self.pending[0].header.stamp <= latest:
            msg = self.pending.popleft()
            try:
                tr = self.tf.lookup_transform(self.world, msg.header.frame_id, msg.header.stamp)
            except tf2_ros.ExtrapolationException:
                continue  # older than the TF buffer
            except Exception as e:  # noqa
                rospy.logwarn_throttle(10, 'SwathQueue: TF %s', e)
                continue
            p = cloud_xyz(msg)
            r = np.linalg.norm(p, axis=1)
            p = p[np.isfinite(r) & (r > 0.3) & (r < self.max_range)]
            if len(p) == 0:
                continue
            t = tr.transform.translation
            out.append((msg, p, p.dot(quat_to_rot(tr.transform.rotation).T) + np.array([t.x, t.y, t.z])))
        return out


class BathyGrid(object):
    """Regular grid of soundings (running mean height per cell with outlier
    rejection), stored in square tiles of tile_cells x tile_cells cells.
    Cell (i, j) covers [i, i+1) x [j, j+1) * cell in world x, y."""

    class _Tile(object):
        def __init__(self, n):
            self.sum = np.zeros((n, n))
            self.cnt = np.zeros((n, n))

    def __init__(self, cell=0.2, tile_cells=64, outlier=0.8, min_count=2, fill_radius=1.0):
        import threading
        self.cell = cell
        self.n = tile_cells
        self.outlier = outlier
        self.min_count = min_count
        self.fill_radius = fill_radius
        self.tiles = {}
        self.lock = threading.Lock()

    def accumulate(self, w):
        """Add (N, 3) world points; returns the set of modified tile keys."""
        n = self.n
        ij = np.floor(w[:, :2] / self.cell).astype(np.int64)
        tk = np.floor_divide(ij, n)
        loc = ij - tk * n
        keys = tk[:, 0] * 1000003 + tk[:, 1]
        dirty = set()
        with self.lock:
            for k in np.unique(keys):
                sel = keys == k
                key = (int(tk[sel][0, 0]), int(tk[sel][0, 1]))
                tile = self.tiles.get(key)
                if tile is None:
                    tile = self.tiles[key] = self._Tile(n)
                li, lj, z = loc[sel, 0], loc[sel, 1], w[sel, 2]
                # Reject soundings far from the current cell mean.
                have = tile.cnt[li, lj] >= 3
                mean = np.where(have, tile.sum[li, lj] / np.maximum(tile.cnt[li, lj], 1), z)
                ok = np.abs(z - mean) < self.outlier
                np.add.at(tile.sum, (li[ok], lj[ok]), z[ok])
                np.add.at(tile.cnt, (li[ok], lj[ok]), 1)
                dirty.add(key)
        return dirty

    def valid_heights(self):
        """1-D array with the mean height of every valid cell."""
        with self.lock:
            zs = [t.sum[t.cnt >= self.min_count] / t.cnt[t.cnt >= self.min_count] for t in self.tiles.values()]
        zs = [z for z in zs if len(z)]
        return np.concatenate(zs) if zs else np.zeros(0)

    def window(self, i0, j0, ni, nj):
        """Raw (sum, count) arrays of the ni x nj cells starting at (i0, j0)."""
        n = self.n
        S = np.zeros((ni, nj))
        C = np.zeros((ni, nj))
        with self.lock:
            for ti in range(i0 // n, (i0 + ni - 1) // n + 1):
                for tj in range(j0 // n, (j0 + nj - 1) // n + 1):
                    t = self.tiles.get((ti, tj))
                    if t is None:
                        continue
                    a0, a1 = max(i0, ti * n), min(i0 + ni, ti * n + n)
                    b0, b1 = max(j0, tj * n), min(j0 + nj, tj * n + n)
                    S[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = t.sum[a0 - ti * n:a1 - ti * n, b0 - tj * n:b1 - tj * n]
                    C[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = t.cnt[a0 - ti * n:a1 - ti * n, b0 - tj * n:b1 - tj * n]
        return S, C

    def margin(self):
        """Cells of context needed around a window for heights()."""
        return int(round(self.fill_radius / self.cell)) + 2

    def heights(self, S, C):
        """Heights of a window (NaN where unknown). Small gaps (acoustic
        shadows, missing detections) are filled with the mean of the valid
        cells within fill_radius, only where at least half of that disc has
        data: interior holes are closed, swath edges are not extended. Then
        a light 3x3 smoothing. Cells closer than margin() to the window
        border lack context and should be cropped by the caller."""
        from scipy.ndimage import convolve
        valid = C >= self.min_count
        Z0 = np.where(valid, S / np.maximum(C, 1), 0.0)
        if self.fill_radius > 0.0:
            r = int(round(self.fill_radius / self.cell))
            y, x = np.mgrid[-r:r + 1, -r:r + 1]
            disc = ((x * x + y * y) <= r * r).astype(np.float64)
            wv = convolve(valid.astype(np.float64), disc, mode='constant')
            fill = ~valid & (wv >= 0.5 * disc.sum())
            Z0 = np.where(fill, convolve(Z0, disc, mode='constant') / np.maximum(wv, 1e-6), Z0)
            valid = valid | fill
        k = np.ones((3, 3))
        Zs = convolve(np.where(valid, Z0, 0.0), k, mode='constant') / \
            np.maximum(convolve(valid.astype(np.float64), k, mode='constant'), 1e-6)
        return np.where(valid, Zs, np.nan)

    def heights_box(self, i0, j0, ni, nj):
        """heights() of the ni x nj cells at (i0, j0), with full context."""
        m = self.margin()
        S, C = self.window(i0 - m, j0 - m, ni + 2 * m, nj + 2 * m)
        return self.heights(S, C)[m:m + ni, m:m + nj]
