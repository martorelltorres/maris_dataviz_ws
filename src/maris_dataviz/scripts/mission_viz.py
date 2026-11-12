#!/usr/bin/env python3
"""Mission plan visualization.

Reads the mission XML (COLA2 captain format) and the NED origin straight
from the navigation bag, so the full plan is shown from the first second
regardless of the playback start offset. Altitude-referenced legs are drawn
at their commanded altitude over the seafloor measured during the mission
(depth + altitude of the vehicle navigation), so the plan sits where the
vehicle was expected to fly. Also draws a translucent sea surface.
"""

import os
import threading
import xml.etree.ElementTree as ET

import numpy as np
import rosbag
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from maris_dataviz.common import geodetic_to_ned, main_bag, ned_to_world, read_first_messages

MISSION_TOPIC = '/sparus2/captain/mission'
NAV_TOPIC = '/sparus2/navigator/navigation'

COLORS = {
    'goto': ColorRGBA(0.20, 0.60, 1.00, 1.0),
    'section': ColorRGBA(1.00, 0.45, 0.10, 1.0),
    'park': ColorRGBA(0.75, 0.30, 1.00, 1.0),
}


def ftext(node, tag, default=None):
    el = node.find(tag)
    if el is None or el.text is None:
        return default
    try:
        return float(el.text)
    except ValueError:
        return el.text.strip()


class Seafloor(object):
    """Seafloor depth (NED down) from the navigation (depth + altitude)."""

    def __init__(self, ne, floor, flat=None):
        self.flat = flat
        self.ok = len(ne) > 10
        if self.ok:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(ne)
            self.floor = floor
            self.default = float(np.median(floor))

    def depth_at(self, n, e):
        if not self.ok:
            return self.flat
        d, idx = self.tree.query([n, e], k=8)
        w = 1.0 / (np.asarray(d) + 0.5)
        return float(np.sum(self.floor[idx] * w) / np.sum(w))


class MissionViz(object):
    def __init__(self):
        self.frame = rospy.get_param('~world_frame', 'world')
        self.line_width = rospy.get_param('~line_width', 0.25)
        self.surface_size = rospy.get_param('~surface_margin', 40.0)
        mission_dir = rospy.get_param('~mission_dir')
        self.pub = rospy.Publisher('~markers', MarkerArray, queue_size=1, latch=True)

        self.origin = None
        self.xml = None
        self.seafloor = Seafloor([], [])
        bag = main_bag(mission_dir)
        first = read_first_messages(bag, [MISSION_TOPIC, NAV_TOPIC])
        nav = first.get(NAV_TOPIC)
        if nav is not None:
            self.origin = (nav.origin.latitude, nav.origin.longitude)
            # Until the full model is loaded, assume a flat seafloor at the
            # depth measured at the first navigation sample.
            if nav.altitude > 0.0:
                self.seafloor = Seafloor([], [], flat=nav.position.depth + nav.altitude)
        if MISSION_TOPIC in first:
            self.xml = first[MISSION_TOPIC].data
            self.publish(self.xml)
        else:
            rospy.logwarn('mission_viz: no mission found at the start of %s', bag)
        rospy.Subscriber('mission', String, self.on_mission, queue_size=1)
        t = threading.Thread(target=self.load_seafloor, args=(bag,))
        t.daemon = True
        t.start()

    def load_seafloor(self, path):
        """Seafloor samples (north, east, depth+altitude) from the whole
        navigation; cached because indexing a large bag is slow."""
        st = os.stat(path)
        cache = os.path.join(os.path.expanduser('~/.ros/maris_dataviz_cache/seafloor'),
                             '%s_%d.npy' % (os.path.basename(path), st.st_size))
        if os.path.exists(cache):
            data = np.load(cache)
        else:
            rospy.loginfo('mission_viz: building seafloor model from %s (first run only)', path)
            rows = []
            with rosbag.Bag(path) as bag:
                topics = bag.get_type_and_topic_info().topics
                topic, decim = NAV_TOPIC + '_throttle', 1
                if topic not in topics:
                    topic, decim = NAV_TOPIC, 5
                for i, (_, nav, _) in enumerate(bag.read_messages(topics=[topic])):
                    if rospy.is_shutdown():
                        return
                    if i % decim == 0 and 0.0 < nav.altitude < 80.0:
                        rows.append((nav.position.north, nav.position.east, nav.position.depth + nav.altitude))
            data = np.array(rows).reshape(-1, 3)
            if not os.path.isdir(os.path.dirname(cache)):
                os.makedirs(os.path.dirname(cache))
            np.save(cache, data)
        if len(data) > 10:
            self.seafloor = Seafloor(data[:, :2], data[:, 2])
            if self.xml:
                self.publish(self.xml)

    def on_mission(self, msg):
        if self.origin is not None and msg.data != self.xml:
            self.xml = msg.data
            self.publish(msg.data)

    # --------------------------------------------------------------- parse
    def leg_z(self, n, e, depth, altitude, heave_mode):
        """World z of a point of the plan."""
        if heave_mode in (1, 2) and altitude is not None:
            floor = self.seafloor.depth_at(n, e)
            if floor is not None:
                d = floor - altitude
                if heave_mode == 2:
                    d = min(d, depth) if depth else d
                return -max(d, 0.0)
        return -(depth or 0.0)

    def parse(self, xml):
        root = ET.fromstring(xml)
        steps = []
        lat0, lon0 = self.origin
        for i, step in enumerate(root.findall('mission_step')):
            man = step.find('maneuver')
            if man is None:
                continue
            kind = man.get('type', 'goto')
            hm = int(ftext(man, 'heave_mode', 0) or 0)
            alt = ftext(man, 'final_altitude')
            fd = ftext(man, 'final_depth', 0.0)
            fn, fe, _ = geodetic_to_ned(ftext(man, 'final_latitude'), ftext(man, 'final_longitude'), lat0, lon0)
            s = {'idx': i, 'type': kind, 'final': (fn, fe), 'depth': fd, 'alt': alt, 'hm': hm,
                 'tol': ftext(man, 'tolerance_xy', 2.0), 'time': ftext(man, 'time'),
                 'actions': [a.findtext('action_name', '') for a in step.findall('actions_list/action')]}
            if man.find('initial_latitude') is not None:
                s['initial'] = geodetic_to_ned(ftext(man, 'initial_latitude'), ftext(man, 'initial_longitude'), lat0, lon0)[:2]
            steps.append(s)
        return steps

    def to_world(self, n, e, z):
        x, y, _ = ned_to_world(n, e, 0.0)
        return Point(x, y, z)

    def densify(self, a, b, step):
        a, b = np.asarray(a), np.asarray(b)
        n = max(2, int(np.linalg.norm(b - a) / 1.0) + 1)
        return [a + (b - a) * k / (n - 1) for k in range(n)]

    # ------------------------------------------------------------- markers
    def publish(self, xml):
        try:
            steps = self.parse(xml)
        except Exception as e:  # noqa
            rospy.logerr('mission_viz: cannot parse mission: %s', e)
            return
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        mid = [0]

        def new(mtype, ns):
            m = Marker()
            m.header.frame_id = self.frame
            m.ns, m.id, m.type = ns, mid[0], mtype
            m.pose.orientation.w = 1.0
            mid[0] += 1
            arr.markers.append(m)
            return m

        prev = None
        all_ne = []
        for s in steps:
            start = s.get('initial', prev) if s['type'] == 'section' else prev
            fin = s['final']
            all_ne.append(fin)
            c = COLORS.get(s['type'], ColorRGBA(0.8, 0.8, 0.8, 1.0))
            if start is not None:
                line = new(Marker.LINE_STRIP, 'legs')
                line.scale.x = self.line_width
                line.color = c
                for p in self.densify(start, fin, 1.0):
                    line.points.append(self.to_world(p[0], p[1], self.leg_z(p[0], p[1], s['depth'], s['alt'], s['hm'])))
                # Same leg projected on the surface (dimmed) to read the plan in 2D.
                ghost = new(Marker.LINE_STRIP, 'legs_surface')
                ghost.scale.x = self.line_width * 0.5
                ghost.color = ColorRGBA(c.r, c.g, c.b, 0.35)
                ghost.points = [self.to_world(start[0], start[1], 0.0), self.to_world(fin[0], fin[1], 0.0)]
                # Direction arrow at mid-leg.
                mid_ne = (np.asarray(start) + np.asarray(fin)) / 2.0
                d = np.asarray(fin) - np.asarray(start)
                if np.linalg.norm(d) > 1.0:
                    arrow = new(Marker.ARROW, 'direction')
                    z = self.leg_z(mid_ne[0], mid_ne[1], s['depth'], s['alt'], s['hm'])
                    u = d / np.linalg.norm(d) * 0.9
                    arrow.points = [self.to_world(mid_ne[0] - u[0], mid_ne[1] - u[1], z),
                                    self.to_world(mid_ne[0] + u[0], mid_ne[1] + u[1], z)]
                    arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.25, 0.6, 0.6
                    arrow.color = c

            z = self.leg_z(fin[0], fin[1], s['depth'], s['alt'], s['hm'])
            wp = new(Marker.SPHERE, 'waypoints')
            wp.pose.position = self.to_world(fin[0], fin[1], z)
            wp.scale.x = wp.scale.y = wp.scale.z = 0.6
            wp.color = c
            # Tolerance / park circle.
            if s['type'] == 'park' or s['tol']:
                ring = new(Marker.CYLINDER, 'tolerance')
                ring.pose.position = self.to_world(fin[0], fin[1], z)
                r = float(s['tol'] or 2.0)
                ring.scale.x = ring.scale.y = 2.0 * r
                ring.scale.z = 0.02
                ring.color = ColorRGBA(c.r, c.g, c.b, 0.15)
            # Vertical drop line from the surface to the waypoint.
            drop = new(Marker.LINE_LIST, 'drop_lines')
            drop.scale.x = 0.04
            drop.color = ColorRGBA(c.r, c.g, c.b, 0.4)
            drop.points = [self.to_world(fin[0], fin[1], 0.0), self.to_world(fin[0], fin[1], z)]
            label = new(Marker.TEXT_VIEW_FACING, 'labels')
            label.pose.position = self.to_world(fin[0], fin[1], z + 0.9)
            label.scale.z = 0.6
            label.color = ColorRGBA(1.0, 1.0, 1.0, 0.9)
            ref = ('alt %.1f m' % s['alt']) if s['hm'] in (1, 2) and s['alt'] is not None else ('depth %.1f m' % s['depth'])
            extra = (' %ds' % s['time']) if s['type'] == 'park' and s['time'] else ''
            label.text = '%d %s%s (%s)' % (s['idx'], s['type'], extra, ref)
            prev = fin

        if all_ne:
            ne = np.array(all_ne)
            c_ne = (ne.min(0) + ne.max(0)) / 2.0
            size = (ne.max(0) - ne.min(0)) + 2.0 * self.surface_size
            sea = new(Marker.CUBE, 'sea_surface')
            sea.pose.position = self.to_world(c_ne[0], c_ne[1], 0.0)
            sea.scale.x, sea.scale.y, sea.scale.z = size[0], size[1], 0.01
            sea.color = ColorRGBA(0.10, 0.45, 0.75, 0.12)
        self.pub.publish(arr)
        rospy.loginfo('mission_viz: %d mission steps published', len(steps))


if __name__ == '__main__':
    rospy.init_node('mission_viz')
    MissionViz()
    rospy.spin()
