#!/usr/bin/env python3
"""Navigation relay.

* Rebuilds the TF tree: world -> world_ned (static), the vehicle static
  sensor frames (read from /tf_static of the navigation bag) and
  world_ned -> base_link from the (de-duplicated) odometry.
* Publishes the vehicle model, the travelled trail, an altitude line to
  the seafloor and a telemetry label.
* Publishes the filtered altitude as sensor_msgs/Range for the projectors.
"""

import math

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Range
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from maris_dataviz.common import decode_any, main_bag, read_first_messages


def color(r, g, b, a=1.0):
    return ColorRGBA(r, g, b, a)


class NavRelay(object):
    def __init__(self):
        self.base_frame = rospy.get_param('~base_frame', 'sparus2/base_link')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.ned_frame = rospy.get_param('~ned_frame', 'world_ned')
        self.mesh = rospy.get_param('~vehicle_mesh', '')
        self.trail_step = rospy.get_param('~trail_step', 0.3)
        mission_dir = rospy.get_param('~mission_dir')

        self.tfb = tf2_ros.TransformBroadcaster()
        self.static_tfb = tf2_ros.StaticTransformBroadcaster()
        self.publish_static(mission_dir)

        self.pub_trail = rospy.Publisher('~trail', Path, queue_size=1, latch=True)
        self.pub_markers = rospy.Publisher('~vehicle', MarkerArray, queue_size=1)
        self.pub_alt = rospy.Publisher('~altitude', Range, queue_size=50)

        self.last_odom = rospy.Time(0)
        self.last_nav = rospy.Time(0)
        self.trail = Path()
        self.trail.header.frame_id = self.world_frame
        self.last_trail = None
        self.altitude = float('nan')
        self.depth = 0.0
        self.speed = 0.0
        self.last_marker_pub = rospy.Time(0)

        rospy.Subscriber('odometry', Odometry, self.on_odom, queue_size=200)
        rospy.Subscriber('navigation', rospy.AnyMsg, self.on_nav, queue_size=200)

    # ------------------------------------------------------------------ TF
    def publish_static(self, mission_dir):
        statics = []
        t = TransformStamped()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.ned_frame
        t.transform.rotation.x = 1.0  # 180 deg about X: NED -> z-up world
        statics.append(t)
        try:
            msg = read_first_messages(main_bag(mission_dir), ['/tf_static']).get('/tf_static')
            if msg is not None:
                statics.extend(msg.transforms)
        except Exception as e:  # noqa
            rospy.logwarn('nav_relay: could not read /tf_static from bag: %s', e)
        for s in statics:
            s.header.stamp = rospy.Time(0)
        self.static_tfb.sendTransform(statics)
        rospy.loginfo('nav_relay: published %d static transforms', len(statics))

    @staticmethod
    def _fresh(stamp, last):
        """True if stamp is newer than last; also accepts large backwards
        jumps (bag restarted / looped)."""
        return stamp > last or (last - stamp).to_sec() > 5.0

    # ----------------------------------------------------------- callbacks
    def on_nav(self, any_msg):
        try:
            nav = decode_any(any_msg)
        except Exception as e:  # noqa
            rospy.logwarn_throttle(10, 'nav_relay: cannot decode navigation: %s', e)
            return
        if not self._fresh(nav.header.stamp, self.last_nav):
            return
        self.last_nav = nav.header.stamp
        self.altitude = nav.altitude
        self.depth = nav.position.depth
        self.speed = math.hypot(nav.body_velocity.x, nav.body_velocity.y)
        if nav.altitude > 0.0:
            r = Range()
            r.header.stamp = nav.header.stamp
            r.header.frame_id = self.base_frame
            r.radiation_type = Range.ULTRASOUND
            r.min_range, r.max_range = 0.3, 100.0
            r.range = nav.altitude
            self.pub_alt.publish(r)

    def on_odom(self, odom):
        stamp = odom.header.stamp
        if not self._fresh(stamp, self.last_odom):
            return
        if stamp < self.last_odom:  # restart: clear trail
            self.trail.poses = []
            self.last_trail = None
        self.last_odom = stamp

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.ned_frame
        t.child_frame_id = self.base_frame
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
        t.transform.rotation = q
        self.tfb.sendTransform(t)

        world = np.array([p.x, -p.y, -p.z])  # NED -> world
        if self.last_trail is None or np.linalg.norm(world - self.last_trail) > self.trail_step:
            self.last_trail = world
            ps = PoseStamped()
            ps.header.frame_id = self.world_frame
            ps.header.stamp = stamp
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = world
            ps.pose.orientation.w = 1.0
            self.trail.poses.append(ps)
            self.trail.header.stamp = stamp
            self.pub_trail.publish(self.trail)

        if (stamp - self.last_marker_pub).to_sec() > 0.1 or stamp < self.last_marker_pub:
            self.last_marker_pub = stamp
            self.publish_markers(stamp)

    # ------------------------------------------------------------- markers
    def publish_markers(self, stamp):
        arr = MarkerArray()
        if self.mesh:
            m = Marker()
            m.header.frame_id = self.base_frame
            m.ns, m.id = 'vehicle', 0
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = self.mesh
            m.mesh_use_embedded_materials = True
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = color(1.0, 0.85, 0.1, 1.0)
            arr.markers.append(m)
        else:
            m = Marker()
            m.header.frame_id = self.base_frame
            m.ns, m.id = 'vehicle', 0
            m.type = Marker.ARROW
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = 1.6, 0.25, 0.25
            m.color = color(1.0, 0.85, 0.1)
            arr.markers.append(m)

        # Vertical line from the vehicle down to the seafloor (altitude).
        alt = self.altitude
        line = Marker()
        line.header.frame_id = self.base_frame
        line.ns, line.id = 'altitude', 0
        line.pose.orientation.w = 1.0
        if alt > 0.0 and alt < 100.0:
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.03
            line.color = color(0.3, 1.0, 0.9, 0.8)
            # base_link is NED-like (z down); ignore attitude, it is small.
            line.points = [Point(0, 0, 0), Point(0, 0, alt)]
            dot = Marker()
            dot.header.frame_id = self.base_frame
            dot.ns, dot.id = 'altitude', 1
            dot.type = Marker.CYLINDER
            dot.pose.position.z = alt
            dot.pose.orientation.w = 1.0
            dot.scale.x = dot.scale.y = 0.6
            dot.scale.z = 0.02
            dot.color = color(0.3, 1.0, 0.9, 0.5)
            arr.markers.append(dot)
        else:
            line.action = Marker.DELETE
        arr.markers.append(line)

        txt = Marker()
        txt.header.frame_id = self.base_frame
        txt.ns, txt.id = 'telemetry', 0
        txt.type = Marker.TEXT_VIEW_FACING
        txt.pose.position.z = -1.2
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.35
        txt.color = color(1.0, 1.0, 1.0, 0.95)
        alt_txt = ('%.1f m' % alt) if alt > 0.0 else 'n/a'
        txt.text = 'depth %.1f m | alt %s | %.2f m/s' % (self.depth, alt_txt, self.speed)
        arr.markers.append(txt)
        self.pub_markers.publish(arr)


if __name__ == '__main__':
    rospy.init_node('nav_relay')
    NavRelay()
    rospy.spin()
