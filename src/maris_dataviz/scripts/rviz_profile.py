#!/usr/bin/env python3
"""Start RViz with the configuration that fits the mission data.

Profiles: 'sss' (side-scan mosaic + waterfall, wide view), 'optical'
(photo mosaic + bathymetry + camera image, close view) and 'full'
(everything). With 'auto' the profile is detected from the topics recorded
in the mission bags.
"""

import os
import sys

import rospkg
import rospy

from maris_dataviz.common import detect_profile


def main():
    args = rospy.myargv(sys.argv)
    if len(args) < 2:
        sys.exit('usage: rviz_profile.py MISSION_DIR [PROFILE] [RVIZ_CONFIG]')
    mission_dir = args[1]
    profile = args[2] if len(args) > 2 else 'auto'
    config = args[3] if len(args) > 3 else ''
    if profile == 'auto':
        profile = detect_profile(mission_dir)
        if profile == 'nav':
            profile = 'full'
    if not config:
        config = os.path.join(rospkg.RosPack().get_path('maris_dataviz'), 'config', profile + '.rviz')
    print('rviz_profile: profile "%s" -> %s' % (profile, config))
    sys.stdout.flush()
    remaps = [a for a in sys.argv[1:] if a.startswith('__')]
    os.execvp('rosrun', ['rosrun', 'rviz', 'rviz', '-d', config] + remaps)


if __name__ == '__main__':
    main()
