#!/usr/bin/env python3
"""Play every bag of a mission folder with a single `rosbag play --clock`,
forwarding only the topics used by the visualization.

With --skip-to-data, the part before the first sensor message (transit,
diving, parking) is played at --ff-rate with navigation only, and the
playback switches to --rate when data collection starts.

/tf and /tf_static are not replayed: every sensor bag carries its own copy
of the navigation TF, and /tf_static would be lost when starting with an
offset. nav_relay.py rebuilds both from the navigation bag instead.
"""

import argparse
import os
import signal
import subprocess
import sys

import rospy

from maris_dataviz.common import bag_time_info, find_mission_bags, motion_start_offset

NS = '/sparus2'
NAV_TOPICS = [
    NS + '/navigator/odometry',
    NS + '/navigator/navigation',
    NS + '/captain/mission',
    NS + '/captain/trajectory_path',
    NS + '/pilot/waypoint_marker',
]
SSS = NS + '/marinesonic_scoutmkii_sidescan/raw_data/'
SSS_TOPICS = [SSS + 'port', SSS + 'starboard', SSS + 'port/sss_info', SSS + 'starboard/sss_info']
MB_TOPICS = [NS + '/norbit_wbms_multibeam/multibeam_scan']
CAM = NS + '/flir_spinnaker_stereo_camera/'
CAM_TOPICS = [CAM + side + suffix for side in ('port', 'starboard')
              for suffix in ('/image_raw/compressed', '/camera_info')]


def str2bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'on')


def main():
    argv = rospy.myargv(sys.argv)
    ros_remaps = [a for a in sys.argv[1:] if a.startswith('__') or ':=' in a]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mission_dir')
    p.add_argument('--rate', type=float, default=1.0)
    p.add_argument('--start', type=float, default=0.0, help='seconds to skip')
    p.add_argument('--duration', type=float, default=0.0, help='0 = until the end')
    p.add_argument('--delay', type=float, default=3.0, help='seconds to wait after advertising')
    p.add_argument('--sss', default='true')
    p.add_argument('--multibeam', default='true')
    p.add_argument('--camera', default='true')
    p.add_argument('--loop', default='false')
    p.add_argument('--pause', default='false')
    p.add_argument('--skip-to-data', default='true',
                   help='fast-forward (navigation only) until the first sensor message')
    p.add_argument('--ff-rate', type=float, default=20.0)
    p.add_argument('--data-margin', type=float, default=5.0,
                   help='seconds of normal-rate playback before the first sensor message')
    p.add_argument('--wait-motion', default='true',
                   help='also keep fast-forwarding until the vehicle starts moving (from *_latlon.csv)')
    a = p.parse_args(argv[1:])

    bags = find_mission_bags(a.mission_dir)
    if not bags['main']:
        sys.exit('play_mission: no navigation bag found in %s' % a.mission_dir)

    files = list(bags['main'])
    topics = list(NAV_TOPICS)
    sensors = []  # (bag files, topic substring) used to detect data start
    if str2bool(a.sss) and bags['sidescan']:
        files += bags['sidescan']
        topics += SSS_TOPICS
        sensors.append((bags['sidescan'], '/raw_data/port'))
    if str2bool(a.multibeam) and bags['multibeam']:
        files += bags['multibeam']
        topics += MB_TOPICS
        sensors.append((bags['multibeam'], 'multibeam_scan'))
    if str2bool(a.camera) and bags['stereo_camera']:
        files += bags['stereo_camera']
        topics += CAM_TOPICS
        sensors.append((bags['stereo_camera'], '/image_raw/compressed'))

    # Offsets are relative to the earliest message of all played bags.
    starts, data_t = [], []
    for f in files:
        starts.append(bag_time_info(f)[0])
    for flist, needle in sensors:
        for f in flist:
            t = bag_time_info(f, lambda tp, n=needle: n in tp)[1]
            if t is not None:
                data_t.append(t)
                break
    t0 = min(t for t in starts if t is not None)
    data_abs = min(data_t) if data_t else None
    if str2bool(a.wait_motion):
        moving = motion_start_offset(a.mission_dir)
        if moving is not None and starts[0] is not None:
            # The CSV starts with the navigation bag.
            data_abs = max(data_abs or 0.0, starts[0] + moving)
    data_off = (data_abs - t0 - a.data_margin) if data_abs else None

    common = ['--clock', '-q']
    phases = []
    if str2bool(a.skip_to_data) and data_off is not None and data_off - a.start > 2.0:
        # Phase 1 plays only the navigation bag: its offset is relative to its own start.
        ff_start = a.start + (t0 - starts[0])
        ff = ['rosbag', 'play'] + common + ['-r', str(a.ff_rate), '-d', str(a.delay), '-s', str(ff_start),
                                            '-u', str(data_off - a.start)]
        phases.append(('fast-forward x%g until data collection starts (+%.0f s)' % (a.ff_rate, data_off),
                       ff + bags['main'] + ['--topics'] + NAV_TOPICS + ros_remaps))
        start, delay = data_off, 0.2
    else:
        start, delay = a.start, a.delay
    cmd = ['rosbag', 'play'] + common + ['-r', str(a.rate), '-d', str(delay), '-s', str(start)]
    if a.duration > 0:
        cmd += ['-u', str(a.duration)]
    if str2bool(a.loop):
        cmd.append('-l')
    if str2bool(a.pause):
        cmd.append('--pause')
    phases.append(('playing %d bag(s) at x%g from +%.0f s:\n  %s' % (len(files), a.rate, start, '\n  '.join(files)),
                   cmd + files + ['--topics'] + topics + ros_remaps))

    child = [None]

    def stop(signum, _frame):
        if child[0] is not None and child[0].poll() is None:
            child[0].send_signal(signal.SIGINT)
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    for text, c in phases:
        print('play_mission: ' + text)
        sys.stdout.flush()
        child[0] = subprocess.Popen(c)
        child[0].wait()
    print('play_mission: playback finished')


if __name__ == '__main__':
    main()
