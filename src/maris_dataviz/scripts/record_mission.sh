#!/usr/bin/env bash
# Record the RViz replay of a mission to a video file.
#
# RViz runs in a nested X server (Xephyr) on its own display and ROS master,
# so the recording is not affected by other windows and does not interfere
# with other ROS sessions. The bags are played at REC_RATE (slower than real
# time gives RViz more rendered frames per mission second with software
# OpenGL); the final video is re-timed to real mission speed.
#
# Usage:
#   record_mission.sh OUTPUT.mp4 PROFILE [replay_mission.launch args...]
#   PROFILE: sss | optical | full
# Example:
#   record_mission.sh ~/videos/sss.mp4 sss dataset:=sss
#
# Environment (defaults): REC_DISPLAY=:5 REC_PORT=11313 REC_SIZE=1920x1080
#   REC_FPS=30 REC_RATE=0.5 REC_CRF=14 REC_TAIL=8 (seconds kept after the end)

set -eo pipefail

OUT=$(realpath -m "$1"); PROFILE=$2; shift 2 || { echo "usage: $0 OUTPUT.mp4 PROFILE [launch args]"; exit 1; }
DISP=${REC_DISPLAY:-:5}
PORT=${REC_PORT:-11313}
SIZE=${REC_SIZE:-1920x1080}
FPS=${REC_FPS:-30}
RATE=${REC_RATE:-0.5}
CRF=${REC_CRF:-14}
TAIL=${REC_TAIL:-8}
W=${SIZE%x*}; H=${SIZE#*x}

WORK=$(mktemp -d)
RAW="$WORK/raw.mkv"
LOG="$WORK/replay.log"
mkdir -p "$(dirname "$OUT")"
PIDS=()

cleanup() {
  for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done
  sleep 3
  for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
}
trap cleanup EXIT

# Nested X server (software OpenGL through Mesa).
if ! DISPLAY=$DISP xset q >/dev/null 2>&1; then
  Xephyr "$DISP" -screen "$SIZE" -br -ac -nolisten tcp >"$WORK/xephyr.log" 2>&1 &
  PIDS+=($!)
  sleep 2
fi
# Disable screen blanking (the default X screensaver blanks after 10 min).
DISPLAY=$DISP xset s off s noblank -dpms 2>/dev/null || true

# Full-screen RViz configuration for the video: no side panel, no labels.
CFG="$WORK/record.rviz"
python3 - "$(rospack find maris_dataviz)/config/$PROFILE.rviz" "$CFG" "$W" "$H" <<'EOF'
import sys, yaml
src, dst, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c = yaml.safe_load(open(src))
c['Window Geometry'].update({'X': 0, 'Y': 0, 'Width': w, 'Height': h})
# Only the 3D view and the image panels: drop the Displays and Time panels.
c['Panels'] = [p for p in c['Panels'] if p['Class'] not in ('rviz/Displays', 'rviz/Time')]
def walk(displays):
    for d in displays:
        if d.get('Name') == 'Mission plan':
            d['Namespaces'] = {'labels': False}
        walk(d.get('Displays', []))
walk(c['Visualization Manager']['Displays'])
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
EOF

export ROS_MASTER_URI=http://localhost:$PORT DISPLAY=$DISP DISABLE_ROS1_EOL_WARNINGS=1
roslaunch -p "$PORT" maris_dataviz replay_mission.launch rate:="$RATE" rviz_config:="$CFG" "$@" >"$LOG" 2>&1 &
LAUNCH=$!
PIDS+=($LAUNCH)
sleep 12  # RViz start-up

ffmpeg -hide_banner -loglevel error -y -f x11grab -draw_mouse 0 -video_size "$SIZE" -framerate "$FPS" \
  -i "$DISP.0" -c:v libx264 -preset ultrafast -crf 0 "$RAW" &
FFMPEG=$!
echo "record_mission: recording $PROFILE replay (rate $RATE) on $DISP -> $OUT"

until grep -q 'play_mission: playback finished' "$LOG"; do
  kill -0 "$LAUNCH" 2>/dev/null || { echo "record_mission: replay exited early, see $LOG"; break; }
  sleep 5
done
sleep "$TAIL"
kill -INT "$FFMPEG"; wait "$FFMPEG" || true
kill -INT "$LAUNCH" 2>/dev/null || true

echo "record_mission: encoding final video"
SPEED=$(python3 -c "print(1.0 / float('$RATE'))")
ffmpeg -hide_banner -loglevel error -y -i "$RAW" -vf "setpts=PTS/$SPEED" -r "$FPS" \
  -c:v libx264 -preset slow -crf "$CRF" -pix_fmt yuv420p -movflags +faststart "$OUT"
rm -f "$RAW"
echo "record_mission: done -> $OUT ($(du -h "$OUT" | cut -f1))"
