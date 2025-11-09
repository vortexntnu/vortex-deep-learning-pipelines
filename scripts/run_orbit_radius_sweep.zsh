#!/usr/bin/env zsh
# Increase --radius only after traversing the entire axis in orbit_respawn.py

set -euo pipefail

# ---------------------- CONFIG ----------------------
PY=python3
SCRIPT="orbit_respawn.py"

NAME="camera_rig"
SERVICE="/stonefish_ros2/respawn_robot"
# pipeline points
P1=("1.85" "0.02" "5.53")
P2=("2.20" "-14.02" "7.49")

RATE=5.0            # Hz -> same as --rate in Python
SPEED=0.3           # rad/s (not critical for axis timing)
AXIAL_SPEED=0.2     # m/s -> same as --axial_speed
BOUNCE_AXIS=true   # if true, Python does ping-pong on the axis

# How we define "running" along the axis:
#  - "to_end":    0 -> 1 (one pass P1->P2). Use BOUNCE_AXIS=false for clean passes.
#  - "roundtrip": 0 -> 1 -> 0 (go and back). Set BOUNCE_AXIS=true to visually match.
TRAVERSE_MODE="roundtrip"   # "to_end" | "roundtrip"

# Sleep used INSIDE Python per tick (must match your time.sleep(..) in _tick)
SLEEP_TIME=2.0

# Optional small margin after a traverse (seconds), for capture safety
EXTRA_MARGIN=0.0

# Radius sweep
RADIUS_START=0.5
RADIUS_END=6.0
RADIUS_STEP=0.25
# ---------------------------------------------------

outline() { echo "[`date +%H:%M:%S`] $*"; }

cleanup_child() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    outline "Stopping PID $pid ..."
    kill -TERM "$pid" 2>/dev/null || true
    # wait up to ~2s
    for i in {1..20}; do
      sleep 0.1
      kill -0 "$pid" 2>/dev/null || return 0
    done
    outline "Force killing PID $pid"
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

# ---- Compute axis length L = ||P2 - P1|| ----
L=$(awk -v x1="${P1[1]}" -v y1="${P1[2]}" -v z1="${P1[3]}" \
        -v x2="${P2[1]}" -v y2="${P2[2]}" -v z2="${P2[3]}" \
        'BEGIN{dx=x2-x1; dy=y2-y1; dz=z2-z1; print sqrt(dx*dx+dy*dy+dz*dz)}')

if [[ "$L" == "0" || "$L" == "0.0" ]]; then
  echo "Axis length is zero; check P1/P2." >&2
  exit 1
fi

# ---- Compute normalized dt per tick: dt = (AXIAL_SPEED / RATE) / L ----
DT_NORM=$(awk -v v="$AXIAL_SPEED" -v r="$RATE" -v l="$L" \
           'BEGIN{ if (r<=0 || l<=0) {print -1; exit}; print (v/r)/l }')

if (( $(awk -v d="$DT_NORM" 'BEGIN{print (d<=0)}') )); then
  echo "Invalid dt: check RATE and AXIAL_SPEED." >&2
  exit 1
fi

# Ticks to traverse to_end = ceil(1/dt); for roundtrip, ×2
TICKS_TO_END=$(awk -v d="$DT_NORM" 'BEGIN{print int((1.0/d)+0.999999)}')
if [[ "$TRAVERSE_MODE" == "roundtrip" ]]; then
  TOTAL_TICKS=$(( TICKS_TO_END * 2 ))
else
  TOTAL_TICKS=$TICKS_TO_END
fi

RUN_SECONDS=$(awk -v ticks="$TOTAL_TICKS" -v s="$SLEEP_TIME" -v m="$EXTRA_MARGIN" \
              'BEGIN{print ticks*s + m}')

outline "Axis length L=${L} m | dt=${DT_NORM} per tick | ticks/traverse=${TOTAL_TICKS} | run≈${RUN_SECONDS}s"

# ---------------------- Main loop ----------------------
radius="$RADIUS_START"
while (( $(awk -v r="$radius" -v end="$RADIUS_END" 'BEGIN {print (r <= end)}') )); do
  outline "Launching at radius=${radius} m for ~${RUN_SECONDS}s (mode=${TRAVERSE_MODE}, bounce_axis=${BOUNCE_AXIS})"

  args=(
    "$SCRIPT"
    --service "$SERVICE"
    --name "$NAME"
    --p1 "${P1[@]}"
    --p2 "${P2[@]}"
    --radius "$radius"
    --speed "$SPEED"
    --rate "$RATE"
    --axial_speed "$AXIAL_SPEED"
  )
  if $BOUNCE_AXIS; then
    args+=(--bounce_axis)
  fi

  $PY "${args[@]}" &
  pid=$!

  trap 'outline "Interrupted"; cleanup_child "$pid"; exit 130' INT TERM

  # Wait exactly one full traverse of the axis (to_end or roundtrip)
  sleep "$RUN_SECONDS" || true

  cleanup_child "$pid"

  radius=$(awk -v r="$radius" -v step="$RADIUS_STEP" 'BEGIN {print r+step}')
done

outline "Completed radius sweep from ${RADIUS_START} to ${RADIUS_END}."
