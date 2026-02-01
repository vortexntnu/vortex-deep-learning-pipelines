#!/usr/bin/env zsh
# Increase --radius only after traversing the entire axis in orbit_respawn.py
# Adapted to Python script that sleeps SLEEP_TIME seconds inside each tick.

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
BOUNCE_AXIS=true    # if true, Python does ping-pong on the axis

# How we define "running" along the axis:
#  - "to_end":    0 -> 1 (one pass P1->P2). Use BOUNCE_AXIS=false for clean passes.
#  - "roundtrip": 0 -> 1 -> 0 (go and back). Set BOUNCE_AXIS=true to visually match.
TRAVERSE_MODE="roundtrip"   # "to_end" | "roundtrip"

# This must match the constant in your Python (SLEEP_TIME = 2)
SLEEP_TIME=1    # seconds, internal sleep per tick in Python

# Optional small margin after a traverse (seconds), for capture safety
EXTRA_MARGIN=0.0

# Radius sweep
RADIUS_START=2.5
RADIUS_END=3.5
RADIUS_STEP=1
# ---------------------------------------------------

outline() { echo "[`date +%H:%M:%S`] $*"; }

cleanup_child() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    outline "Stopping PID $pid ..."
    kill -TERM "$pid" 2>/dev/null || true
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

# ---- Effective tick period due to Python's internal sleep ----
# Python: timer at 1/RATE but _tick() blocks SLEEP_TIME, so the real cadence is:
# T_tick = max(1/RATE, SLEEP_TIME)
T_TICK=$(awk -v r="$RATE" -v s="$SLEEP_TIME" 'BEGIN{t=1.0/r; if (s>t) t=s; print t}')

# ---- Effective axial speed and run time per traverse ----
# v_ax_eff = AXIAL_SPEED * ((1/RATE)/T_TICK)
V_AX_EFF=$(awk -v v="$AXIAL_SPEED" -v r="$RATE" -v t="$T_TICK" \
           'BEGIN{print v * ((1.0/r)/t)}')

if (( $(awk -v v="$V_AX_EFF" 'BEGIN{print (v<=0)}') )); then
  echo "Computed effective axial speed <= 0; check RATE and SLEEP_TIME." >&2
  exit 1
fi

# Base time for one pass P1->P2
T_TO_END=$(awk -v L="$L" -v v="$V_AX_EFF" 'BEGIN{print L/v}')

if [[ "$TRAVERSE_MODE" == "roundtrip" ]]; then
  RUN_SECONDS=$(awk -v t="$T_TO_END" -v m="$EXTRA_MARGIN" 'BEGIN{print 2*t + m}')
else
  RUN_SECONDS=$(awk -v t="$T_TO_END" -v m="$EXTRA_MARGIN" 'BEGIN{print t + m}')
fi

outline "Axis length L=${L} m | T_tick=${T_TICK}s | v_ax_eff=${V_AX_EFF} m/s | run≈${RUN_SECONDS}s (${TRAVERSE_MODE})"

# ---------------------- Main loop ----------------------
radius="$RADIUS_START"
while (( $(awk -v r="$radius" -v end="$RADIUS_END" 'BEGIN {print (r <= end)}') )); do
  outline "Launching at radius=${radius} m for ~${RUN_SECONDS}s (bounce_axis=${BOUNCE_AXIS})"

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
  $BOUNCE_AXIS && args+=(--bounce_axis)

  $PY "${args[@]}" &
  pid=$!

  trap 'outline "Interrupted"; cleanup_child "$pid"; exit 130' INT TERM

  # Wait exactly one full traverse (to_end or roundtrip) per the Python's real cadence
  sleep "$RUN_SECONDS" || true

  cleanup_child "$pid"

  radius=$(awk -v r="$radius" -v step="$RADIUS_STEP" 'BEGIN {print r+step}')
done

outline "Completed radius sweep from ${RADIUS_START} to ${RADIUS_END}."
