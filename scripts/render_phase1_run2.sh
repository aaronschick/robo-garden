#!/usr/bin/env bash
# Post-chain spot render for Phase 1 run2 final_checkpoint.pt. Six bearings
# evenly spaced across 360 deg on the 1.41 m goal ring around start
# (-0.5, -0.5), plus a regression render at the Phase 0 baseline goal
# (+0.5, +0.5). Forces span_start=span_end=0 so each episode resamples
# to the same deterministic bearing.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
RENDER="workspace/robots/urchin_v3/scripts/render_policy_video.py"
CKPT="workspace/checkpoints/urchin_v3_pathB_phase1_run2/final_checkpoint.pt"

export URCHIN_GOAL_DIR_SPAN_START_DEG="0"
export URCHIN_GOAL_DIR_SPAN_END_DEG="0"
export URCHIN_GOAL_DIR_ANNEAL_START="0"
export URCHIN_GOAL_DIR_ANNEAL_END="1"

# (bearing_deg, goal_x, goal_y, tag)
# Six bearings evenly across 360 deg + Phase 0 regression goal (+0.5,+0.5) at 45 deg.
CASES=(
  "0:0.91:-0.5:phase1_run2_bearing000"
  "60:0.205:0.721:phase1_run2_bearing060"
  "120:-1.205:0.721:phase1_run2_bearing120"
  "180:-1.91:-0.5:phase1_run2_bearing180"
  "240:-1.205:-1.721:phase1_run2_bearing240"
  "300:0.205:-1.721:phase1_run2_bearing300"
  "45:0.5:0.5:phase1_run2_regression_phase0goal"
)

for entry in "${CASES[@]}"; do
  IFS=':' read -r bearing gx gy tag <<< "${entry}"
  echo "[render] === bearing ${bearing} deg  goal=(${gx},${gy}) ==="
  URCHIN_GOAL_DIR_BASE_DEG="${bearing}" \
  "${PY}" "${RENDER}" \
    --checkpoint "${CKPT}" \
    --goal-xy="${gx},${gy}" \
    --start-xy=-0.5,-0.5 \
    --seconds 8.0 \
    --episodes 2 \
    --tag "${tag}"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[render] ABORT: bearing ${bearing} exit=${rc}"
    exit 1
  fi
done

echo "[render] ALL BEARINGS RENDERED"
