#!/usr/bin/env bash
# Spot-render the Phase 1 smoke checkpoint at three bearings spanning the
# 30->90 deg smoke curriculum (base +45 deg). Forces span_start=span_end=0
# so each episode resamples to the same deterministic bearing. Start stays
# at (-0.5,-0.5); goal sits on a 1.41m ring at the commanded bearing.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
RENDER="workspace/robots/urchin_v3/scripts/render_policy_video.py"
CKPT="workspace/checkpoints/urchin_v3_pathB_phase1_smoke/final_checkpoint.pt"

export URCHIN_GOAL_DIR_SPAN_START_DEG="0"
export URCHIN_GOAL_DIR_SPAN_END_DEG="0"
export URCHIN_GOAL_DIR_ANNEAL_START="0"
export URCHIN_GOAL_DIR_ANNEAL_END="1"

# (bearing_deg, goal_x, goal_y, tag)
CASES=(
  "0:0.91:-0.5:phase1_smoke_bearing000"
  "45:0.5:0.5:phase1_smoke_bearing045"
  "90:-0.5:0.91:phase1_smoke_bearing090"
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
