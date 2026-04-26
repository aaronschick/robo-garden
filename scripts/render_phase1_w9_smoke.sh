#!/usr/bin/env bash
# Six-bearing render of the W9 Phase 1 smoke checkpoint
# (warmstarted from W8 BC concat-MLP, 148-D conditioned input).
#
# Same 6 bearings + Phase-0 regression goal as render_phase1_run2.sh.
# Conditioning matches the chain script: straight_roll:neutral, fixed.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
RENDER="workspace/robots/urchin_v3/scripts/render_policy_video.py"
CKPT="workspace/checkpoints/urchin_v3_w9_phase1_smoke/final_checkpoint.pt"

# W1 hygiene + W9 conditioning consumed by the conditioning wrapper in
# train.py. render_policy_video.py reads cfg.observation_space at module
# import; its current implementation does not yet apply the conditioning
# wrapper, so render_phase1_w9_smoke.sh assumes render_policy_video.py
# was patched with the same wrapper, OR train.py's checkpoint stores
# the 148-D Policy weights and the renderer's policy_space falls back to
# 148. (See note below — if the render fails on shape mismatch, patch
# render_policy_video.py mirroring the wrapper in train.py.)
export URCHIN_RESET_MODE="canonical"
export URCHIN_PRIMITIVE_ID="straight_roll"
export URCHIN_STYLE_ID="neutral"

export URCHIN_GOAL_DIR_SPAN_START_DEG="0"
export URCHIN_GOAL_DIR_SPAN_END_DEG="0"
export URCHIN_GOAL_DIR_ANNEAL_START="0"
export URCHIN_GOAL_DIR_ANNEAL_END="1"

# (bearing_deg, goal_x, goal_y, tag)
CASES=(
  "0:0.91:-0.5:w9_phase1_smoke_bearing000"
  "60:0.205:0.721:w9_phase1_smoke_bearing060"
  "120:-1.205:0.721:w9_phase1_smoke_bearing120"
  "180:-1.91:-0.5:w9_phase1_smoke_bearing180"
  "240:-1.205:-1.721:w9_phase1_smoke_bearing240"
  "300:0.205:-1.721:w9_phase1_smoke_bearing300"
  "45:0.5:0.5:w9_phase1_smoke_regression_phase0goal"
)

if [[ ! -f "${CKPT}" ]]; then
  echo "[render] ABORT: checkpoint not found: ${CKPT}"
  exit 1
fi

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
