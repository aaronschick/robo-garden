#!/usr/bin/env bash
# Render scripted-oracle clips in partial-oracle mode:
#   URCHIN_ORACLE_AMPLITUDE=0.5  (half-strength baked contactpush oracle)
#   URCHIN_RESIDUAL_SCALE_{INIT,FINAL}=1.0 (primitive residual at full weight)
#
# Rationale (from oracleoff review on 2026-04-24):
#  - oracle OFF + lazy 0.5x primitive => robot stiction-locks at 5cm
#  - engine clamps 42-D panel output to [-1,+1]; oracle at amp=1.0 already
#    fills that range, so primitive amp multipliers > 1.0 saturate.
#  - partial oracle (0.5) gives lazy a baseline to ride on AND leaves
#    headroom for snappy/accelerate residuals to show through.
#
# This is the test of whether the env's native oracle+residual
# architecture can produce visible style differentiation at all.

set -euo pipefail

PY=${PY:-C:/isaac-venv/Scripts/python.exe}
SCRIPT=workspace/robots/urchin_v3/scripts/scripted_roll_video.py
OUT_ROOT=workspace/checkpoints/scripted_oracle_review
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_ROOT}/v3_oraclehalf_${STAMP}"
mkdir -p "${OUT_DIR}"
echo "[oraclehalf_review] out dir: ${OUT_DIR}"

export URCHIN_ORACLE_AMPLITUDE=0.5
export URCHIN_RESIDUAL_SCALE_INIT=1.0
export URCHIN_RESIDUAL_SCALE_FINAL=1.0

render() {
    local prim="$1"; local style="$2"; local label="${prim}__${style}"
    echo "[oraclehalf_review] --- ${label} ---"
    "${PY}" "${SCRIPT}" \
        --primitive "${prim}" \
        --style "${style}" \
        --seconds 12.0 \
        --episodes 1 \
        --start-xy 0.0,0.0 \
        --goal-xy 3.0,0.0 \
        2>&1 | tee "${OUT_DIR}/${label}.log"
    latest=$(ls -td workspace/checkpoints/scripted_roll_video/v3_* 2>/dev/null | head -1)
    if [ -n "${latest}" ]; then
        mv "${latest}" "${OUT_DIR}/${label}"
        echo "[oraclehalf_review] moved ${latest} -> ${OUT_DIR}/${label}"
    fi
}

render straight_roll lazy
render straight_roll neutral
render straight_roll snappy
render accelerate    neutral

echo "[oraclehalf_review] done. videos under: ${OUT_DIR}"
ls -la "${OUT_DIR}"
