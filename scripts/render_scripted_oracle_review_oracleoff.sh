#!/usr/bin/env bash
# Render scripted-oracle clips in oracle-OFF mode (env's baked-in
# contactpush oracle disabled; primitive SH action drives the env).
#
# Follow-up to scripts/render_scripted_oracle_review.sh: the oracle-ON
# renders showed zero visible style differentiation, which we now
# understand is because the env adds the policy action as a *residual*
# on top of the baked oracle. This script tests whether disabling the
# oracle exposes the style + accelerate differences that the primitive
# library authors.

set -euo pipefail

PY=${PY:-C:/isaac-venv/Scripts/python.exe}
SCRIPT=workspace/robots/urchin_v3/scripts/scripted_roll_video.py
OUT_ROOT=workspace/checkpoints/scripted_oracle_review
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_ROOT}/v3_oracleoff_${STAMP}"
mkdir -p "${OUT_DIR}"
echo "[oracleoff_review] out dir: ${OUT_DIR}"

export URCHIN_ORACLE_AMPLITUDE=0.0
export URCHIN_RESIDUAL_SCALE_INIT=1.0
export URCHIN_RESIDUAL_SCALE_FINAL=1.0

render() {
    local prim="$1"; local style="$2"; local label="${prim}__${style}"
    echo "[oracleoff_review] --- ${label} ---"
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
        echo "[oracleoff_review] moved ${latest} -> ${OUT_DIR}/${label}"
    fi
}

render straight_roll lazy
render straight_roll neutral
render straight_roll snappy
render accelerate    neutral

echo "[oracleoff_review] done. videos under: ${OUT_DIR}"
ls -la "${OUT_DIR}"
