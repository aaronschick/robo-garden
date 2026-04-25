#!/usr/bin/env bash
# Smoke-render the momentum-gated arc primitives.
#
# Expected behavior: the first ~1 second of each clip should be visually
# identical across straight_roll, arc_left, arc_right (all :neutral).
# Once the blob accelerates past the ang_vel gate (~0.8 rad/s), arc_left
# and arc_right should diverge from straight_roll in opposite directions.
#
# Partial-oracle config (URCHIN_ORACLE_AMPLITUDE=0.5) matches the
# production Phase 3 BC dataset recording preset so the visible rolling
# physics is comparable to prior oracle reviews.

set -euo pipefail

PY=${PY:-C:/isaac-venv/Scripts/python.exe}
SCRIPT=workspace/robots/urchin_v3/scripts/scripted_roll_video.py
OUT_ROOT=workspace/checkpoints/gate_smoke
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT_DIR}"
echo "[gate_smoke] out dir: ${OUT_DIR}"

export URCHIN_ORACLE_AMPLITUDE=0.5
export URCHIN_RESIDUAL_SCALE_INIT=1.0
export URCHIN_RESIDUAL_SCALE_FINAL=1.0

render() {
    local prim="$1"; local style="$2"; local label="${prim}__${style}"
    echo "[gate_smoke] --- ${label} ---"
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
        echo "[gate_smoke] moved ${latest} -> ${OUT_DIR}/${label}"
    fi
}

render straight_roll neutral
render arc_left      neutral
render arc_right     neutral

echo "[gate_smoke] done. videos under: ${OUT_DIR}"
ls -la "${OUT_DIR}"
