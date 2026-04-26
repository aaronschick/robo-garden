#!/usr/bin/env bash
# Render scripted-oracle clips for the 4 combos that failed BC review:
#   straight_roll:{lazy,neutral,snappy}  (F1: style differentiation)
#   accelerate:neutral                   (F2: visible ramp)
#
# This is the user-gate before re-recording the BC dataset with the
# new STYLES + accelerate numbers. If the 4 oracle clips are visibly
# differentiated, proceed to re-record; if they still look flat, the
# new numbers need more tuning before any pipeline work.

set -euo pipefail

PY=${PY:-C:/isaac-venv/Scripts/python.exe}
SCRIPT=workspace/robots/urchin_v3/scripts/scripted_roll_video.py
OUT_ROOT=workspace/checkpoints/scripted_oracle_review
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_ROOT}/v3_oracle_${STAMP}"
mkdir -p "${OUT_DIR}"
echo "[oracle_review] out dir: ${OUT_DIR}"

render() {
    local prim="$1"; local style="$2"; local label="${prim}__${style}"
    echo "[oracle_review] --- ${label} ---"
    "${PY}" "${SCRIPT}" \
        --primitive "${prim}" \
        --style "${style}" \
        --seconds 12.0 \
        --episodes 1 \
        --start-xy 0.0,0.0 \
        --goal-xy 3.0,0.0 \
        2>&1 | tee "${OUT_DIR}/${label}.log"
    # The script writes into workspace/checkpoints/scripted_roll_video/v3_<ts>/;
    # move the latest such dir under ours with a clean label.
    latest=$(ls -td workspace/checkpoints/scripted_roll_video/v3_* 2>/dev/null | head -1)
    if [ -n "${latest}" ]; then
        mv "${latest}" "${OUT_DIR}/${label}"
        echo "[oracle_review] moved ${latest} -> ${OUT_DIR}/${label}"
    fi
}

render straight_roll lazy
render straight_roll neutral
render straight_roll snappy
render accelerate    neutral

echo "[oracle_review] done. videos under: ${OUT_DIR}"
ls -la "${OUT_DIR}"
