#!/usr/bin/env bash
# W8 step 2: re-render scripted-oracle clips after retuning STYLES +
# primitive defaults onto the W6 sample-17 basin.
#
# Pre-W8 the styles were multipliers on engine-default amps (oracle@1.0)
# which the W6 256-sample atlas showed sit outside the success basin.
# The new STYLES are absolute-value replacements anchored on W6 samples
# 17 (neutral), 28/40 (snappy), and 86 (lazy) -- inside the basin.
#
# Combos rendered (per plan W8 step 2):
#   straight_roll:neutral   -> canonical W6 sample-17 baseline
#   arc_left:snappy         -> momentum-gated arc on the snappy basin
#   accelerate:neutral      -> visible ramp on the W6 baseline
#   wobble_idle:neutral     -> idle / non-rolling reference
#
# Substrate hygiene: URCHIN_RESET_MODE=canonical (W1).
#
# Output: workspace/checkpoints/scripted_oracle_review/v3_w8_<ts>/

set -euo pipefail

PY=${PY:-C:/isaac-venv/Scripts/python.exe}
SCRIPT=workspace/robots/urchin_v3/scripts/scripted_roll_video.py
OUT_ROOT=workspace/checkpoints/scripted_oracle_review
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_ROOT}/v3_w8_${STAMP}"
mkdir -p "${OUT_DIR}"
echo "[w8_oracle_review] out dir: ${OUT_DIR}"

# W1 substrate hygiene.
export URCHIN_RESET_MODE=canonical

# Mixed-mode oracle: rolling primitives need the env's baked-in oracle ON
# (that's the deployment regime — policy is a residual on top of the oracle,
# the W6 atlas measured 0.81 m for sample 17 in this configuration), but
# wobble_idle needs the oracle OFF (otherwise its zero-amp breathing pulse
# inherits the oracle's forward push and visibly drifts).
export URCHIN_RESIDUAL_SCALE_INIT=1.0
export URCHIN_RESIDUAL_SCALE_FINAL=1.0

render() {
    local prim="$1"; local style="$2"; local oracle_amp="$3"
    local label="${prim}__${style}"
    echo "[w8_oracle_review] --- ${label} (oracle_amp=${oracle_amp}) ---"
    URCHIN_ORACLE_AMPLITUDE="${oracle_amp}" "${PY}" "${SCRIPT}" \
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
        echo "[w8_oracle_review] moved ${latest} -> ${OUT_DIR}/${label}"
    fi
}

# Rolling primitives: oracle ON (deployment regime).
render straight_roll neutral 1.0
render arc_left      snappy  1.0
render accelerate    neutral 1.0
# Idle primitive: oracle OFF (so the breathing pulse is direction-free).
render wobble_idle   neutral 0.0

echo "[w8_oracle_review] done. videos under: ${OUT_DIR}"
ls -la "${OUT_DIR}"
