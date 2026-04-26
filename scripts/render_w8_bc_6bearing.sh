#!/usr/bin/env bash
# W8 final gate: render the BC checkpoint at 6 bearings around the spawn,
# matching the Phase 1 chain approval pattern (memory:
# project_urchin_v3_phase1_chain_approved.md). Filters to
# straight_roll:neutral so the user can verify the policy reaches each
# goal under the canonical primitive without conditioning noise.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
RENDER="workspace/robots/urchin_v3/scripts/bc_rollout_video.py"
CKPT="workspace/checkpoints/urchin_v3_bc_phase5_concat/model.pt"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="workspace/checkpoints/urchin_v3_bc_phase5_concat/bearing_review_${STAMP}"

# W1 substrate hygiene + deployment regime.
export URCHIN_RESET_MODE=canonical
export URCHIN_ORACLE_AMPLITUDE=1.0

# Spawn at (-0.5, -0.5) like Phase 1 chain. Six goals evenly spaced
# 60 deg apart on a 1.41 m ring around the spawn.
START_XY="-0.5,-0.5"
CASES=(
  "0:0.91:-0.5"
  "60:0.205:0.721"
  "120:-1.205:0.721"
  "180:-1.91:-0.5"
  "240:-1.205:-1.721"
  "300:0.205:-1.721"
)

for entry in "${CASES[@]}"; do
  IFS=':' read -r bearing gx gy <<< "${entry}"
  out_dir="${OUT_BASE}/bearing${bearing}"
  mkdir -p "${out_dir}"
  echo "[w8_bearing] === bearing ${bearing} deg  goal=(${gx},${gy}) ==="
  "${PY}" "${RENDER}" \
    --checkpoint "${CKPT}" \
    --out-dir "${out_dir}" \
    --start-xy="${START_XY}" \
    --goal-xy="${gx},${gy}" \
    --combos straight_roll:neutral \
    --seconds 12.0
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[w8_bearing] ABORT: bearing ${bearing} exit=${rc}"
    exit 1
  fi
done

echo "[w8_bearing] ALL 6 BEARINGS RENDERED -> ${OUT_BASE}"
ls -la "${OUT_BASE}"/bearing*/
