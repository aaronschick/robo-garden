#!/usr/bin/env bash
# G0 — Ceiling measurement for plan snazzy-forging-valiant.
#
# Scores the contactpush oracle directly (policy residual head zeroed) at
# fixed yaw spans ±17°/±90°/±180° under the same arena-mode env vars the
# E-series used. Every later gate score is compared against this number.
#
# Pass if oracle ≥ 300 at ±17° → proceed to G1.
# Fail if oracle < 250 → stop; fix compute_contactpush_oracle first.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
EVAL="workspace/robots/urchin_v3/scripts/eval_yaw_response.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
OUT_JSON="workspace/eval/g0_oracle_ceiling.json"

echo "[g0] Oracle ceiling via eval_yaw_response.py --zero-residual"

export URCHIN_START_XY="-0.5,-0.5"
export URCHIN_GOAL_XY="0.5,0.5"
export URCHIN_EPISODE_S="15.0"
export URCHIN_GOAL_SAMPLING_MODE="arena"
export URCHIN_ARENA_HALF_EXTENT="2.0"
export URCHIN_POTENTIAL_SCALE_M="1.5"
export URCHIN_DIST_SCALE_START="1.0"
export URCHIN_DIST_SCALE_END="1.0"

if [[ ! -f "${SEED_CKPT}" ]]; then
  echo "[g0] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

mkdir -p "$(dirname "${OUT_JSON}")"

echo "[g0] checkpoint=${SEED_CKPT} (loaded but policy output zeroed)"
echo "[g0] rungs: yaw=[0.297, 1.5708, 3.14159] (±17° / ±90° / ±180°)"
echo "[g0] out=${OUT_JSON}"

t_start=$(date +%s)
echo "[g0] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${EVAL}" \
  --headless \
  --num-envs 64 \
  --checkpoint "${SEED_CKPT}" \
  --zero-residual \
  --yaw-spans-rad "0.297,1.5708,3.14159" \
  --episodes-per-rung 3 \
  --out "${OUT_JSON}"
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[g0] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
