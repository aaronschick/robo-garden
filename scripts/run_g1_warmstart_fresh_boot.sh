#!/usr/bin/env bash
# G1 — Warmstart fresh-boot sanity for plan snazzy-forging-valiant.
#
# Scores the approved Phase-1 warmstart (pathB_phase2b_run2/final) on a
# fresh Python process, arena-mode E-series env vars, fixed yaw spans
# ±17°/±90°/±180°. Confirms the warmstart hasn't rotted since the prior
# 318-at-±17° / 278-at-±180° diagnostic.
#
# Pass ≥ 300 at ±17° → proceed to G2.
# 150–300 → G2 with caveat.
# < 150 → skip troubleshooting, go P1 (pure-BC pivot).

set -u

PY="C:/isaac-venv/Scripts/python.exe"
EVAL="workspace/robots/urchin_v3/scripts/eval_yaw_response.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
OUT_JSON="workspace/eval/g1_warmstart_fresh_boot.json"

echo "[g1] Warmstart fresh-boot score via eval_yaw_response.py (no zero-residual)"

export URCHIN_START_XY="-0.5,-0.5"
export URCHIN_GOAL_XY="0.5,0.5"
export URCHIN_EPISODE_S="15.0"
export URCHIN_GOAL_SAMPLING_MODE="arena"
export URCHIN_ARENA_HALF_EXTENT="2.0"
export URCHIN_POTENTIAL_SCALE_M="1.5"
export URCHIN_DIST_SCALE_START="1.0"
export URCHIN_DIST_SCALE_END="1.0"

if [[ ! -f "${SEED_CKPT}" ]]; then
  echo "[g1] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

mkdir -p "$(dirname "${OUT_JSON}")"

echo "[g1] checkpoint=${SEED_CKPT}"
echo "[g1] rungs: yaw=[0.297, 1.5708, 3.14159] (±17° / ±90° / ±180°)"
echo "[g1] out=${OUT_JSON}"

t_start=$(date +%s)
echo "[g1] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${EVAL}" \
  --headless \
  --num-envs 64 \
  --checkpoint "${SEED_CKPT}" \
  --yaw-spans-rad "0.297,1.5708,3.14159" \
  --episodes-per-rung 3 \
  --out "${OUT_JSON}"
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[g1] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
