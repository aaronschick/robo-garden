#!/usr/bin/env bash
# G2 — Scaler-drift audit for plan snazzy-forging-valiant.
#
# Clean A/B vs E12-override: match its config exactly except set
# --freeze-scaler-after-warmstart on (was auto), and narrow the det-eval
# rungs to a single easy ±8.5° / goal_dir=0° setting so the signal
# is maximally sensitive. If freezing the scaler prevents the ±17°
# collapse pattern (338 → -41), then scaler drift during eval rollouts
# was the accumulating env-state all along — proceed to G3 overnight run.
# If the eval trajectory still collapses, scaler is ruled out too →
# proceed to G4 (periodic-sim-rebuild probe).
#
# Env-step budget matches E12-override (300k @ 1024 envs) so the
# eval-1/eval-2 signal is directly comparable to the E12-override log.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="g2_scaler_drift_audit"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[g2] lr=1e-12 + freeze-scaler=on (vs E12-override's auto) + fixed-yaw det-eval"

export URCHIN_START_XY="-0.5,-0.5"
export URCHIN_GOAL_XY="0.5,0.5"
export URCHIN_EPISODE_S="15.0"

export URCHIN_GOAL_SAMPLING_MODE="arena"
export URCHIN_ARENA_HALF_EXTENT="2.0"
export URCHIN_MIN_SPAWN_GOAL_DIST="0.3"
export URCHIN_RESAMPLE_DIST_MARGIN="1.2"
export URCHIN_POTENTIAL_SCALE_M="1.5"
export URCHIN_DIST_SCALE_START="1.0"
export URCHIN_DIST_SCALE_END="1.0"

if [[ ! -f "${SEED_CKPT}" ]]; then
  echo "[g2] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[g2] run_id=${RUN_ID}"
echo "[g2] warmstart=${SEED_CKPT}"
echo "[g2] ckpt_dir=${CKPT_DIR}"
echo "[g2] num_envs=1024 timesteps=300000 lr=1e-12 bc_reg=0.0 freeze_scaler=on"
echo "[g2] det-eval single rung: yaw=0.297 goal_dir=0.0"

t_start=$(date +%s)
echo "[g2] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 300000 \
  --load-checkpoint "${SEED_CKPT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 1e-12 \
  --bc-reg-coef 0.0 \
  --freeze-scaler-after-warmstart on \
  --det-eval-yaw-spans-rad "0.297" \
  --det-eval-goal-dir-spans-rad "0.0"
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[g2] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
