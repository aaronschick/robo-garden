#!/usr/bin/env bash
# E12-override — re-run of E12 (lr=1e-12, warmstart, arena-mode) but with
# fixed-yaw det-eval overrides that bypass the curriculum entirely.
#
# Decisive test for the E-series collapse diagnosis:
#   Outcome A (curriculum is the full cause): all four det-evals score
#     similarly (~300 at easy rung, ~170 at hard rung) and the trajectory
#     is FLAT across evals. The -7 number in E12 came from the curriculum
#     walking the eval deeper into hard yaws, nothing else.
#   Outcome B (curriculum is partial): hardest-rung det-eval still collapses
#     toward -7 by eval 4. Something in 295k env steps of training state
#     (PhysX caches, articulation warm-starts, RNG advancement) is the
#     remaining cause.
#
# Rungs picked to match the E-series eval 1 (yaw≈0.6, gd≈0.524) and eval 4
# (yaw≈2.82, gd≈1.66) conditions exactly, so the eval numbers are directly
# comparable to the E12 log.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e12_override_repro"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e12-override] E1 reward + lr=1e-12 + fixed-yaw det-eval overrides"

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
  echo "[e12-override] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e12-override] run_id=${RUN_ID}"
echo "[e12-override] warmstart=${SEED_CKPT}"
echo "[e12-override] ckpt_dir=${CKPT_DIR}"
echo "[e12-override] num_envs=1024 timesteps=300000 lr=1e-12 bc_reg=0.0 freeze_scaler=auto"
echo "[e12-override] det-eval rungs: yaw=[0.6, 2.82] goal_dir=[0.524, 1.66]"

t_start=$(date +%s)
echo "[e12-override] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 300000 \
  --load-checkpoint "${SEED_CKPT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 1e-12 \
  --bc-reg-coef 0.0 \
  --freeze-scaler-after-warmstart auto \
  --det-eval-yaw-spans-rad "0.6,2.82" \
  --det-eval-goal-dir-spans-rad "0.524,1.66"
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e12-override] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
