#!/usr/bin/env bash
# E10 — E1 setup + learning_epochs dropped from 5 -> 2.
#
# E1/E7/E8/E9 all converge to det-eval ~217-219 at the second eval
# (step ~147k), regardless of reward shape, noise freeze, scaler freeze,
# or value_loss_scale. This is a remarkably tight PPO attractor.
#
# Remaining structural hyperparameters inside PPO's update step:
#   * learning_epochs (5) — how many gradient passes per rollout
#   * mini_batches (4)   — minibatch count inside each epoch
#   * ratio_clip (0.2)   — PPO clipping threshold
#   * gae_lambda (0.95)  — advantage smoothing
#
# learning_epochs=5 means 20 total gradient steps per PPO update. On a
# noisy advantage signal, those later passes over-fit to a single
# rollout's quirks, pushing the policy well beyond the trust region
# that ratio_clip was designed to enforce. Lowering to 2 = 8 gradient
# steps per PPO update, closer to the spec-PPO 3-10 range but toward
# the low end.
#
# Expected outcomes:
#   - eval 2 ≠ 217-219: the over-update hypothesis holds; follow up with
#     eval-3/4 trajectory to see whether collapse is merely slowed or
#     actually arrested
#   - eval 2 still ~217-219: PPO update-count isn't the lever either.
#     Next candidate: E11 — reinitialize the critic fresh at warmstart
#     (preserve policy) to test whether a mis-bootstrapped value head
#     is poisoning advantages from step 1.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e10_fewer_epochs"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e10] reward = E1 minimal potential | warmstart + learning_epochs=2 (was 5)"

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
  echo "[e10] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e10] run_id=${RUN_ID}"
echo "[e10] warmstart=${SEED_CKPT}"
echo "[e10] ckpt_dir=${CKPT_DIR}"
echo "[e10] num_envs=1024 timesteps=300000 lr=5e-5 learning_epochs=2 bc_reg=0.0 freeze_scaler=auto"
echo "[e10] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e10] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 300000 \
  --load-checkpoint "${SEED_CKPT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 5e-5 \
  --bc-reg-coef 0.0 \
  --freeze-scaler-after-warmstart auto \
  --learning-epochs 2
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e10] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
