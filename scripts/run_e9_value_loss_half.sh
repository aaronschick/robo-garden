#!/usr/bin/env bash
# E9 — E1 setup + value_loss_scale dropped from 2.0 -> 0.5.
#
# E1/E4/E5/E6/E7/E8 all collapse with the same ~monotonic slope from
# eval 2 onward, regardless of warmstart/scratch, reward variant,
# scaler-freeze, log_std-freeze, or BC regularization. The common
# denominator is PPO's hyperparameters. value_loss_scale=2.0 is
# aggressive (standard PPO uses 0.5); if the critic loss dominates
# backward(), actor gradient direction can be overwhelmed by critic
# regression error, even with ratio_clip=0.2.
#
# Expected outcomes:
#   - det-eval trends flat/up or collapses more slowly: critic-scale
#     was the dominant issue; move on to tuning entropy / learning_epochs
#   - still collapses at same slope: next candidate is the episodic
#     reward telescoping (Jensen gap / episode termination-on-reach)
#     or learning_epochs=5 being too many for the noisy gradient.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e9_value_loss_half"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e9] reward = E1 minimal potential | warmstart + value_loss_scale=0.5 (was 2.0)"

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
  echo "[e9] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e9] run_id=${RUN_ID}"
echo "[e9] warmstart=${SEED_CKPT}"
echo "[e9] ckpt_dir=${CKPT_DIR}"
echo "[e9] num_envs=1024 timesteps=300000 lr=5e-5 value_loss_scale=0.5 bc_reg=0.0 freeze_scaler=auto"
echo "[e9] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e9] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

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
  --value-loss-scale 0.5
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e9] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
