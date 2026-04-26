#!/usr/bin/env bash
# E8 — E1 minimal-potential reward + warmstart + SCALER NOT FROZEN.
#
# E7 (frozen log_std=-2.3 on warmstart) collapsed harder than E1 and
# brought mean_reward down with det_reward. That rules out the
# "noise-aided goal-reach" hypothesis and points the finger at the
# policy itself degrading from step 1.
#
# New suspect: --freeze-scaler-after-warmstart=auto pins the
# RunningStandardScaler's running_mean / running_variance at the
# phase2b_run2 stats. If the arena-scale E-experiment obs distribution
# (URCHIN_START_XY=-0.5,-0.5, URCHIN_GOAL_XY=0.5,0.5, arena_half=2.0,
# min_spawn_goal_dist=0.3) differs from what phase2b_run2 saw at train
# time, the policy gets mis-normalized inputs on every step -> all
# downstream gradients push into garbage.
#
# E8 = E1 setup with --freeze-scaler-after-warmstart=off. The scaler's
# running stats update on the actual E8 rollouts, re-aligning to the
# current obs distribution within a few updates.
#
# Expected outcomes:
#   - det-eval trends flat or up: scaler mismatch was the collapse
#     cause; next step is either (a) refit scaler then freeze, or
#     (b) leave scaler permanently unfrozen.
#   - det-eval still collapses: scaler is NOT the culprit. Next
#     suspect: critic is being overwritten by value_loss_scale=2.0
#     dominated updates, ruining advantages (run E9 with value_loss_scale
#     dropped to 0.5).

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e8_scaler_unfrozen"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e8] reward = E1 minimal potential | warmstart + scaler UNFROZEN (updates every rollout)"

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
  echo "[e8] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e8] run_id=${RUN_ID}"
echo "[e8] warmstart=${SEED_CKPT}"
echo "[e8] ckpt_dir=${CKPT_DIR}"
echo "[e8] num_envs=1024 timesteps=300000 lr=5e-5 bc_reg=0.0 freeze_scaler=OFF"
echo "[e8] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e8] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 300000 \
  --load-checkpoint "${SEED_CKPT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 5e-5 \
  --bc-reg-coef 0.0 \
  --freeze-scaler-after-warmstart off
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e8] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
