#!/usr/bin/env bash
# E6 — Warmstarted fine-tuning with 10x smaller learning rate.
#
# All three reward experiments (E1 minimal potential, E2 dot-product
# velocity, E3 sparse shortest-path) collapse monotonically from a
# warmstarted phase2b_run2 policy. Reward shape doesn't matter — even Ng
# 1999 canonical potential shaping (E1), which is theoretically
# optimal-policy-preserving, decays -70% over 4 det-evals.
#
# Hypothesis: PPO at lr=5e-5 updates the warmstart too aggressively.
# Gradient variance from small env count + off-distribution reward
# landscape drives the policy into penalty-avoiding local minima before
# it can even evaluate the current policy's rollouts.
#
# This experiment keeps reward = E1 (minimal potential, already in
# env_cfg) and the warmstart from phase2b_run2, but drops learning rate
# to 5e-6 (10x smaller). If the collapse is caused by LR being too large
# for fine-tuning, det-eval should trend flat or up.
#
# If E6 still collapses → warmstart itself is incompatible → run E4
# (from scratch on E1 reward) next.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e6_tiny_lr"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e6] reward = E1 minimal potential (progress + goal_bonus) | lr = 5e-6 (10x smaller)"

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
  echo "[e6] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e6] run_id=${RUN_ID}"
echo "[e6] warmstart=${SEED_CKPT}"
echo "[e6] ckpt_dir=${CKPT_DIR}"
echo "[e6] num_envs=1024 timesteps=300000 lr=5e-6 bc_reg=0.0 freeze_scaler=auto"
echo "[e6] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e6] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 300000 \
  --load-checkpoint "${SEED_CKPT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 5e-6 \
  --bc-reg-coef 0.0 \
  --freeze-scaler-after-warmstart auto
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e6] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
