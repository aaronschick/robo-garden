#!/usr/bin/env bash
# E3 — Sparse shortest-path reward (absolute minimum).
#
# Phase 2d (cos^2 then cos^1.5 velocity reward) collapsed identically:
# warmstart det-eval 2249 at step 55k then monotonically declines to ~500
# by step 390k. Same failure mode as phase2c. Diagnosis: too many additive
# reward terms fighting; PPO finds "sit still" local minima.
#
# This experiment strips the reward to the absolute bone:
#   every step: -1.0   (fixed per-step time penalty)
#   on reach:   +goal_bonus (currently 500)
# Nothing else. No progress, no velocity, no distance penalty, no aspherity.
# Classic sparse-reward formulation. If this produces a goal-seeking policy
# (or at least doesn't collapse), dense shaping was actively harmful. If it
# fails (sit-still), dense shaping is needed — and we have the baseline.
#
# Reward edit (urchin_env_cfg.py): `return goal_r - 1.0`. Broadcast yields
# shape (N,) identical to what the multi-term sum produced.
#
# Single 300k-step run. No chain, no collapse gate, no smoke sentinel.
# Warmstart from phase2b_run2/final (same seed phase2d and E1/E2 use).

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e3_sparse_shortestpath"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e3] reward = -1 per step + goal_bonus on reach ONLY (sparse shortest-path)"

# --- Arena-scale task geometry (same as phase2d / E1 / E2) ---------------
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
  echo "[e3] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e3] run_id=${RUN_ID}"
echo "[e3] warmstart=${SEED_CKPT}"
echo "[e3] ckpt_dir=${CKPT_DIR}"
echo "[e3] num_envs=1024 timesteps=300000 lr=5e-5 bc_reg=0.0 freeze_scaler=auto"
echo "[e3] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e3] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 300000 \
  --load-checkpoint "${SEED_CKPT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 5e-5 \
  --bc-reg-coef 0.0 \
  --freeze-scaler-after-warmstart auto
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e3] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
