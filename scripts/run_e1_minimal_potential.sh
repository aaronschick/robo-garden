#!/usr/bin/env bash
# E1 — Minimal Ng-1999 potential-shaped reward.
#
# Phase 2d (cos^2 then cos^1.5 velocity reward) collapsed identically:
# warmstart det-eval 2249 at step 55k then monotonically declines to ~500
# by step 390k (7 consecutive drops). Same failure mode as phase2c
# lateral_penalty run. Diagnosis: ~10 additive reward terms fighting each
# other; PPO keeps finding penalty-avoiding "sit still" local minima.
#
# This experiment strips the reward to ONLY:
#   progress = progress_reward_weight * (gamma * phi(s') - phi(s))
#   goal_r   = at_goal.float() * goal_bonus
# Nothing else. Ng 1999 proves potential-shaping preserves the optimal
# policy, so if this still collapses reward isn't the problem — it's
# PPO/exploration/something architectural.
#
# Reward edit (urchin_env_cfg.py L1061-1063) replaces the 10-term return
# tuple with `return progress + goal_r`. Upstream computations retained.
#
# Single 300k-step run. No chain, no collapse gate, no smoke sentinel.
# Warmstart from phase2b_run2/final (same seed phase2d used).

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e1_minimal_potential"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e1] reward = progress (potential-based) + goal_bonus ONLY"

# --- Arena-scale task geometry (same as phase2d) -------------------------
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
  echo "[e1] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e1] run_id=${RUN_ID}"
echo "[e1] warmstart=${SEED_CKPT}"
echo "[e1] ckpt_dir=${CKPT_DIR}"
echo "[e1] num_envs=512 timesteps=300000 lr=5e-5 bc_reg=0.0 freeze_scaler=auto"
echo "[e1] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e1] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 512 \
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
echo "[e1] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
