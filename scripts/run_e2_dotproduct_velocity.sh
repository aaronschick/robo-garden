#!/usr/bin/env bash
# E2 experiment (2026-04-23): pure linear dot-product velocity reward.
#
#   r_vel = velocity_reward_weight * (vel_xy · unit_to_goal)
#   reward = r_vel + (at_goal * goal_bonus)
#
# One-shot 300k-step training — no chain, no smoke gate. Parallel with E1
# (potential-only) and E3 (TBD) to diagnose the phase2d collapse
# (det-eval 2249 -> ~500 over 335k steps, identical under cos^2 and cos^1.5).
# Hypothesis: ~10 additive reward terms were fighting each other; a single
# linear signed gradient is more PPO-friendly.
#
# Reward edits applied in urchin_env_cfg.py (2026-04-23, E2):
#   L896-906: replaced cos^1.5 vel_reward block with
#             r_vel = velocity_reward_weight * vel_toward  (signed, linear)
#   L1061-1067: return r_vel + goal_r  (no progress, no penalties)
# velocity_reward_weight=5.0 (L157) and goal_bonus=500.0 (L142) kept.
#
# Warmstart: phase2b_run2/final_checkpoint.pt (last video-reviewed seed).
# No scaler refit (obs unchanged). No BC regularization.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
BASE="workspace/checkpoints"
SEED_CKPT="${BASE}/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"

RUN_ID="e2_dotproduct_velocity"
CKPT_DIR="${BASE}/urchin_v3_${RUN_ID}"

# --- Arena-scale task geometry (unchanged from Phase 2b/2c/2d) -----------
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

echo "[e2] reward = velocity_reward_weight * (vel_xy · unit_to_goal) + goal_bonus ONLY"
echo "[e2] run_id=${RUN_ID}"
echo "[e2] warmstart=${SEED_CKPT}"
echo "[e2] ckpt_dir=${CKPT_DIR}"
echo "[e2] num_envs=512  timesteps=300000  lr=5e-5  bc_reg=0.0  freeze_scaler=auto"
echo "[e2] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

if [[ ! -f "${SEED_CKPT}" ]]; then
  echo "[e2] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

t_start=$(date +%s)
echo "[e2] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

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
echo "[e2] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  exit=${rc}"

exit ${rc}
