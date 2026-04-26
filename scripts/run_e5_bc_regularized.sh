#!/usr/bin/env bash
# E5 — Warmstarted fine-tuning with BC regularization (KL anchor to
# phase2b_run2 policy).
#
# E1/E2/E3/E6 all collapse when PPO is free to move the warmstart as far
# as it wants. The classical RL fine-tuning fix: add a KL-divergence
# penalty pulling the current policy toward the anchor. train.py exposes
# --bc-reg-coef (default 0.5) for exactly this purpose.
#
# Anchor = phase2b_run2/final_checkpoint.pt (same as warmstart).
# Reward = E1 (minimal Ng-1999 potential) — already in env_cfg.
# lr = 5e-5 (same as E1). We want to isolate the effect of BC reg alone.
#
# --bc-reg-coef 0.5 is the default. It anneals over 2M steps; for a 300k
# run that means ~15% decay — essentially constant.
#
# Expected outcomes:
#   - det-eval trends flat/up: KL anchor prevents the destructive drift;
#     BC-regularized fine-tuning is the path forward
#   - det-eval still collapses (slower): the destructive force is
#     stronger than what 0.5 can hold; bump to 1.0 or pretrain on BC
#     dataset first

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e5_bc_regularized"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e5] reward = E1 minimal potential + BC-reg (KL to phase2b_run2 anchor, coef=0.5)"

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
  echo "[e5] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e5] run_id=${RUN_ID}"
echo "[e5] warmstart=${SEED_CKPT}"
echo "[e5] ckpt_dir=${CKPT_DIR}"
echo "[e5] num_envs=1024 timesteps=300000 lr=5e-5 bc_reg=0.5 freeze_scaler=auto"
echo "[e5] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e5] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 300000 \
  --load-checkpoint "${SEED_CKPT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 5e-5 \
  --bc-reg-coef 0.5 \
  --freeze-scaler-after-warmstart auto
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e5] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
