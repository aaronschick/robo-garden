#!/usr/bin/env bash
# E4 — Train from scratch (random init) on E1 minimal potential reward.
#
# E1/E2/E3 all collapsed from the warmstart. E6 at lr=5e-6 also collapsing
# (eval 2 at -36%). This isolates the question: is the warmstart itself
# incompatible with the new reward landscape, or is the training pipeline
# broken regardless of init?
#
# No --load-checkpoint. 500k steps (scratch needs more than 300k to show
# any learning signal). Reward = E1 (minimal Ng-1999 potential) — already
# set in urchin_env_cfg.py.
#
# Expected outcomes:
#   - det-eval starts near 0 or mildly negative (untrained policy)
#   - if det-eval TRENDS UP over the run: warmstart is the poison; fix is
#     to retrain from scratch or use BC-regularized fine-tuning (E5)
#   - if det-eval stays flat near 0: pipeline is architecturally broken
#     beyond warmstart — need deeper debugging (gradient norms, critic
#     loss, PhysX state issues, scaler stats)

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
RUN_ID="e4_scratch_minimal"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e4] reward = E1 minimal potential (progress + goal_bonus) | SCRATCH init (no warmstart)"

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

echo "[e4] run_id=${RUN_ID}"
echo "[e4] warmstart=NONE (scratch)"
echo "[e4] ckpt_dir=${CKPT_DIR}"
echo "[e4] num_envs=1024 timesteps=500000 lr=5e-5 bc_reg=0.0 freeze_scaler=off"
echo "[e4] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e4] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 500000 \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 5e-5 \
  --bc-reg-coef 0.0 \
  --freeze-scaler-after-warmstart off
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e4] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
