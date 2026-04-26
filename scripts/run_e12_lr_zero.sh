#!/usr/bin/env bash
# E12 — E1 setup + learning-rate dropped to effectively 0 (1e-12).
#
# The null-update control. Six interventions (E1/E7/E8/E9/E10/E11)
# against PPO hyperparameters, critic init, scaler state, and log_std
# all produced the SAME collapse trajectory:
#   eval1=~358  eval2=~220  eval3=~50  eval4=~-7
# E6 tested lr=5e-6 (10x lower than standard 5e-5) and still collapsed.
#
# E12 pins lr all the way down (1e-12 — effectively zero under float32).
# Every other knob is identical to E1. Adam with lr=1e-12 produces
# essentially zero weight updates per step.
#
# Expected outcomes:
#   - det-eval stays ~358 across all 4 evals: PPO weight updates are
#     unambiguously the destructive force. The *direction* of the
#     gradient, not its magnitude, is what's breaking the policy.
#     Next experiments target advantage computation (normalization,
#     GAE lambda, or raw advantage direction).
#   - det-eval still collapses: destruction is NOT from gradient
#     updates. Candidates: running-stats drift in scaler/normalizer,
#     eval stochasticity against a distribution the policy no longer
#     matches, or a state the agent traversed that locked in during
#     init.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e12_lr_zero"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e12] reward = E1 minimal potential | warmstart + lr=1e-12 (effectively zero)"

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
  echo "[e12] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e12] run_id=${RUN_ID}"
echo "[e12] warmstart=${SEED_CKPT}"
echo "[e12] ckpt_dir=${CKPT_DIR}"
echo "[e12] num_envs=1024 timesteps=300000 lr=1e-12 bc_reg=0.0 freeze_scaler=auto"
echo "[e12] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e12] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

"${PY}" "${TRAIN}" \
  --headless \
  --num-envs 1024 \
  --timesteps 300000 \
  --load-checkpoint "${SEED_CKPT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-id "${RUN_ID}" \
  --learning-rate 1e-12 \
  --bc-reg-coef 0.0 \
  --freeze-scaler-after-warmstart auto
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e12] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
