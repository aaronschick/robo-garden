#!/usr/bin/env bash
# E11 — E1 setup + CRITIC re-randomized at warmstart (policy preserved).
#
# E1/E7/E8/E9/E10 collapse identically:
#   eval1=~358  eval2=~222  eval3=~48  eval4=~-7
# across 5 orthogonal interventions (value_loss_scale, log_std freeze,
# scaler freeze, learning_epochs). E4 scratch (policy AND critic random)
# shows the SAME pattern.
#
# Common denominator in every case: the critic outputs are miscalibrated
# at step 0 w.r.t. the current reward/env distribution.
#   - Warmstart runs: critic learned on phase2b_run2 reward scale +
#     arena. E-runs use different arena_half (2.0) and potential_scale
#     (1.5m). Critic values are biased along the OBS manifold the new
#     rollouts explore.
#   - Scratch run: critic is random. Same effective result — biased
#     advantages on every sample.
#
# PPO computes advantages as A = R - V(s). Systematically wrong V(s)
# produces systematically wrong A, and PPO follows those advantages to
# a worse policy with full confidence (because ratio_clip caps only
# magnitude, not direction).
#
# E11 removes this specific asymmetry: the policy remains warmstart-
# good, but the critic is reset to random. Random critic produces
# UNBIASED (high-variance) advantages. The policy should degrade more
# slowly — or stay stable longer as the critic burns in — if critic
# bias was the mechanism.
#
# Expected outcomes:
#   - eval 2 >> 220 (say >300): critic bias was the mechanism. Follow
#     up: value-pretraining phase before joint optimization.
#   - eval 2 ≈ 220: critic bias is NOT the lever. Next suspect is the
#     raw PPO-objective gradient direction under the potential-based
#     reward (the Jensen-gap / telescoping artifact in the returns).

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e11_critic_reinit"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e11] reward = E1 minimal potential | warmstart policy + RE-INIT critic"

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
  echo "[e11] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e11] run_id=${RUN_ID}"
echo "[e11] warmstart=${SEED_CKPT}"
echo "[e11] ckpt_dir=${CKPT_DIR}"
echo "[e11] num_envs=1024 timesteps=300000 lr=5e-5 bc_reg=0.0 freeze_scaler=auto REINIT_CRITIC=1"
echo "[e11] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e11] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

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
  --reinit-critic
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e11] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
