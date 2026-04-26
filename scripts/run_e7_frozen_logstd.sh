#!/usr/bin/env bash
# E7 — Warmstart on E1 minimal potential reward, but with log_std frozen
# at -2.3 (std ~ 0.1), an order of magnitude below the seed's 0.28.
#
# Cross-run analysis of E1/E4/E5 revealed:
#   * mean_reward >> det_reward from the first rollout onward
#     (E1: 482 > 360, E4: 485 > 382, E5: 484 > 319 at first eval)
#   * log_std itself did NOT drift during E1 (final == seed at -1.300)
#   * E5 kl_bc blew up to 1-2 million despite bc_reg_coef=0.5 — the BC
#     anchor couldn't hold the policy
#
# Interpretation: PPO is converging on a policy whose MEAN ACTION fails
# at goal-seeking, but whose STOCHASTIC SAMPLES (mean + noise) hit the
# goal by exploration. Stripping the noise in deterministic eval reveals
# the degenerate mean action -> det-eval collapses.
#
# E7 removes PPO's ability to rely on noise by freezing log_std at a
# small constant. PPO is then forced to learn a policy whose mean action
# itself is goal-seeking, because it has no high-noise alternative to
# fall back on.
#
# --freeze-log-std -2.3 -> std ~ 0.1 (vs seed 0.28, scratch 1.00)
#
# Expected outcomes:
#   - det-eval trends flat or UP: confirmed — noise-reliance was the
#     collapse mechanism. Next experiment: tune the frozen std value.
#   - det-eval still collapses: something deeper than noise, probably
#     the potential-shaped reward itself (Jensen-gap farming).
#     Next experiment: swap to raw delta-distance reward.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
RUN_ID="e7_frozen_logstd"
CKPT_DIR="workspace/checkpoints/urchin_v3_${RUN_ID}"

echo "[e7] reward = E1 minimal potential | warmstart + frozen log_std=-2.3 (std~0.1)"

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
  echo "[e7] ABORT: seed checkpoint not found: ${SEED_CKPT}"
  exit 1
fi

echo "[e7] run_id=${RUN_ID}"
echo "[e7] warmstart=${SEED_CKPT}"
echo "[e7] ckpt_dir=${CKPT_DIR}"
echo "[e7] num_envs=1024 timesteps=300000 lr=5e-5 freeze_log_std=-2.3 freeze_scaler=auto"
echo "[e7] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"

t_start=$(date +%s)
echo "[e7] START at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

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
  --freeze-log-std -2.3
rc=$?

t_end=$(date +%s)
elapsed=$(( t_end - t_start ))
echo "[e7] END at $(date -u +'%Y-%m-%dT%H:%M:%SZ')  elapsed=${elapsed}s  rc=${rc}"

exit ${rc}
