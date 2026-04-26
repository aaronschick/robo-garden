#!/usr/bin/env bash
# Chain of warmstart PPO runs for urchin_v3 rolling-policy hardening.
# Each run warmstarts from the previous run's final_checkpoint.pt to
# preserve full training-state continuity (optimizer momentum,
# RunningStandardScaler stats, KL history). The collapse-detection gate
# aborts the chain if a run's best_reward drops catastrophically vs the
# prior run — this is the guard against tail-of-run PPO collapse.
# If collapse does happen under the current fixed pipeline, surface it
# and discuss the handoff policy with the user before switching sources.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
BASE="workspace/checkpoints"
# Path-B BC seed: the residual-on-oracle BC warmstart. Retargeted so the
# residual head outputs ~0 on contactpush data; verified 2026-04-21 to
# match oracle-alone det-eval (mean 2342 fixed-yaw, 5/5 seeds rolling).
SEED_CKPT="${BASE}/urchin_v3_pathB_bc/bc_warmstart_checkpoint.pt"
# Abort if a run's best_reward drops more than this fraction below the
# prior run's best (relative to abs(prev_best)). Prevents poisoning.
COLLAPSE_FRAC="0.5"

# Match PPO task geometry to the BC recording (see
# record_bc_dataset.py defaults): 1.41m diagonal, 8s episodes.
# Episodes were bumped from 5s -> 8s on 2026-04-21 (residual-on-oracle
# rollout) so the goal_r=500 bonus at 0.5m radius is comfortably
# reachable (need ~0.91m traveled at 0.2 m/s speed-gate => ~4.55s;
# 8s gives headroom for non-ideal trajectories).
# The env defaults (4.24m, 20s) are out-of-distribution for the BC prior
# and cause catastrophic forgetting. Graduate to full arena later via
# dist_curriculum_start/end_sim_steps once short task converges.
export URCHIN_START_XY="-0.5,-0.5"
export URCHIN_GOAL_XY="0.5,0.5"
export URCHIN_EPISODE_S="8.0"

# (run_id, timesteps) pairs. Prefix 'pathB_chain_run' disambiguates from
# pre-Path-B 'harden_run' dirs left by earlier (2026-04-20) sessions.
RUNS=(
  "pathB_chain_run1:1000000"
  "pathB_chain_run2:2000000"
  "pathB_chain_run3:2000000"
  "pathB_chain_run4:2000000"
  "pathB_chain_run5:2000000"
  "pathB_chain_run6:2000000"
)

# Hand off on final_checkpoint.pt to preserve full training-state
# continuity (optimizer state, RunningStandardScaler statistics, etc.).
# Switching to best_checkpoint.pt throws that continuity away and is
# only appropriate if PPO actually collapses in the tail of runs — do
# not change this without explicit user approval (see memory
# feedback_checkpoint_handoff).
pick_handoff() {
  local dir="$1"
  if [[ -f "${dir}/final_checkpoint.pt" ]]; then
    echo "${dir}/final_checkpoint.pt"
  else
    echo ""
  fi
}

prev_ckpt="${SEED_CKPT}"
prev_best=""

for entry in "${RUNS[@]}"; do
  run_id="${entry%%:*}"
  timesteps="${entry##*:}"
  ckpt_dir="${BASE}/urchin_v3_${run_id}"

  if [[ ! -f "${prev_ckpt}" ]]; then
    echo "[chain] ABORT: prev checkpoint not found: ${prev_ckpt}"
    exit 1
  fi

  # Skip if run already finished successfully (e.g. resuming chain).
  existing="${ckpt_dir}/result.json"
  if [[ -f "${existing}" ]]; then
    ok=$(python -c "import json; print(json.load(open(r'${existing}')).get('success', False))")
    if [[ "${ok}" == "True" ]]; then
      handoff=$(pick_handoff "${ckpt_dir}")
      if [[ -n "${handoff}" ]]; then
        echo "[chain] SKIP ${run_id} (already success, handoff=${handoff})"
        prev_ckpt="${handoff}"
        prev_best=$(python -c "import json; print(json.load(open(r'${existing}')).get('best_reward'))")
        continue
      fi
    fi
  fi

  echo "[chain] ====== ${run_id} ======"
  echo "[chain] timesteps=${timesteps}  warmstart=${prev_ckpt}"
  echo "[chain] ckpt_dir=${ckpt_dir}"

  "${PY}" "${TRAIN}" \
    --headless \
    --num-envs 64 \
    --timesteps "${timesteps}" \
    --load-checkpoint "${prev_ckpt}" \
    --checkpoint-dir "${ckpt_dir}" \
    --run-id "${run_id}" \
    --learning-rate 5e-5 \
    --bc-post-log-std -1.6 \
    --bc-reg-coef 0.5 \
    --bc-reg-anneal-steps 2000000 \
    --bc-reg-checkpoint "${SEED_CKPT}" \
    --freeze-scaler-after-warmstart auto
  rc=$?

  result_json="${ckpt_dir}/result.json"
  if [[ $rc -ne 0 ]]; then
    echo "[chain] ABORT: ${run_id} exit=${rc}"
    exit 1
  fi
  if [[ ! -f "${result_json}" ]]; then
    echo "[chain] ABORT: missing ${result_json}"
    exit 1
  fi
  success=$(python -c "import json,sys; print(json.load(open(r'${result_json}')).get('success', False))")
  if [[ "${success}" != "True" ]]; then
    echo "[chain] ABORT: ${run_id} reported success=${success}"
    exit 1
  fi
  best=$(python -c "import json; print(json.load(open(r'${result_json}')).get('best_reward'))")
  echo "[chain] ${run_id} DONE  best_reward=${best}"

  # Quality gate: detect catastrophic collapse vs prior run's best.
  # Skips the first run (no prev_best) and the BC-seeded first run
  # (prev_best empty).
  if [[ -n "${prev_best}" ]]; then
    if python -c "
import sys
best, prev, frac = float('${best}'), float('${prev_best}'), float('${COLLAPSE_FRAC}')
# Allow drops within COLLAPSE_FRAC of abs(prev). Abort on worse.
threshold = prev - frac * abs(prev)
sys.exit(0 if best >= threshold else 1)
"; then
      :
    else
      echo "[chain] ABORT: ${run_id} collapsed (best=${best} vs prev=${prev_best}, frac=${COLLAPSE_FRAC})"
      echo "[chain] Inspect ${ckpt_dir} and retune before resuming."
      exit 1
    fi
  fi

  handoff=$(pick_handoff "${ckpt_dir}")
  if [[ -z "${handoff}" ]]; then
    echo "[chain] ABORT: no checkpoint to hand off from ${ckpt_dir}"
    exit 1
  fi
  echo "[chain] handoff -> ${handoff}"
  prev_ckpt="${handoff}"
  prev_best="${best}"
done

echo "[chain] ALL RUNS COMPLETE"
