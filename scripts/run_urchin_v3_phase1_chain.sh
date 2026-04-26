#!/usr/bin/env bash
# Phase 1 chain: random goal direction. Extends the Phase-0 hardening
# chain by breaking the fixed (+0.5,+0.5) goal assumption. Each reset
# samples a per-env goal on a 1.41 m ring around (-0.5,-0.5), at an
# angle drawn from a curriculum-annealed span around the Phase-0
# base heading (+45°). Spawn, radius, episode length unchanged.
#
# Workflow:
#   1. pathB_phase1_smoke (200k steps, span 30→90°) runs first.
#   2. After smoke, the chain pauses on a sentinel file. User renders
#      the smoke checkpoint, reviews, then deletes the sentinel to
#      authorize the full runs.
#   3. pathB_phase1_run1 (1M, span 30→180°).
#   4. pathB_phase1_run2 (2M, span 180→360°).
#
# Chain handoff stays on final_checkpoint.pt per
# `feedback_checkpoint_handoff.md`. Do not change without user approval.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
BASE="workspace/checkpoints"
# Phase 0's final (user video-approved 2026-04-21).
SEED_CKPT="${BASE}/urchin_v3_pathB_chain_run6/final_checkpoint.pt"
# Abort if a run's best_reward drops more than this fraction below the
# prior run's best (relative to abs(prev_best)).
COLLAPSE_FRAC="0.5"

# Task geometry matches Phase 0. NOTE: URCHIN_GOAL_XY is NOT exported —
# the goal is now per-env and per-episode, resampled in _reset_idx.
# GOAL_XY module constant is still read as the base-heading anchor
# (atan2(GOAL_XY - START_XY) => +45°) when URCHIN_GOAL_DIR_BASE_DEG is
# unset, so we leave it at its default (1.5, 1.5) direction — only the
# ATAN2 result matters for the base heading.
export URCHIN_START_XY="-0.5,-0.5"
export URCHIN_GOAL_XY="0.5,0.5"   # keeps Phase-0 base heading at +45°
export URCHIN_EPISODE_S="8.0"

# Phase 1 defaults (per-run overrides below).
export URCHIN_GOAL_RADIUS="1.41"
export URCHIN_GOAL_DIR_BASE_DEG=""   # "" = derive +45° from START/GOAL_XY

# (run_id, timesteps, span_start_deg, span_end_deg, anneal_end_steps)
RUNS=(
  "pathB_phase1_smoke:200000:30:90:200000"
  "pathB_phase1_run1:1000000:30:180:1000000"
  "pathB_phase1_run2:2000000:180:360:2000000"
)

# Hand off on final_checkpoint.pt (preserves optimizer + scaler state).
# feedback_checkpoint_handoff.md: never change to best_checkpoint.pt
# without explicit user approval.
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
  IFS=':' read -r run_id timesteps span_start span_end anneal_end <<< "${entry}"
  ckpt_dir="${BASE}/urchin_v3_${run_id}"

  if [[ ! -f "${prev_ckpt}" ]]; then
    echo "[chain] ABORT: prev checkpoint not found: ${prev_ckpt}"
    exit 1
  fi

  # Smoke gate: block full runs until user has rendered + reviewed the smoke.
  # Sentinel file is created below after the smoke run finishes, and the
  # user deletes it to authorize the rest of the chain.
  if [[ "${run_id}" != "pathB_phase1_smoke" ]]; then
    smoke_sentinel="${BASE}/urchin_v3_pathB_phase1_smoke/.awaiting_review"
    if [[ -f "${smoke_sentinel}" ]]; then
      echo "[chain] PAUSE: smoke awaiting user video review."
      echo "[chain] Render checkpoint at:"
      echo "        ${BASE}/urchin_v3_pathB_phase1_smoke/final_checkpoint.pt"
      echo "[chain] Delete sentinel to resume: rm '${smoke_sentinel}'"
      exit 0
    fi
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
  echo "[chain] goal_dir_span=${span_start}->${span_end} deg  anneal_end=${anneal_end}"
  echo "[chain] ckpt_dir=${ckpt_dir}"

  URCHIN_GOAL_DIR_SPAN_START_DEG="${span_start}" \
  URCHIN_GOAL_DIR_SPAN_END_DEG="${span_end}" \
  URCHIN_GOAL_DIR_ANNEAL_START="0" \
  URCHIN_GOAL_DIR_ANNEAL_END="${anneal_end}" \
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

  # Collapse gate — skip for the smoke run (no prev Phase-1 baseline yet)
  # and for the first full run (widening distribution is expected to dip).
  if [[ -n "${prev_best}" && "${run_id}" != "pathB_phase1_run1" ]]; then
    if python -c "
import sys
best, prev, frac = float('${best}'), float('${prev_best}'), float('${COLLAPSE_FRAC}')
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

  # Drop the smoke-review sentinel so the next chain invocation pauses
  # until the user has rendered + reviewed the smoke checkpoint.
  if [[ "${run_id}" == "pathB_phase1_smoke" ]]; then
    touch "${ckpt_dir}/.awaiting_review"
    echo "[chain] SMOKE DONE. Render + review:"
    echo "        ${ckpt_dir}/final_checkpoint.pt"
    echo "[chain] Then: rm '${ckpt_dir}/.awaiting_review' and re-run this script."
  fi
done

echo "[chain] ALL RUNS COMPLETE"
