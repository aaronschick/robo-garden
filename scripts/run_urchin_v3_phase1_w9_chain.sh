#!/usr/bin/env bash
# W9.1b run2: extend the validated W9.1b recipe to wider yaw + 2x duration.
# Run1 (1M, yaw 30->90, frozen residual=0.3, BC reg beta=0.1 anneal 1M)
# successfully prevented the E-series collapse — best det_reward 339.32
# held throughout, 19 det_evals mean-reverted in 220-340 band, 6-bearing
# video user-approved. Run2 keeps the same recipe (frozen residual + BC
# reg beta=0.1) but widens yaw to 180->360 deg and runs 2M steps with the
# BC-reg anneal stretched to 2M to match. Warmstart is from run1's
# final_checkpoint.pt via skrl --load-checkpoint.
# See plan section W9.1 and memory `project_urchin_v3_w9_1b_run1_success.md`.
#
# Per W9.1b run2:
#   - yaw span 180 deg -> 360 deg (full bearing coverage).
#   - residual scale stays frozen at 0.3 via URCHIN_RESIDUAL_SCALE_INIT/FINAL.
#   - PPO loss adds beta * KL(pi||pi_bc), beta0=0.1 anneal=2M.
#   - chain seeds prev_ckpt from run1 final_checkpoint.pt (skip-loop hits
#     it because run1's result.json has success=true).
#
# W9 Phase 1 chain: PPO warmstart from the W8 BC concat-MLP checkpoint
# (148-D conditioned input, primitive=straight_roll, style=neutral).
#
# Workflow mirrors run_urchin_v3_phase1_chain.sh:
#   1. w9_phase1_smoke (200k, span 30->90 deg) — uses --bc-checkpoint to
#      load the Phase-5 BC weights into the 148-D Policy MLP and seed the
#      scaler. Sentinel-gates the rest of the chain.
#   2. w9_phase1_run1 (1M, span 30->180 deg) — handoff from smoke
#      final_checkpoint.pt via --load-checkpoint (skrl format).
#   3. w9_phase1_run2 (2M, span 180->360 deg) — handoff from run1
#      final_checkpoint.pt.
#
# Conditioning is fixed across all envs and all episodes for Phase 1.
# Reward is goal-reach only (no per-primitive shaping) — the conditioning
# is functionally constant; it's the architectural input matching BC.
#
# Chain handoff stays on final_checkpoint.pt per
# `feedback_checkpoint_handoff.md`.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
BASE="workspace/checkpoints"
# W8 BC checkpoint (user-approved 6-bearing review 2026-04-25).
BC_CKPT="${BASE}/urchin_v3_bc_phase5_concat/model.pt"
COLLAPSE_FRAC="0.5"

# W1 hygiene + W9 conditioning (constant across the chain).
export URCHIN_RESET_MODE="canonical"
export URCHIN_PRIMITIVE_ID="straight_roll"
export URCHIN_STYLE_ID="neutral"

# Task geometry — same Phase-1 geometry as run_urchin_v3_phase1_chain.sh.
export URCHIN_START_XY="-0.5,-0.5"
export URCHIN_GOAL_XY="0.5,0.5"   # base heading +45 deg
export URCHIN_EPISODE_S="8.0"
export URCHIN_GOAL_RADIUS="1.41"
export URCHIN_GOAL_DIR_BASE_DEG=""

# (run_id, timesteps, span_start_deg, span_end_deg, anneal_end_steps)
RUNS=(
  "w9_phase1_smoke:200000:30:90:200000"
  # "w9_1a_phase1_run1:1000000:30:90:1000000"   # W9.1a (collapsed 361->172). Dir kept as evidence; do NOT delete.
  "w9_1b_phase1_run1:1000000:30:90:1000000"   # W9.1b run1 (success: best 339.32, video-approved). Skip-loop reuses its result.json + final_checkpoint.pt as the run2 warmstart.
  "w9_1b_phase1_run2:2000000:180:360:2000000" # W9.1b run2: wider yaw 180->360 + frozen residual + BC reg beta=0.1 (anneal 2M); warmstart from 1b run1 final_checkpoint.pt
)

pick_handoff() {
  local dir="$1"
  if [[ -f "${dir}/final_checkpoint.pt" ]]; then
    echo "${dir}/final_checkpoint.pt"
  else
    echo ""
  fi
}

prev_ckpt=""
prev_best=""

for entry in "${RUNS[@]}"; do
  IFS=':' read -r run_id timesteps span_start span_end anneal_end <<< "${entry}"
  ckpt_dir="${BASE}/urchin_v3_${run_id}"

  # Smoke gate: block full runs until user has rendered + reviewed the smoke.
  if [[ "${run_id}" != "w9_phase1_smoke" ]]; then
    smoke_sentinel="${BASE}/urchin_v3_w9_phase1_smoke/.awaiting_review"
    if [[ -f "${smoke_sentinel}" ]]; then
      echo "[chain] PAUSE: smoke awaiting user video review."
      echo "[chain] Render checkpoint at:"
      echo "        ${BASE}/urchin_v3_w9_phase1_smoke/final_checkpoint.pt"
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
  echo "[chain] timesteps=${timesteps}"
  echo "[chain] goal_dir_span=${span_start}->${span_end} deg  anneal_end=${anneal_end}"
  echo "[chain] ckpt_dir=${ckpt_dir}"
  echo "[chain] cond=${URCHIN_PRIMITIVE_ID}:${URCHIN_STYLE_ID}  reset=${URCHIN_RESET_MODE}"

  if [[ "${run_id}" == "w9_phase1_smoke" ]]; then
    echo "[chain] warmstart: --bc-checkpoint ${BC_CKPT}"
    if [[ ! -f "${BC_CKPT}" ]]; then
      echo "[chain] ABORT: BC checkpoint not found: ${BC_CKPT}"
      exit 1
    fi
    URCHIN_GOAL_DIR_SPAN_START_DEG="${span_start}" \
    URCHIN_GOAL_DIR_SPAN_END_DEG="${span_end}" \
    URCHIN_GOAL_DIR_ANNEAL_START="0" \
    URCHIN_GOAL_DIR_ANNEAL_END="${anneal_end}" \
    "${PY}" "${TRAIN}" \
      --headless \
      --num-envs 64 \
      --timesteps "${timesteps}" \
      --bc-checkpoint "${BC_CKPT}" \
      --checkpoint-dir "${ckpt_dir}" \
      --run-id "${run_id}" \
      --learning-rate 5e-5 \
      --bc-post-log-std -1.6 \
      --bc-reg-coef 0.0 \
      --freeze-scaler-after-warmstart auto
    rc=$?
  else
    if [[ ! -f "${prev_ckpt}" ]]; then
      echo "[chain] ABORT: prev checkpoint not found: ${prev_ckpt}"
      exit 1
    fi
    echo "[chain] warmstart: --load-checkpoint ${prev_ckpt}"
    # W9.1a/W9.1b: freeze residual_scale at 0.3 for the W9.1 family.
    # Unset for anything else so we don't leak the override.
    if [[ "${run_id}" == "w9_1a_phase1_run1" || "${run_id}" == "w9_1b_phase1_run1" || "${run_id}" == "w9_1b_phase1_run2" ]]; then
      export URCHIN_RESIDUAL_SCALE_INIT=0.3
      export URCHIN_RESIDUAL_SCALE_FINAL=0.3
      echo "[chain] residual freeze: URCHIN_RESIDUAL_SCALE_INIT=${URCHIN_RESIDUAL_SCALE_INIT} URCHIN_RESIDUAL_SCALE_FINAL=${URCHIN_RESIDUAL_SCALE_FINAL}"
    else
      unset URCHIN_RESIDUAL_SCALE_INIT
      unset URCHIN_RESIDUAL_SCALE_FINAL
    fi
    # W9.1b: BC regularization (KL(pi||pi_bc)) at beta0=0.1 with anneal
    # spanning the full run length. W9.1a / others stay at beta=0.
    BC_REG_COEF="0.0"
    BC_REG_ANNEAL_FLAGS=()
    if [[ "${run_id}" == "w9_1b_phase1_run1" ]]; then
      BC_REG_COEF="0.1"
      BC_REG_ANNEAL_FLAGS=(--bc-reg-anneal-steps 1000000)
      echo "[chain] W9.1b: --bc-reg-coef ${BC_REG_COEF} --bc-reg-anneal-steps 1000000"
    elif [[ "${run_id}" == "w9_1b_phase1_run2" ]]; then
      BC_REG_COEF="0.1"
      BC_REG_ANNEAL_FLAGS=(--bc-reg-anneal-steps 2000000)
      echo "[chain] W9.1b run2: --bc-reg-coef ${BC_REG_COEF} --bc-reg-anneal-steps 2000000"
    fi
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
      --bc-reg-coef "${BC_REG_COEF}" \
      "${BC_REG_ANNEAL_FLAGS[@]}" \
      --freeze-scaler-after-warmstart auto
    rc=$?
  fi

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

  # Collapse gate — skip for the smoke (no Phase-1 baseline yet) and for
  # the first full run (widening distribution is expected to dip).
  if [[ -n "${prev_best}" && "${run_id}" != "w9_phase1_run1" && "${run_id}" != "w9_1a_phase1_run1" && "${run_id}" != "w9_1b_phase1_run1" && "${run_id}" != "w9_1b_phase1_run2" ]]; then
    if python -c "
import sys
best, prev, frac = float('${best}'), float('${prev_best}'), float('${COLLAPSE_FRAC}')
threshold = prev - frac * abs(prev)
sys.exit(0 if best >= threshold else 1)
"; then
      :
    else
      echo "[chain] ABORT: ${run_id} collapsed (best=${best} vs prev=${prev_best}, frac=${COLLAPSE_FRAC})"
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
  if [[ "${run_id}" == "w9_phase1_smoke" ]]; then
    touch "${ckpt_dir}/.awaiting_review"
    echo "[chain] SMOKE DONE. Render + review:"
    echo "        ${ckpt_dir}/final_checkpoint.pt"
    echo "[chain] Then: rm '${ckpt_dir}/.awaiting_review' and re-run this script."
  fi
done

echo "[chain] ALL RUNS COMPLETE"
