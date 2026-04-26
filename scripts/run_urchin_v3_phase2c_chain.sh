#!/usr/bin/env bash
# Phase 2c chain: heading-alignment + velocity bump on top of Phase 2b.
#
# Phase 2b run2 (best_reward=2059, +33% vs Phase 2) was video-reviewed as
# "slow and often indirect". Per-segment gain was diminishing
# (smoke 1622 -> run1 1854 -> run2 2059; +14% -> +11%) and mean_reward
# flattened at ~1M of run2's 2M budget — more training at the same reward
# landscape would add ~5-8% not fix the indirect path.
#
# Root cause: velocity_reward = cos(theta) * speed * weight rewards forward
# motion but does NOT penalise lateral/perpendicular motion, so orbit /
# wobble trajectories score well. Phase 2c introduces an explicit lateral
# penalty + bumps the velocity weight to further pressure "slow."
#
# Reward edits applied in urchin_env_cfg.py (2026-04-22):
#   L157: velocity_reward_weight         5.0 -> 8.0
#   L158 (new): lateral_velocity_penalty_weight = 3.0
#   _compute_reward (after vel_reward):
#     vel_lateral_vec = vel_xy - vel_toward.unsqueeze(-1) * unit_to_goal
#     vel_lateral_mag = vel_lateral_vec.norm(dim=-1)
#     lateral_pen     = -vel_lateral_mag * cfg.lateral_velocity_penalty_weight
#   return tuple: + lateral_pen
# No other env or reward changes vs Phase 2b.
#
# No scaler refit (obs distribution unchanged vs Phase 2b; warmstart
# propagates the frozen scaler stats per
# feedback_scaler_refit_not_bc_pretrain.md). No BC regularization (Phase 1
# anchor would pull toward ring-scale; we want arena-shape behaviour).
#
# Workflow mirrors Phase 2b:
#   1. pathB_phase2c_smoke (200k): warmstart from phase2b_run2/final_checkpoint.pt
#      under the NEW reward terms. Chain pauses on sentinel after smoke;
#      user renders --arena-mode --arena-view and reviews straightness +
#      speed before unblocking full runs.
#   2. pathB_phase2c_run1 (2M) warmstarts from smoke.
#   3. pathB_phase2c_run2 (2M) warmstarts from run1, collapse-gated vs run1.
#
# Chain handoff stays on final_checkpoint.pt per
# feedback_checkpoint_handoff.md. Do not change without user approval.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
BASE="workspace/checkpoints"
# Phase 2b run2 final (2026-04-22, best_reward=2059, video-reviewed as
# "slow and indirect" — goal-seeking correct, speed/straightness gap).
SEED_CKPT="${BASE}/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
# Abort if a run's best_reward drops more than this fraction below the
# prior run's best (relative to abs(prev_best)).
COLLAPSE_FRAC="0.5"

# --- Arena-scale task geometry (unchanged from Phase 2/2b) ---------------
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

# (run_id, timesteps, freeze_scaler_mode)
# "auto" freezes the scaler at warmstart stats (inherited from Phase 2b
# run2's frozen scaler via final_checkpoint.pt state_dict).
RUNS=(
  "pathB_phase2c_smoke:200000:auto"
  "pathB_phase2c_run1:2000000:auto"
  "pathB_phase2c_run2:2000000:auto"
)

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
  IFS=':' read -r run_id timesteps freeze_mode <<< "${entry}"
  ckpt_dir="${BASE}/urchin_v3_${run_id}"

  if [[ ! -f "${prev_ckpt}" ]]; then
    echo "[chain] ABORT: prev checkpoint not found: ${prev_ckpt}"
    exit 1
  fi

  # Smoke gate: block full runs until user has rendered + reviewed the smoke.
  if [[ "${run_id}" != "pathB_phase2c_smoke" ]]; then
    smoke_sentinel="${BASE}/urchin_v3_pathB_phase2c_smoke/.awaiting_review"
    if [[ -f "${smoke_sentinel}" ]]; then
      echo "[chain] PAUSE: smoke awaiting user video review."
      echo "[chain] Render (arena view) checkpoint at:"
      echo "        ${BASE}/urchin_v3_pathB_phase2c_smoke/final_checkpoint.pt"
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
  echo "[chain] arena=[-${URCHIN_ARENA_HALF_EXTENT},+${URCHIN_ARENA_HALF_EXTENT}]^2  episode_s=${URCHIN_EPISODE_S}  potential_scale=${URCHIN_POTENTIAL_SCALE_M}m"
  echo "[chain] reward_edits: vel_rw=8.0 lateral_pen_w=3.0 (on phase2b base)  bc_reg=off  freeze_scaler=${freeze_mode}"
  echo "[chain] ckpt_dir=${ckpt_dir}"

  "${PY}" "${TRAIN}" \
    --headless \
    --num-envs 64 \
    --timesteps "${timesteps}" \
    --load-checkpoint "${prev_ckpt}" \
    --checkpoint-dir "${ckpt_dir}" \
    --run-id "${run_id}" \
    --learning-rate 5e-5 \
    --bc-reg-coef 0.0 \
    --freeze-scaler-after-warmstart "${freeze_mode}"
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

  # Collapse gate — skip for the smoke run (reward scale shifted by new
  # lateral_pen term + weight bump, so absolute best_reward not directly
  # comparable to phase2b) and for the first full run (absorbing the new
  # reward landscape). run2 must not collapse vs run1.
  if [[ -n "${prev_best}" && "${run_id}" != "pathB_phase2c_run1" ]]; then
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
  if [[ "${run_id}" == "pathB_phase2c_smoke" ]]; then
    touch "${ckpt_dir}/.awaiting_review"
    echo "[chain] SMOKE DONE. Render (arena view) + review:"
    echo "        ${ckpt_dir}/final_checkpoint.pt"
    echo "[chain] Example render cmd:"
    echo "        ${PY} workspace/robots/urchin_v3/scripts/render_policy_video.py \\"
    echo "          --checkpoint ${ckpt_dir}/final_checkpoint.pt \\"
    echo "          --arena-mode --arena-view --arena-half-extent 2.0 \\"
    echo "          --potential-scale-m 1.5 --seconds 15 --episodes 4 \\"
    echo "          --tag phase2c_smoke_review"
    echo "[chain] Then: rm '${ckpt_dir}/.awaiting_review' and re-run this script."
  fi
done

echo "[chain] ALL RUNS COMPLETE"
