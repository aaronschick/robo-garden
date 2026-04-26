#!/usr/bin/env bash
# Phase 2 chain: arena-scale random spawn + random goal.
# Each reset samples spawn AND goal uniformly in [-a, +a]^2 (a=2.0 m),
# with rejection-resample on post-scaled distance < goal_radius*margin.
# Episodes extended to 15 s; potential-field decay widened to 1.5 m.
#
# Scaler refit design (Phase 2): Phase 1's RunningStandardScaler was fit
# on a ring-sampled obs distribution (~1.41 m radius around a fixed
# spawn). Arena-scale obs have wider `to_goal`, wider contact patterns,
# etc. Smoke segment refits the scaler on the fresh arena BC dataset
# (POLICY WEIGHTS UNTOUCHED — see train.py --scaler-refit-dataset),
# then freezes at arena-refit stats. Subsequent segments inherit the
# arena-fit scaler via warmstart load + auto-freeze. This keeps the
# Phase 1 policy's behavior intact while fixing input normalisation.
#
# Workflow:
#   1. pathB_phase2_smoke (200k steps): warmstart from Phase 1 run2,
#      --scaler-refit-dataset on fresh arena BC obs,
#      --freeze-scaler-after-warmstart=on (freezes AFTER refit).
#   2. After smoke, the chain pauses on a sentinel file. User renders
#      with --arena-mode --arena-view, reviews, deletes sentinel to
#      authorize the full runs.
#   3. pathB_phase2_run1 (2M) warmstarts from smoke, scaler auto-freezes
#      at smoke's (arena-refit) stats.
#   4. pathB_phase2_run2 (2M) warmstarts from run1, scaler auto-freezes.
#
# Chain handoff stays on final_checkpoint.pt per
# `feedback_checkpoint_handoff.md`. Do not change without user approval.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
BASE="workspace/checkpoints"
# Phase 1 run2's final (user video-approved 2026-04-22).
SEED_CKPT="${BASE}/urchin_v3_pathB_phase1_run2/final_checkpoint.pt"
# Fresh arena-scale BC dataset (recorded with --sampling-mode arena).
# Used ONLY for scaler refit in smoke — actions are ignored. Policy
# weights come from SEED_CKPT via warmstart load.
BC_DATASET="workspace/datasets/urchin_v3_bc_sh.h5"
# Abort if a run's best_reward drops more than this fraction below the
# prior run's best (relative to abs(prev_best)).
COLLAPSE_FRAC="0.5"

# --- Arena-scale task geometry -------------------------------------------
# URCHIN_START_XY / URCHIN_GOAL_XY become inert when sampling_mode="arena"
# (they're only read as the Phase-1 base-heading anchor, which ring mode
# uses). Leave at Phase-1 defaults so an accidental fallback to ring mode
# still picks up the +45° Phase-1 heading.
export URCHIN_START_XY="-0.5,-0.5"
export URCHIN_GOAL_XY="0.5,0.5"
export URCHIN_EPISODE_S="15.0"

export URCHIN_GOAL_SAMPLING_MODE="arena"
export URCHIN_ARENA_HALF_EXTENT="2.0"
export URCHIN_MIN_SPAWN_GOAL_DIST="0.3"
export URCHIN_RESAMPLE_DIST_MARGIN="1.2"
export URCHIN_POTENTIAL_SCALE_M="1.5"
# No distance curriculum during Phase 2 (full arena from step 0). Kept
# as pair of env vars for consistency with env-cfg plumbing.
export URCHIN_DIST_SCALE_START="1.0"
export URCHIN_DIST_SCALE_END="1.0"

# (run_id, timesteps, use_scaler_refit, freeze_scaler_mode)
# use_scaler_refit=1 triggers --scaler-refit-dataset on smoke only;
# freeze_scaler_mode "on" locks in arena-refit stats after the refit.
# Subsequent segments inherit arena-refit scaler via warmstart load
# and "auto" freezes at that load-time state.
RUNS=(
  "pathB_phase2_smoke:200000:1:on"
  "pathB_phase2_run1:2000000:0:auto"
  "pathB_phase2_run2:2000000:0:auto"
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
  IFS=':' read -r run_id timesteps use_scaler_refit freeze_mode <<< "${entry}"
  ckpt_dir="${BASE}/urchin_v3_${run_id}"

  if [[ ! -f "${prev_ckpt}" ]]; then
    echo "[chain] ABORT: prev checkpoint not found: ${prev_ckpt}"
    exit 1
  fi

  # Smoke gate: block full runs until user has rendered + reviewed the smoke.
  if [[ "${run_id}" != "pathB_phase2_smoke" ]]; then
    smoke_sentinel="${BASE}/urchin_v3_pathB_phase2_smoke/.awaiting_review"
    if [[ -f "${smoke_sentinel}" ]]; then
      echo "[chain] PAUSE: smoke awaiting user video review."
      echo "[chain] Render (arena view) checkpoint at:"
      echo "        ${BASE}/urchin_v3_pathB_phase2_smoke/final_checkpoint.pt"
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
  echo "[chain] scaler_refit=${use_scaler_refit}  freeze_scaler=${freeze_mode}  ckpt_dir=${ckpt_dir}"

  extra_args=()
  if [[ "${use_scaler_refit}" == "1" ]]; then
    if [[ ! -f "${BC_DATASET}" ]]; then
      echo "[chain] ABORT: scaler-refit dataset missing: ${BC_DATASET}"
      exit 1
    fi
    extra_args+=(--scaler-refit-dataset "${BC_DATASET}")
  fi

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
    --freeze-scaler-after-warmstart "${freeze_mode}" \
    "${extra_args[@]}"
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

  # Collapse gate — skip for the smoke run (distribution shift vs Phase 1
  # expected to dip) and for the first full run (still absorbing the
  # wider arena + longer episode). run2 must not collapse vs run1.
  if [[ -n "${prev_best}" && "${run_id}" != "pathB_phase2_run1" ]]; then
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
  if [[ "${run_id}" == "pathB_phase2_smoke" ]]; then
    touch "${ckpt_dir}/.awaiting_review"
    echo "[chain] SMOKE DONE. Render (arena view) + review:"
    echo "        ${ckpt_dir}/final_checkpoint.pt"
    echo "[chain] Example render cmd:"
    echo "        ${PY} workspace/robots/urchin_v3/scripts/render_policy_video.py \\"
    echo "          --checkpoint ${ckpt_dir}/final_checkpoint.pt \\"
    echo "          --arena-mode --arena-view --arena-half-extent 2.0 \\"
    echo "          --potential-scale-m 1.5 --seconds 15 --episodes 4 \\"
    echo "          --tag phase2_smoke_review"
    echo "[chain] Then: rm '${ckpt_dir}/.awaiting_review' and re-run this script."
  fi
done

echo "[chain] ALL RUNS COMPLETE"
