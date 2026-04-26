#!/usr/bin/env bash
# Short smoke-chain: exercises the same handoff + gate logic as
# run_urchin_v3_harden_chain.sh but with two 40k-step runs so the full
# BC seed -> run1 -> run2 chain completes in ~3-4 min on an RTX 3070.
#
# Purpose: verify the pipeline (train.py + chain script) end-to-end
# after the 2026-04-20 skrl 2.0 shape + inputs-key fixes. Not part of
# real training — delete the output dirs after verification.

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
BASE="workspace/checkpoints"
SEED_CKPT="${BASE}/urchin_v3_contactpush_bc/final_checkpoint.pt"
COLLAPSE_FRAC="0.5"

export URCHIN_START_XY="-0.5,-0.5"
export URCHIN_GOAL_XY="0.5,0.5"
export URCHIN_EPISODE_S="5.0"

RUNS=(
  "smoke_chain_a:40000"
  "smoke_chain_b:40000"
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
  run_id="${entry%%:*}"
  timesteps="${entry##*:}"
  ckpt_dir="${BASE}/urchin_v3_${run_id}"

  if [[ ! -f "${prev_ckpt}" ]]; then
    echo "[smoke] ABORT: prev checkpoint not found: ${prev_ckpt}"
    exit 1
  fi

  echo "[smoke] ====== ${run_id} ======"
  echo "[smoke] timesteps=${timesteps}  warmstart=${prev_ckpt}"

  "${PY}" "${TRAIN}" \
    --headless \
    --num-envs 64 \
    --timesteps "${timesteps}" \
    --load-checkpoint "${prev_ckpt}" \
    --checkpoint-dir "${ckpt_dir}" \
    --run-id "${run_id}" \
    --learning-rate 5e-5
  rc=$?

  result_json="${ckpt_dir}/result.json"
  if [[ $rc -ne 0 ]]; then
    echo "[smoke] ABORT: ${run_id} exit=${rc}"
    exit 1
  fi
  if [[ ! -f "${result_json}" ]]; then
    echo "[smoke] ABORT: missing ${result_json}"
    exit 1
  fi
  success=$(python -c "import json; print(json.load(open(r'${result_json}')).get('success', False))")
  if [[ "${success}" != "True" ]]; then
    echo "[smoke] ABORT: ${run_id} reported success=${success}"
    exit 1
  fi
  best=$(python -c "import json; print(json.load(open(r'${result_json}')).get('best_reward'))")
  echo "[smoke] ${run_id} DONE  best_reward=${best}"

  if [[ -n "${prev_best}" ]]; then
    if python -c "
import sys
best, prev, frac = float('${best}'), float('${prev_best}'), float('${COLLAPSE_FRAC}')
threshold = prev - frac * abs(prev)
sys.exit(0 if best >= threshold else 1)
"; then
      echo "[smoke] gate OK (best=${best} vs prev=${prev_best})"
    else
      echo "[smoke] ABORT: ${run_id} collapsed (best=${best} vs prev=${prev_best})"
      exit 1
    fi
  fi

  handoff=$(pick_handoff "${ckpt_dir}")
  if [[ -z "${handoff}" ]]; then
    echo "[smoke] ABORT: no checkpoint to hand off from ${ckpt_dir}"
    exit 1
  fi
  echo "[smoke] handoff -> ${handoff}"
  prev_ckpt="${handoff}"
  prev_best="${best}"
done

echo "[smoke] ALL RUNS COMPLETE"
