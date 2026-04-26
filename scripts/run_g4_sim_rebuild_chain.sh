#!/usr/bin/env bash
# G4 — Periodic-sim-rebuild probe for plan snazzy-forging-valiant.
#
# G2 ruled out scaler drift (forcing freeze-scaler=on still collapsed
# 338 → 171 → -5 → -45). G4 now isolates Isaac Sim / PhysX persistent
# state: does tearing down the env (via full Python process restart
# between chunks) prevent the collapse?
#
# Usage: bash scripts/run_g4_sim_rebuild_chain.sh <chunk_env_steps>
#   chunk sizes: 30000 (10 rebuilds), 75000 (4 rebuilds), 150000 (2)
#
# Each chunk: 1024 envs, lr=1e-12, freeze-scaler=on, bc-reg=0.0,
# det-eval every 25k, single rung yaw=0.297 goal_dir=0.0. Chunks
# chain via warmstart = prev chunk final_checkpoint. Total env-steps
# across chain == 300k (matching G2). Final eval score is the verdict:
#   * near G0 oracle (≥ 300 at yaw17) → sim rebuild breaks collapse → fix
#   * near G2 collapsed (≤ 0 at yaw17) → sim state is not the bug → P1

set -u

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <chunk_env_steps>"
  exit 2
fi

CHUNK=$1
TOTAL=300000
NUM_ENVS=1024
PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"

if [[ "$CHUNK" -le 0 ]] || [[ $((TOTAL % CHUNK)) -ne 0 ]]; then
  echo "[g4] ERROR: chunk ($CHUNK) must divide TOTAL ($TOTAL)"
  exit 2
fi

NCHUNKS=$(( TOTAL / CHUNK ))
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
CHAIN_ID="g4_rebuild_chunk${CHUNK}"
CKPT_ROOT="workspace/checkpoints/urchin_v3_${CHAIN_ID}"
AGG_LOG="workspace/_tasks_out/${CHAIN_ID}_aggregated.txt"

mkdir -p "${CKPT_ROOT}"
mkdir -p "workspace/_tasks_out"
: > "${AGG_LOG}"

echo "[g4] chain_id=${CHAIN_ID}" | tee -a "${AGG_LOG}"
echo "[g4] chunks=${NCHUNKS}  chunk_size=${CHUNK}  total=${TOTAL}  num_envs=${NUM_ENVS}" \
  | tee -a "${AGG_LOG}"
echo "[g4] seed=${SEED_CKPT}" | tee -a "${AGG_LOG}"

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
  echo "[g4] ABORT: seed checkpoint not found: ${SEED_CKPT}" | tee -a "${AGG_LOG}"
  exit 1
fi

t_chain_start=$(date +%s)
WARMSTART="${SEED_CKPT}"

for (( i=1; i<=NCHUNKS; i++ )); do
  RUN_ID="${CHAIN_ID}_c${i}"
  CKPT_DIR="${CKPT_ROOT}/c${i}"
  LOG="workspace/_tasks_out/${RUN_ID}.log"

  mkdir -p "${CKPT_DIR}"

  echo "" | tee -a "${AGG_LOG}"
  echo "=== chunk ${i}/${NCHUNKS}  warmstart=${WARMSTART}" | tee -a "${AGG_LOG}"
  echo "=== timesteps=${CHUNK}  ckpt_dir=${CKPT_DIR}  log=${LOG}" | tee -a "${AGG_LOG}"

  t_start=$(date +%s)
  "${PY}" "${TRAIN}" \
    --headless \
    --num-envs "${NUM_ENVS}" \
    --timesteps "${CHUNK}" \
    --load-checkpoint "${WARMSTART}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --run-id "${RUN_ID}" \
    --learning-rate 1e-12 \
    --bc-reg-coef 0.0 \
    --freeze-scaler-after-warmstart on \
    --det-eval-interval-sim-steps 20000 \
    --det-eval-yaw-spans-rad "0.297" \
    --det-eval-goal-dir-spans-rad "0.0" \
    > "${LOG}" 2>&1
  rc=$?
  t_end=$(date +%s)
  elapsed=$(( t_end - t_start ))
  echo "=== chunk ${i} rc=${rc}  elapsed=${elapsed}s" | tee -a "${AGG_LOG}"

  # Extract det-eval lines for this chunk, annotate with cumulative env-step offset.
  offset=$(( (i - 1) * CHUNK ))
  grep -a "det-eval rungs at step" "${LOG}" | while read -r line; do
    inner_step=$(echo "$line" | sed -n 's/.*at step \([0-9]*\):.*/\1/p')
    cum_step=$(( offset + inner_step ))
    echo "  chunk${i} cum=${cum_step} ${line}" | tee -a "${AGG_LOG}"
  done

  if [[ ${rc} -ne 0 ]]; then
    echo "[g4] ABORT: chunk ${i} failed rc=${rc}" | tee -a "${AGG_LOG}"
    exit ${rc}
  fi

  # train.py writes final_checkpoint.pt at end (line ~1212). That's the
  # canonical chain handoff. Fall back to skrl experiment dir if absent.
  if [[ -f "${CKPT_DIR}/final_checkpoint.pt" ]]; then
    WARMSTART="${CKPT_DIR}/final_checkpoint.pt"
  elif [[ -f "${CKPT_DIR}/${RUN_ID}/checkpoints/agent.pt" ]]; then
    WARMSTART="${CKPT_DIR}/${RUN_ID}/checkpoints/agent.pt"
  else
    echo "[g4] ABORT: no checkpoint from chunk ${i} at ${CKPT_DIR}" | tee -a "${AGG_LOG}"
    exit 1
  fi
  echo "=== chunk ${i} next warmstart: ${WARMSTART}" | tee -a "${AGG_LOG}"
done

t_chain_end=$(date +%s)
chain_elapsed=$(( t_chain_end - t_chain_start ))
echo "" | tee -a "${AGG_LOG}"
echo "[g4] CHAIN DONE  chain_id=${CHAIN_ID}  total_elapsed=${chain_elapsed}s" | tee -a "${AGG_LOG}"
echo "[g4] full aggregated log: ${AGG_LOG}"
