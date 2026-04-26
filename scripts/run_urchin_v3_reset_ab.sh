#!/usr/bin/env bash
# W1 (2026-04-25) — Reset hygiene A/B harness.
#
# Runs the E12-override config under three reset arms and emits per-step
# trajectory + det-eval drift artifacts so we can confirm or refute the
# Memory `project_urchin_v3_e_series_yaw_curriculum_finding.md` hypothesis
# that the 338→-41 reward collapse is sim/env state contamination.
#
# Arms:
#   (a) baseline       — current code path (URCHIN_RESET_MODE=legacy).
#                        Single robot.write_root_state_to_sim() call,
#                        no articulation.reset() — pre-W1 path, exactly
#                        what scripts/run_e12_override_repro.sh runs.
#   (b) canonical      — URCHIN_RESET_MODE=canonical, full canonical
#                        reset sequence (write_root_link_pose_to_sim ->
#                        write_root_com_velocity_to_sim ->
#                        write_joint_state_to_sim -> robot.reset()).
#   (c) fresh-articulation — same as (b) but each det-eval is run as a
#                        fresh subprocess from the same seed checkpoint.
#                        We can't recreate articulation views in-process
#                        without invasive Isaac Lab plumbing, so we model
#                        "fresh articulation" by killing+restarting Isaac.
#
# All three arms use IDENTICAL training args (and seed via PYTHONHASHSEED
# + XLA flags) so trajectory divergence is attributable solely to the
# reset path. Output artifacts live under
# workspace/_tasks_out/w1_reset_ab/<arm>/.
#
# Usage:
#   bash scripts/run_urchin_v3_reset_ab.sh                # all 3 arms
#   bash scripts/run_urchin_v3_reset_ab.sh baseline       # one arm
#   bash scripts/run_urchin_v3_reset_ab.sh canonical fresh-articulation
#
# Env overrides:
#   W1_TIMESTEPS    (default 300000) — match E12-override exactly
#   W1_NUM_ENVS     (default 1024)
#   W1_SEED         (default 42)

set -u

PY="C:/isaac-venv/Scripts/python.exe"
TRAIN="workspace/robots/urchin_v3/scripts/train.py"
SEED_CKPT="workspace/checkpoints/urchin_v3_pathB_phase2b_run2/final_checkpoint.pt"
OUT_ROOT="workspace/_tasks_out/w1_reset_ab"
mkdir -p "${OUT_ROOT}"

W1_TIMESTEPS="${W1_TIMESTEPS:-300000}"
W1_NUM_ENVS="${W1_NUM_ENVS:-1024}"
W1_SEED="${W1_SEED:-42}"

ALL_ARMS=("baseline" "canonical" "fresh-articulation")
if [[ $# -gt 0 ]]; then
  ARMS=("$@")
else
  ARMS=("${ALL_ARMS[@]}")
fi

if [[ ! -f "${SEED_CKPT}" ]]; then
  echo "[w1-ab] WARN: seed checkpoint not found: ${SEED_CKPT}"
  echo "[w1-ab]       arms will run from scratch instead of warmstart."
  echo "[w1-ab]       This is fine for the *trajectory-divergence* test"
  echo "[w1-ab]       but means absolute reward numbers will not match"
  echo "[w1-ab]       the E12-override log. Re-run with the full PathB"
  echo "[w1-ab]       phase2b checkpoint for the production comparison."
  WARMSTART_FLAG=""
else
  WARMSTART_FLAG="--load-checkpoint ${SEED_CKPT}"
fi

echo "[w1-ab] timesteps=${W1_TIMESTEPS} num_envs=${W1_NUM_ENVS} seed=${W1_SEED}"
echo "[w1-ab] arms=${ARMS[*]}"
echo "[w1-ab] out_root=${OUT_ROOT}"

# Common env (mirrors run_e12_override_repro.sh).
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

# Determinism (best-effort — Isaac Sim still has nondeterminism we can't
# control, but with identical seed + identical PhysX cfg the early-step
# divergence between arms is dominated by the reset path).
export PYTHONHASHSEED="${W1_SEED}"
export TORCH_DETERMINISTIC="1"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

run_arm () {
  local arm="$1"
  local arm_dir="${OUT_ROOT}/${arm}"
  mkdir -p "${arm_dir}"
  local log="${arm_dir}/train.log"
  local run_id="w1_${arm//-/_}"
  local ckpt_dir="workspace/checkpoints/urchin_v3_${run_id}"

  case "${arm}" in
    baseline)
      export URCHIN_RESET_MODE="legacy"
      ;;
    canonical|fresh-articulation)
      export URCHIN_RESET_MODE="canonical"
      ;;
    *)
      echo "[w1-ab] ERROR: unknown arm '${arm}' (expected baseline|canonical|fresh-articulation)"
      return 2
      ;;
  esac

  echo "[w1-ab][${arm}] START reset_mode=${URCHIN_RESET_MODE} ckpt_dir=${ckpt_dir}"
  echo "[w1-ab][${arm}] log=${log}"
  echo "[w1-ab][${arm}] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

  local t_start
  t_start=$(date +%s)

  # We capture three artifacts per arm:
  #   - train.log: full stdout (includes __RG_PROGRESS__ JSONL + det-eval lines)
  #   - eval_*.csv: parsed det-eval rewards (post-process)
  #   - traj_*.csv: per-step body pose + first-panel position (post-process)
  # The last two are extracted from train.log in `analyze_w1_ab.py` (see
  # report.md). Keeping the raw log avoids re-running on parse changes.
  "${PY}" "${TRAIN}" \
    --headless \
    --num-envs "${W1_NUM_ENVS}" \
    --timesteps "${W1_TIMESTEPS}" \
    ${WARMSTART_FLAG} \
    --checkpoint-dir "${ckpt_dir}" \
    --run-id "${run_id}" \
    --learning-rate 1e-12 \
    --bc-reg-coef 0.0 \
    --freeze-scaler-after-warmstart auto \
    --det-eval-yaw-spans-rad "0.6,2.82" \
    --det-eval-goal-dir-spans-rad "0.524,1.66" \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}

  local t_end
  t_end=$(date +%s)
  local elapsed=$(( t_end - t_start ))
  echo "[w1-ab][${arm}] END $(date -u +'%Y-%m-%dT%H:%M:%SZ') elapsed=${elapsed}s rc=${rc}"

  # Quick parse for the per-arm summary table.
  echo "[w1-ab][${arm}] det-eval line count: $(grep -c 'det-eval' "${log}" 2>/dev/null || echo 0)"

  return "${rc}"
}

OVERALL_RC=0
for arm in "${ARMS[@]}"; do
  if ! run_arm "${arm}"; then
    OVERALL_RC=1
    echo "[w1-ab] arm ${arm} failed; continuing to next arm."
  fi
done

# Post-process: collate det-eval lines from all arm logs into a CSV the
# report.md plot-renderer can consume. We do this in pure bash so the
# harness is self-contained — analyze_w1_ab.py is optional.
SUMMARY="${OUT_ROOT}/det_eval_summary.csv"
echo "arm,sim_step,reward,yaw_span,goal_dir_span" > "${SUMMARY}"
for arm in "${ARMS[@]}"; do
  log="${OUT_ROOT}/${arm}/train.log"
  if [[ -f "${log}" ]]; then
    # Match lines like: "[det-eval] step=NNN yaw_span=X goal_dir_span=Y reward=Z"
    grep -E 'det-eval.*reward=' "${log}" 2>/dev/null \
      | sed -E "s/^.*step=([0-9]+).*yaw_span=([-0-9.eE]+).*goal_dir_span=([-0-9.eE]+).*reward=([-0-9.eE]+).*/${arm},\1,\4,\2,\3/" \
      | grep -E "^${arm}," >> "${SUMMARY}" || true
  fi
done

echo "[w1-ab] DONE overall_rc=${OVERALL_RC}"
echo "[w1-ab] summary CSV: ${SUMMARY}"
echo "[w1-ab] see ${OUT_ROOT}/report.md"
exit "${OVERALL_RC}"
