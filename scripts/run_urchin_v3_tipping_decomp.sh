#!/usr/bin/env bash
# W5 (2026-04-25) — Quasi-static tipping decomposition driver.
#
# Runs the six-condition tipping decomposition test once per condition
# (one pytest invocation per condition, so each gets a *fresh*
# articulation — see test docstring for the rationale), then collates
# the per-condition JSONs into workspace/_tasks_out/w5_tipping/report.md
# via workspace/scratch/w5_postprocess.py.
#
# Conditions (research report §"Experimental Plan", Table row 3):
#   lean_only      rear_push=0   front_reach=1   front_retract=0
#   retract_only   rear_push=0   front_reach=0   front_retract=1
#   push_only      rear_push=1   front_reach=0   front_retract=0
#   lean_retract   rear_push=0   front_reach=1   front_retract=1
#   retract_push   rear_push=1   front_reach=0   front_retract=1
#   full_triplet   rear_push=1   front_reach=1   front_retract=1
#
# Decision rule (machine-readable, in verdict.json):
#   retract_only.net_forward_rotation_rad / full_triplet.net_forward_rotation_rad
#     < 0.25 -> HALT (controller hypothesis suspect)
#     >= 0.25 -> PROCEED to W6 (locomotion atlas)
#
# Usage (from a WSL shell, with Isaac Lab installed):
#   bash scripts/run_urchin_v3_tipping_decomp.sh
#   bash scripts/run_urchin_v3_tipping_decomp.sh retract_only full_triplet
#
# Env overrides:
#   W5_PYTEST          override pytest command (default: "uv run pytest")
#   W5_TIMEOUT_S       per-condition pytest timeout (default 600)
#   W5_OUT_DIR         output dir (default workspace/_tasks_out/w5_tipping)

set -u

ALL_CONDITIONS=(
  "lean_only"
  "retract_only"
  "push_only"
  "lean_retract"
  "retract_push"
  "full_triplet"
)

if [[ $# -gt 0 ]]; then
  CONDITIONS=("$@")
else
  CONDITIONS=("${ALL_CONDITIONS[@]}")
fi

W5_OUT_DIR="${W5_OUT_DIR:-workspace/_tasks_out/w5_tipping}"
W5_PYTEST="${W5_PYTEST:-uv run pytest}"
W5_TIMEOUT_S="${W5_TIMEOUT_S:-600}"

# Canonical reset is required for fresh-articulation semantics.
# Oracle off + residual scale 1.0 gives us a clean engine -> joint-target
# pipeline (engine output is the *only* source of panel commands).
export URCHIN_RESET_MODE="canonical"
export URCHIN_ORACLE_AMPLITUDE="0.0"
export URCHIN_RESIDUAL_SCALE_INIT="1.0"
export URCHIN_RESIDUAL_SCALE_FINAL="1.0"

# Determinism (best-effort; PhysX still has its own RNG).
export PYTHONHASHSEED="42"
export TORCH_DETERMINISTIC="1"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

mkdir -p "${W5_OUT_DIR}"

# Truncate any prior summary.csv so the per-invocation appends start fresh.
SUMMARY_CSV="${W5_OUT_DIR}/summary.csv"
if [[ -f "${SUMMARY_CSV}" ]]; then
  echo "[w5] removing prior ${SUMMARY_CSV}"
  rm -f "${SUMMARY_CSV}"
fi

echo "[w5] conditions: ${CONDITIONS[*]}"
echo "[w5] out_dir:    ${W5_OUT_DIR}"
echo "[w5] pytest:     ${W5_PYTEST}"

OVERALL_RC=0
for cond in "${CONDITIONS[@]}"; do
  log="${W5_OUT_DIR}/${cond}.log"
  echo "[w5][${cond}] START $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "[w5][${cond}] log=${log}"

  URCHIN_W5_CONDITION="${cond}" \
    timeout "${W5_TIMEOUT_S}" \
    ${W5_PYTEST} tests/test_substrate/test_tipping_decomposition.py -x -s \
      2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}

  echo "[w5][${cond}] END $(date -u +'%Y-%m-%dT%H:%M:%SZ') rc=${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[w5][${cond}] FAILED rc=${rc}; continuing to next condition."
    OVERALL_RC=1
  fi
done

# Collate. The post-processor handles missing conditions gracefully so
# this still emits a partial report.md / verdict.json on partial failure.
echo "[w5] running post-processor"
uv run python workspace/scratch/w5_postprocess.py \
  --input-dir "${W5_OUT_DIR}" || OVERALL_RC=1

echo "[w5] DONE overall_rc=${OVERALL_RC}"
echo "[w5] report:   ${W5_OUT_DIR}/report.md"
echo "[w5] verdict:  ${W5_OUT_DIR}/verdict.json"
exit "${OVERALL_RC}"
