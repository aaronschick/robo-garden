#!/usr/bin/env bash
# W4 (2026-04-25) — Substrate identification harness for urchin_v3.
#
# Runs the four sim-only physics-ID tests (single-panel step, paired-panel
# step, coast-down on flat ground, incline static-slip onset) and collates
# their CSVs into workspace/_tasks_out/w4_substrate_id/substrate_baseline.json
# — the deliverable consumed by W6 (locomotion atlas) and W7 (slip-aware
# scheduler thresholds).
#
# WHY: deep-research report Table row 1-2 flags actuator characterization
# and friction-map identification as the HIGHEST priority. Without these
# baseline numbers, W6 and W7 have no real anchor.
#
# WSL launch (recommended — Isaac Sim only runs in WSL):
#
#   On Windows PowerShell:
#     wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/c/Users/aaron/Documents/repositories/robo-garden && \
#       URCHIN_RESET_MODE=canonical bash scripts/run_urchin_v3_substrate_id.sh"
#
#   Or from a WSL bash shell directly:
#     cd /mnt/c/Users/aaron/Documents/repositories/robo-garden
#     URCHIN_RESET_MODE=canonical bash scripts/run_urchin_v3_substrate_id.sh
#
# Direct (Windows + Isaac venv) is also supported but requires Isaac Lab
# binaries on the host — usually not the case for this project per
# CLAUDE.md "GPU Training (WSL2)".
#
# Env overrides:
#   W4_OUT_DIR     output root (default workspace/_tasks_out/w4_substrate_id)
#   W4_PYTHON      python interpreter (default tries Isaac venv then uv)
#   W4_TESTS       which tests to run, space-sep (default: all four)
#                  one or more of: actuator friction
#
# Output layout:
#   workspace/_tasks_out/w4_substrate_id/
#     ├── actuator_response/
#     │   ├── single_panel.csv
#     │   └── paired_panel.csv
#     ├── friction/
#     │   ├── coast_down.csv          (per-step trace)
#     │   ├── coast_down.summary.csv  (fit + tau + mu_r)
#     │   ├── incline_slip.csv        (per-step trace)
#     │   └── incline_slip.summary.csv
#     ├── substrate_baseline.json     (W6/W7 deliverable)
#     ├── report.md
#     └── run.log

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W4_OUT_DIR="${W4_OUT_DIR:-${REPO_ROOT}/workspace/_tasks_out/w4_substrate_id}"
W4_TESTS="${W4_TESTS:-actuator friction}"

mkdir -p "${W4_OUT_DIR}"
LOG="${W4_OUT_DIR}/run.log"
: > "${LOG}"

log() {
  echo "[w4-substrate] $*" | tee -a "${LOG}"
}

# Pick a python. On WSL we expect uv-managed env; on Windows we expect the
# Isaac venv at C:/isaac-venv/Scripts/python.exe (matches the rest of the
# urchin_v3 chain scripts).
pick_python() {
  if [[ -n "${W4_PYTHON:-}" ]]; then
    echo "${W4_PYTHON}"
    return
  fi
  if [[ -x "/mnt/c/isaac-venv/Scripts/python.exe" ]]; then
    echo "/mnt/c/isaac-venv/Scripts/python.exe"
    return
  fi
  if [[ -x "C:/isaac-venv/Scripts/python.exe" ]]; then
    echo "C:/isaac-venv/Scripts/python.exe"
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "uv run python"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
  echo "python3"
}

PY="$(pick_python)"
log "REPO_ROOT=${REPO_ROOT}"
log "W4_OUT_DIR=${W4_OUT_DIR}"
log "W4_TESTS=${W4_TESTS}"
log "PY=${PY}"

# Force canonical reset for clean state on every Isaac instantiation.
export URCHIN_RESET_MODE="canonical"
log "URCHIN_RESET_MODE=${URCHIN_RESET_MODE}"

cd "${REPO_ROOT}" || { log "FATAL: cannot cd to ${REPO_ROOT}"; exit 2; }

OVERALL_RC=0

run_test_file() {
  local label="$1"
  local file_path="$2"
  log "===== START ${label} ====="
  log "  file: ${file_path}"
  local t0
  t0=$(date +%s)
  set +e
  ${PY} -m pytest "${file_path}" -x -s 2>&1 | tee -a "${LOG}"
  local rc=${PIPESTATUS[0]}
  set -e || true
  local t1
  t1=$(date +%s)
  log "===== END   ${label} rc=${rc} elapsed=$((t1 - t0))s ====="
  return "${rc}"
}

for test_group in ${W4_TESTS}; do
  case "${test_group}" in
    actuator)
      if ! run_test_file "actuator_response" \
          "tests/test_substrate/test_actuator_response.py"; then
        OVERALL_RC=1
        log "WARN: actuator_response test failed; continuing."
      fi
      ;;
    friction)
      if ! run_test_file "friction_map" \
          "tests/test_substrate/test_friction_map.py"; then
        OVERALL_RC=1
        log "WARN: friction_map test failed; continuing."
      fi
      ;;
    *)
      log "WARN: unknown test group '${test_group}', skipping."
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Collate CSVs into substrate_baseline.json (the W6/W7 deliverable) +
# render report.md. Both delegated to standalone helpers under
# scripts/_w4_helpers/ so heredoc quote/path-translation issues across
# Git Bash / WSL bash / Linux bash all disappear -- argv is plain paths.
# ---------------------------------------------------------------------------

log "collating substrate_baseline.json + report.md ..."

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
NOW="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# Translate the output dir to a Windows path if Python is the Windows
# interpreter (it can't read `/c/Users/...` Git Bash paths). cygpath is
# present on Git Bash; on Linux/WSL we leave the path alone.
PY_OUT_DIR="${W4_OUT_DIR}"
if [[ "${PY}" == *.exe || "${PY}" == */isaac-venv/* ]]; then
  if command -v cygpath >/dev/null 2>&1; then
    PY_OUT_DIR="$(cygpath -m "${W4_OUT_DIR}" 2>/dev/null || echo "${W4_OUT_DIR}")"
  fi
fi

if ! ${PY} "${REPO_ROOT}/scripts/_w4_helpers/collate_baseline.py" \
      --out-dir "${PY_OUT_DIR}" \
      --git-commit "${GIT_COMMIT}" \
      --generated-at "${NOW}" 2>&1 | tee -a "${LOG}"; then
  OVERALL_RC=1
fi

if ! ${PY} "${REPO_ROOT}/scripts/_w4_helpers/write_report.py" \
      --out-dir "${PY_OUT_DIR}" 2>&1 | tee -a "${LOG}"; then
  OVERALL_RC=1
fi

log "DONE overall_rc=${OVERALL_RC}"
log "  baseline: ${W4_OUT_DIR}/substrate_baseline.json"
log "  report:   ${W4_OUT_DIR}/report.md"
log "  log:      ${LOG}"

exit "${OVERALL_RC}"
