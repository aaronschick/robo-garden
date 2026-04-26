#!/usr/bin/env bash
# W6 (2026-04-25) — Urchin v3 open-loop locomotion atlas driver.
#
# Pipeline:
#   1. Generate LHS samples (Windows Python OK -- pure stdlib + numpy).
#   2. Run the Isaac harness inside WSL (Isaac Lab on Windows is not
#      supported per CLAUDE.md "GPU Training (WSL2)").
#   3. Analyse outputs (heatmaps + top10 + report) — Windows Python OK.
#
# Smoke usage (~16 samples, end-to-end validation):
#   W6_NUM_SAMPLES=16 bash scripts/run_urchin_v3_locomotion_atlas.sh
#
# Full sweep (~256 samples, hours on Isaac):
#   bash scripts/run_urchin_v3_locomotion_atlas.sh
#
# Env overrides:
#   W6_NUM_SAMPLES   number of LHS samples (default 256)
#   W6_NUM_ENVS      Isaac parallel envs per sample (default 4)
#   W6_EPISODE_S     episode duration in sec (default 4.0)
#   W6_SEED          LHS RNG seed (default 42)
#   W6_DISTRO        WSL distro name (default Ubuntu-22.04)
#   W6_SKIP_RUN      if set, skips step 2 (useful when only re-analysing)
#
# Note on path translation: the WSL invocation cd's into the project
# under /mnt/c so all relative paths (workspace/...) resolve identically
# in Windows and WSL.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

W6_NUM_SAMPLES="${W6_NUM_SAMPLES:-256}"
W6_NUM_ENVS="${W6_NUM_ENVS:-4}"
W6_EPISODE_S="${W6_EPISODE_S:-4.0}"
W6_SEED="${W6_SEED:-42}"
W6_DISTRO="${W6_DISTRO:-Ubuntu-22.04}"

OUT_DIR="workspace/_tasks_out/w6_atlas"
SAMPLES_CSV="${OUT_DIR}/lhs_samples.csv"
ATLAS_BASE="${OUT_DIR}/atlas"

mkdir -p "${OUT_DIR}"

echo "[w6] num_samples=${W6_NUM_SAMPLES} num_envs=${W6_NUM_ENVS} "\
"episode_s=${W6_EPISODE_S} seed=${W6_SEED} distro=${W6_DISTRO}"

# ---------- 1. LHS samples (Windows-side) ----------
echo "[w6] step 1/3 — generate LHS samples -> ${SAMPLES_CSV}"
uv run python workspace/scratch/w6_atlas_lhs.py \
    --num-samples "${W6_NUM_SAMPLES}" \
    --seed "${W6_SEED}" \
    --output "${SAMPLES_CSV}"
rc=$?
if [[ ${rc} -ne 0 ]]; then
    echo "[w6] ERROR: LHS generator failed rc=${rc}"
    exit ${rc}
fi

# ---------- 2. Isaac harness ----------
# Prefer the Windows isaac-venv (where Isaac Lab actually lives for urchin_v3,
# matching scripts/run_urchin_v3_phase*_chain.sh and run_e12_override_repro.sh).
# Fall back to WSL only if isaac-venv is missing (legacy Brax/JAX path).
if [[ -n "${W6_SKIP_RUN:-}" ]]; then
    echo "[w6] step 2/3 — SKIPPED (W6_SKIP_RUN set)"
else
    ISAAC_PY="${W6_ISAAC_PY:-/c/isaac-venv/Scripts/python.exe}"
    if [[ -x "${ISAAC_PY}" ]]; then
        echo "[w6] step 2/3 — Isaac sweep via ${ISAAC_PY}"
        URCHIN_RESET_MODE=canonical "${ISAAC_PY}" workspace/scratch/w6_atlas_run.py \
            --samples "${SAMPLES_CSV}" \
            --output "${ATLAS_BASE}" \
            --num-envs ${W6_NUM_ENVS} \
            --episode-s ${W6_EPISODE_S} \
            --seed ${W6_SEED}
        rc=$?
        if [[ ${rc} -ne 0 ]]; then
            echo "[w6] ERROR: Isaac harness failed rc=${rc}"
            exit ${rc}
        fi
    else
        echo "[w6] ERROR: isaac-venv not found at ${ISAAC_PY}"
        echo "[w6]        Override with W6_ISAAC_PY=/path/to/python.exe or install Isaac Lab."
        exit 2
    fi
fi

# ---------- 3. Analysis (Windows-side) ----------
echo "[w6] step 3/3 — analysis (heatmaps + top10 + report)"
uv run python workspace/scratch/w6_atlas_analysis.py \
    --atlas "${ATLAS_BASE}.parquet" \
    --output-dir "${OUT_DIR}"
rc=$?
if [[ ${rc} -ne 0 ]]; then
    # Fallback: maybe Isaac wrote CSV instead of parquet (no pyarrow).
    uv run python workspace/scratch/w6_atlas_analysis.py \
        --atlas "${ATLAS_BASE}.csv" \
        --output-dir "${OUT_DIR}"
    rc=$?
fi
if [[ ${rc} -ne 0 ]]; then
    echo "[w6] ERROR: analysis failed rc=${rc}"
    exit ${rc}
fi

echo "[w6] DONE — outputs in ${OUT_DIR}"
ls -la "${OUT_DIR}" 2>/dev/null || true
