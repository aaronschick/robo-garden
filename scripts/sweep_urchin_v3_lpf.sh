#!/usr/bin/env bash
# LPF alpha sweep for urchin_v3 contactpush scripted roller.
# One Isaac Sim run per alpha; videos + logs go under workspace/videos/lpf_sweep_<ts>/.

set -u
set -o pipefail

PY="C:/isaac-venv/Scripts/python.exe"
SCRIPT="workspace/robots/urchin_v3/scripts/scripted_roll_video.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="workspace/videos/lpf_sweep_${STAMP}"
mkdir -p "${OUT_ROOT}"

ALPHAS=(1.0 0.5 0.3 0.15 0.03)

echo "[sweep] start ${STAMP}  out=${OUT_ROOT}"
echo "[sweep] alphas: ${ALPHAS[*]}"

SUMMARY="${OUT_ROOT}/summary.txt"
: > "${SUMMARY}"

for alpha in "${ALPHAS[@]}"; do
  tag="alpha_${alpha}"
  log="${OUT_ROOT}/${tag}.log"
  echo "[sweep] ====== alpha=${alpha} ======" | tee -a "${SUMMARY}"
  echo "[sweep] log: ${log}" | tee -a "${SUMMARY}"

  "${PY}" "${SCRIPT}" \
    --mode contactpush \
    --amplitude 2.0 \
    --baseline-hz 0 \
    --seconds 7.5 \
    --episodes 2 \
    --start-xy="-0.5,-0.5" \
    --goal-xy="0.5,0.5" \
    --lpf-alpha "${alpha}" \
    > "${log}" 2>&1 \
    && status="ok" || status="FAIL($?)"

  # Extract the video dir that this run created.
  video_dir=$(grep -oE 'scripted_roll_video[\\/]v3_[0-9_]+' "${log}" | head -1 | tr '\\' '/') || true
  # Pull travel / speed numbers from the tail (scripted_roll_video.py logs pos/vel each 30 steps).
  tail_info=$(grep -E 'pos=|Saved:' "${log}" | tail -8 || true)

  echo "[sweep] status=${status}  video_dir=${video_dir}" | tee -a "${SUMMARY}"
  echo "--- tail ---" | tee -a "${SUMMARY}"
  echo "${tail_info}" | tee -a "${SUMMARY}"
  echo | tee -a "${SUMMARY}"
done

echo "[sweep] done.  summary: ${SUMMARY}"
