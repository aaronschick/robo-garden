#!/usr/bin/env bash
# =============================================================================
# setup_wsl2.sh — Configure robo-garden for GPU training inside WSL2
#
# Run this ONCE from inside a WSL2 terminal (Ubuntu 22.04 recommended):
#
#   bash /mnt/c/Users/aaron/Documents/repositories/robo-garden/scripts/setup_wsl2.sh
#
# What it does:
#   1. Verifies the NVIDIA GPU is accessible (drivers come from Windows)
#   2. Installs uv (if not present)
#   3. Runs `uv sync` to install the Linux GPU stack:
#        jax[cuda12], mujoco-mjx, brax — all gated sys_platform == 'linux'
#   4. Verifies JAX sees the GPU
#   5. Runs a 4096-step smoke test with Brax PPO
#
# After this script succeeds, launch training with:
#   uv run robo-garden --mode train --robot cartpole --timesteps 1000000
#
# Or from Windows Terminal (no WSL2 shell needed):
#   uv run robo-garden --mode train --robot cartpole --timesteps 1000000 --wsl
# =============================================================================

set -euo pipefail
BOLD='\033[1m'; CYAN='\033[36m'; GREEN='\033[32m'; RED='\033[31m'; RESET='\033[0m'

# When invoked from setup_wsl2.ps1 the parent PowerShell pipes our stdout
# line-by-line. Force unbuffered python output and UTF-8 encoding so progress
# shows up in the Windows console without getting stuck in a buffer or mangled
# by the default Windows codepage.
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

log()  { echo -e "${CYAN}==>${RESET} ${BOLD}$*${RESET}"; }
ok()   { echo -e "${GREEN}[OK]${RESET}  $*"; }
fail() { echo -e "${RED}[FAIL]${RESET}  $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo ""
echo -e "${BOLD}Robo Garden — WSL2 GPU Setup${RESET}"
echo "Project: $PROJECT_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. Check WSL2
# ---------------------------------------------------------------------------
log "Checking WSL2 environment"
if grep -qEi "(microsoft|wsl)" /proc/version 2>/dev/null; then
    ok "Running inside WSL2"
else
    echo "  Warning: /proc/version doesn't show WSL2 markers. Continuing anyway."
fi

# ---------------------------------------------------------------------------
# 2. Check NVIDIA GPU
# ---------------------------------------------------------------------------
log "Checking NVIDIA GPU"
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found inside WSL2.

WSL2 gets its GPU access through the Windows NVIDIA driver (no separate Linux
driver needed). If nvidia-smi is missing, usually one of these is true:

  1. Your Windows NVIDIA driver is older than 470. Upgrade at nvidia.com.
  2. You installed 'nvidia-cuda-toolkit' via apt — don't. The WSL2 workflow
     ships drivers from Windows; apt's toolkit shadows them. Remove with:
        sudo apt remove --purge nvidia-cuda-toolkit
  3. WSL2 needs a restart after a fresh Windows driver install. From Windows:
        wsl --shutdown
     Then re-run this script.

Docs: https://docs.nvidia.com/cuda/wsl-user-guide/"
fi
if ! GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1); then
    fail "nvidia-smi is installed but failed to run:
${GPU_INFO}

Try 'wsl --shutdown' from Windows then re-run this script."
fi
ok "GPU detected: $GPU_INFO"

# ---------------------------------------------------------------------------
# 3. Install uv
# ---------------------------------------------------------------------------
log "Checking uv"
if command -v uv &>/dev/null; then
    ok "uv already installed: $(uv --version)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to current shell PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    ok "uv installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 4. Install Python dependencies
# ---------------------------------------------------------------------------
# Put the venv on the ext4 filesystem ($HOME), not /mnt/c, for two reasons:
#   1. Windows and WSL share the project dir via /mnt/c, so a single .venv
#      there would collide between the two platforms (Windows uv tries to
#      rebuild it and fails on Linux symlinks like .venv/lib64).
#   2. ext4 is dramatically faster than NTFS-through-9P for site-packages
#      reads. Cuts `uv sync` install time from ~10 min to under 2 min on a
#      warm cache.
#
# We pin Python 3.12 explicitly because mujoco 3.7 has no wheels for 3.14;
# without the pin uv grabs 3.14 (newest), tries to build mujoco from source,
# and fails on MUJOCO_PATH. The Windows side also runs 3.12, keeping parity.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/robo-garden/venv}"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

log "Installing Python dependencies (jax[cuda12], mujoco-mjx, brax, ...)"
log "Venv location: $UV_PROJECT_ENVIRONMENT"
cd "$PROJECT_DIR"
# --upgrade forces re-resolution against the current pyproject so that
# transitive CUDA libs (notably nvidia-cudnn-cu12) match what the
# currently-resolved jax expects. Without this, an older lockfile from a
# previous failed run can leave cuDNN one minor version behind what
# `jax[cuda12]` needs at runtime, producing "RuntimeError: cuDNN not found."
uv sync --python 3.12 --upgrade
ok "uv sync complete"

# ---------------------------------------------------------------------------
# 5. Verify JAX GPU
# ---------------------------------------------------------------------------
log "Verifying JAX GPU backend"
JAX_RESULT=$(uv run python -c "
import jax
devices = jax.devices()
backend = jax.default_backend()
print(f'{backend}|{devices}')
")
JAX_BACKEND="${JAX_RESULT%%|*}"
JAX_DEVICES="${JAX_RESULT#*|}"

if [[ "$JAX_BACKEND" == "gpu" ]]; then
    ok "JAX backend: gpu  — devices: $JAX_DEVICES"
elif [[ "$JAX_BACKEND" == "cuda" ]]; then
    ok "JAX backend: cuda — devices: $JAX_DEVICES"
else
    echo ""
    echo -e "${RED}JAX is not using the GPU (backend='$JAX_BACKEND').${RESET}"
    echo ""
    echo "Possible causes:"
    echo "  • jax[cuda12] wheel installed but CUDA libraries not found."
    echo "    Try: uv run python -c \"import jax; jax.devices()\""
    echo "         and look for 'Unable to load CUDA' in the output."
    echo ""
    echo "  • Your NVIDIA driver is too old (requires >=470 for WSL2 CUDA)."
    echo "    Run:  nvidia-smi   and check Driver Version."
    echo ""
    echo "  • CUDA 12.x mismatch — the JAX wheel bundles its own CUDA runtime"
    echo "    but the libstdc++ version in WSL2 may be too old."
    echo "    Try: sudo apt install -y libstdc++6"
    echo ""
    fail "JAX GPU not available. Fix the issue above and re-run this script."
fi

# ---------------------------------------------------------------------------
# 6. Smoke test — short Brax PPO run
# ---------------------------------------------------------------------------
if [[ "${ROBO_GARDEN_SKIP_SMOKE:-0}" == "1" ]]; then
    ok "Skipping Brax PPO smoke test (ROBO_GARDEN_SKIP_SMOKE=1)"
    echo ""
    echo -e "${GREEN}${BOLD}Setup complete (smoke test skipped).${RESET}"
    exit 0
fi

log "Running Brax PPO smoke test (4096 timesteps)"
uv run python - <<'PYEOF'
import sys
from robo_garden.training.mujoco_engine import MuJoCoMJXEngine
from robo_garden.training.models import TrainingConfig
from robo_garden.training.gym_env import cartpole_reward_jax, cartpole_done_jax
from robo_garden.config import ROBOTS_DIR

xml = (ROBOTS_DIR / "cartpole.xml").read_text()
config = TrainingConfig(num_envs=64, total_timesteps=4096, max_episode_steps=200)
engine = MuJoCoMJXEngine()
engine.setup(xml, "", config)

steps_reported = []
def cb(step, metrics):
    r = metrics.get("eval/episode_reward", 0)
    steps_reported.append(step)
    print(f"  step={step:,}  reward={r:.3f}")

result = engine.train(
    "",
    jax_reward_fn=cartpole_reward_jax,
    jax_done_fn=cartpole_done_jax,
    callback=cb,
)
engine.cleanup()

if not steps_reported:
    print("ERROR: No progress callbacks received — Brax PPO may have silently failed.")
    sys.exit(1)

print(f"\nSmoke test OK: best_reward={result.best_reward:.3f}  time={result.training_time_seconds:.1f}s")
PYEOF

ok "Smoke test passed"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}Setup complete!${RESET}"
echo ""
echo "Run training from WSL2:"
echo "  cd '$PROJECT_DIR'"
echo "  uv run robo-garden --mode train --robot cartpole --timesteps 1000000 --envs 128"
echo ""
echo "Or from Windows Terminal (without opening a WSL2 shell):"
echo "  uv run robo-garden --mode train --robot cartpole --timesteps 1000000 --envs 128 --wsl"
echo ""
