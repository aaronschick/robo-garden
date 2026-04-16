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

log()  { echo -e "${CYAN}==>${RESET} ${BOLD}$*${RESET}"; }
ok()   { echo -e "${GREEN}✓${RESET}  $*"; }
fail() { echo -e "${RED}✗${RESET}  $*" >&2; exit 1; }

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
    fail "nvidia-smi not found.
    WSL2 requires the Windows NVIDIA driver (>=470) — no separate Linux driver needed.
    See: https://docs.nvidia.com/cuda/wsl-user-guide/
    Install/update your Windows NVIDIA driver then restart WSL2."
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
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
log "Installing Python dependencies (jax[cuda12], mujoco-mjx, brax, ...)"
cd "$PROJECT_DIR"
uv sync
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
