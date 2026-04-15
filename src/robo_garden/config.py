"""Global configuration: paths, API keys, device settings."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)

ROBOTS_DIR = WORKSPACE_DIR / "robots"
ROBOTS_DIR.mkdir(exist_ok=True)

CHECKPOINTS_DIR = WORKSPACE_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

LOGS_DIR = WORKSPACE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent / "data"

# API configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Simulation defaults
DEFAULT_TIMESTEP = 0.002
DEFAULT_NUM_ENVS = 128  # Tuned for RTX 3070 8GB VRAM
MAX_EPISODE_STEPS = 1000
