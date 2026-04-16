"""Global configuration: paths, API keys, device settings."""

import os
from pathlib import Path

# Load .env file from project root if present (before reading env vars below)
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _val = _val.strip()
            # Strip matching quotes
            if len(_val) >= 2 and _val[0] == _val[-1] and _val[0] in ('"', "'"):
                _val = _val[1:-1]
            else:
                # Remove inline comments only for unquoted values
                _comment_idx = _val.find(" #")
                if _comment_idx >= 0:
                    _val = _val[:_comment_idx].rstrip()
            os.environ.setdefault(_key.strip(), _val)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)

ROBOTS_DIR = WORKSPACE_DIR / "robots"
ROBOTS_DIR.mkdir(exist_ok=True)

ENVIRONMENTS_DIR = WORKSPACE_DIR / "environments"
ENVIRONMENTS_DIR.mkdir(exist_ok=True)

CHECKPOINTS_DIR = WORKSPACE_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

LOGS_DIR = WORKSPACE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

RENDERS_DIR = WORKSPACE_DIR / "renders"
RENDERS_DIR.mkdir(exist_ok=True)

PROMPTS_DIR = WORKSPACE_DIR / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent / "data"

# API configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Isaac Sim bridge configuration
# ISAAC_BRIDGE_ENABLED: "auto" = try at startup (graceful degradation if unavailable)
#                        "on"   = require connection, warn if unavailable
#                        "off"  = never attempt connection (CI, headless)
ISAAC_BRIDGE_URL = os.environ.get("ISAAC_BRIDGE_URL", "ws://localhost:8765")
ISAAC_BRIDGE_ENABLED = os.environ.get("ISAAC_BRIDGE_ENABLED", "auto")

# Simulation defaults
DEFAULT_TIMESTEP = 0.002
DEFAULT_NUM_ENVS = 128  # Tuned for RTX 3070 8GB VRAM
MAX_EPISODE_STEPS = 1000
