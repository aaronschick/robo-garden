# Robo Garden

Claude-powered robot creation studio with simulation, training, and real-world deployment.

## What This Project Does

Users describe robots in natural language. Claude generates physically-valid MJCF robot descriptions, simulates them in MuJoCo, evaluates results, and iterates. The core loop is:

```
User prompt → Claude generates MJCF → Validate (physics + real components) → Simulate → Evaluate → Claude refines → Repeat
```

All robot designs must be **buildable with real-world components** — actual servos from the actuator catalog and real materials (3D-printable or machined).

## Architecture: 5 Spaces

1. **Robot Building Space** (`building/`) — Claude generates MJCF XML, validates against actuator/material databases
2. **Environment Building Space** (`environments/`) — Procedural terrain, obstacles, Gymnasium-compatible wrapping
3. **Training Gym** (`training/`) — MuJoCo MJX (local GPU) or Isaac Lab (cloud). RL with Brax PPO
4. **Incentive Design** (`rewards/`) — Eureka-style loop: Claude generates reward → short train → stats → Claude refines
5. **Claude Connector** (`claude/`) — Anthropic API tool-use agentic loop, 7 registered tools

## Key Technical Decisions

- **MuJoCo + MJX** is the primary local physics engine (not Isaac Sim — too heavy for 8GB VRAM)
- **Isaac Lab** is the secondary cloud-scale engine for A100/H100 training
- **MJCF** is the primary robot format (more expressive than URDF for actuators/contacts)
- **Brax PPO** for RL training (JAX-native, zero CPU-GPU transfer with MJX)
- **SB3** as fallback for CPU debugging
- Claude generates MJCF directly (no intermediate DSL)
- Reward functions are executable Python, sandboxed with restricted globals
- **Textual** TUI for terminal interface

## Hardware Context

Development target: **Razer Blade 15 Advanced, RTX 3070 (8GB VRAM)**
- MJX: 64-256 parallel environments locally
- Isaac Lab: 2048+ envs on cloud (A100/H100)
- Always use mixed precision where possible

## Project Structure

```
src/robo_garden/
├── cli.py              # Entry point: --mode tui|chat|train
├── config.py           # Paths, API keys, defaults
├── core/               # Robot, simulation, format validation
├── claude/             # Anthropic API, tool defs, handlers, prompts
├── building/           # Actuator/material DBs, MJCF validation
├── environments/       # Terrain gen, Gymnasium wrapper, domain randomization
├── training/           # MJX engine, vectorized env, algorithms, curriculum
├── rewards/            # Eureka loop, sandboxed reward runner, analysis
├── data/               # YAML catalogs (actuators, materials)
└── tui/                # Textual app
```

## Claude Tools (7 registered in claude/tools.py)

| Tool | What it does |
|------|-------------|
| `generate_robot` | Create MJCF + actuator/material assignments |
| `simulate` | Run MuJoCo simulation, return trajectories |
| `evaluate` | Compute metrics from simulation results |
| `generate_environment` | Create terrain/objects as MJCF |
| `generate_reward` | Write reward function Python code |
| `train` | Launch RL training via MJX |
| `query_catalog` | Search actuator/material/robot databases |

## Implementation Status

Phase 0-1 are scaffolded. Implementation needed:

- **Phase 1** (current): Wire up `generate_robot` → `simulate` end-to-end. Make the basic loop work.
- **Phase 2**: Flesh out actuator/material validation in `building/validator.py`
- **Phase 3**: Environment generation + Gymnasium wrapper
- **Phase 4**: `training/vectorized_env.py` (MJX `jax.vmap` — hardest module) + Brax PPO
- **Phase 5**: Eureka reward loop with Claude
- **Phase 6**: Textual TUI screens
- **Phase 7**: Curriculum learning

## Running

```bash
# Install deps
uv sync

# Interactive chat with Claude
uv run robo-garden --mode chat

# TUI (placeholder)
uv run robo-garden --mode tui

# Tests
uv run pytest tests/ -x
```

## Dependencies

- `mujoco` + `mujoco-mjx` — physics simulation
- `jax[cuda12]` — GPU acceleration for MJX
- `anthropic` — Claude API
- `brax` — JAX-native RL (PPO)
- `gymnasium` + `gymnasium-robotics` — RL environment API
- `stable-baselines3` — fallback RL algorithms
- `textual` — TUI framework
- `robot-descriptions` — 175+ reference robot models
- `pydantic`, `pyyaml`, `rich` — data/display

## Catalogs

Actuator catalogs in `data/actuators/`: Dynamixel (XL/XM/XH series), hobby servos (SG90, MG996R, DS3218), BLDC motors (ODrive, mjbots qdd100, MyActuator RMD-X8).

Material catalogs in `data/materials/`: 3D-printable (PLA, PETG, ABS, TPU, Nylon, CF composites), metals (aluminum 6061/7075, steel 304, carbon fiber tube).

## Key Files to Start With

1. `claude/client.py` — The agentic loop (tool dispatch)
2. `claude/tools.py` — Tool schemas (the Claude ↔ system contract)
3. `claude/tool_handlers.py` — Where tools connect to actual modules
4. `core/formats.py` — MJCF validation
5. `core/simulation.py` — MuJoCo step loop
6. `building/actuators.py` + `building/materials.py` — Catalog loading
