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

## GPU Training (WSL2)

Windows cannot run the JAX/MJX/Brax GPU physics path — Google does not ship
CUDA JAX wheels for Windows. The fix is to run the training backend inside
WSL2, which accesses your NVIDIA GPU through the Windows driver. Isaac Sim,
the Claude design loop, and the CLI all stay on Windows.

**One-time setup** (from a Windows PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_wsl2.ps1
```

The script will:
1. Verify `wsl.exe` is present (install via `wsl --install` from an elevated
   PowerShell if not).
2. Offer to install `Ubuntu-22.04` if no distro exists. You finish the Ubuntu
   first-run setup (pick a username + password), then re-run the script.
3. Preflight-check that bash actually executes inside the distro (a fresh
   distro can sometimes silently no-op until first-run setup completes).
4. Offer to set `Ubuntu-22.04` as the default WSL distro. **Important** if
   you also have Docker Desktop installed: its bundled `docker-desktop`
   distro is Alpine-based and has no `bash`, so leaving it as default
   breaks `wsl bash -c "..."`-style invocations.
5. Invoke `scripts/setup_wsl2.sh` inside the distro, which:
   - Installs `uv`, then `uv sync --upgrade --python 3.12`
   - Verifies the install with a sentinel string check (catches the silent
     `wsl.exe` exits-zero-with-no-output failure mode)
   - Verifies JAX sees the GPU
   - Optionally runs a Brax PPO smoke test on cartpole

Pass `-SkipSmokeTest` to skip the final Brax run (faster).
Pass `-Distro <name>` to target a different distro than `Ubuntu-22.04`.

**Where the WSL venv lives.** It is intentionally **not** at
`./.venv` (which is the Windows venv). Both Windows and WSL share the
project dir through `/mnt/c`, so a single `.venv` would cause Windows `uv`
to choke on Linux symlinks (`.venv/lib64`). Instead the WSL venv goes to
`$HOME/.cache/robo-garden/venv` on the Linux ext4 filesystem — both for
collision-avoidance and for ~5x faster site-packages reads vs NTFS-via-9P.
Override via `UV_PROJECT_ENVIRONMENT` if you need a different location.

**Headless GPU training runs** (from Windows PowerShell):

```powershell
uv run robo-garden --mode train --wsl --robot cartpole --timesteps 1000000 --envs 128
uv run robo-garden --mode train --wsl --robot go2_walker --timesteps 5000000 --envs 128
```

The `--wsl` flag translates the Windows project path to `/mnt/c/...`,
explicitly targets `Ubuntu-22.04` (override with `$env:ROBO_GARDEN_WSL_DISTRO`),
points uv at the ext4 venv via `UV_PROJECT_ENVIRONMENT`, then invokes the
trainer inside WSL and streams stdout back to the Windows terminal. If you
see `Algorithm: Brax PPO (GPU/JAX)` in the live panel, the GPU path is
working. If you see `Algorithm: Random rollout (no learning)`, something
upstream fell back — re-run setup or check the WSL torch/jax versions.

**Version pin notes** (don't edit `pyproject.toml` blindly here):
- `requires-python = ">=3.11,<3.13"` — `mujoco` 3.7 ships no wheel for
  cpython 3.14, and uv otherwise grabs the newest available, breaking the
  build with `MUJOCO_PATH not set`.
- `jax[cuda12]<0.9` — JAX 0.9 removed `jax.device_put_replicated` (which
  was deprecated in 0.8.1 in Nov 2025). Brax 0.14.x still calls it during
  PPO training, so we cap jax at the 0.8.x line until brax ships a fix.
- `torch>=2.3` is intentionally **not** capped on Linux even though torch
  2.10/2.11 wheels have an NCCL ABI mismatch (`undefined symbol:
  ncclDevCommDestroy`). The WSL training path uses Brax (JAX), not SB3
  (torch), so the broken torch is fine — and capping it forces JAX's
  cuDNN to an older minor version that JAX 0.8 then refuses to load.

**Claude-driven training in `--mode gym` on GPU.** Set
`ROBO_GARDEN_TRAIN_IN_WSL=1` before launching the gym session:

```powershell
$env:ROBO_GARDEN_TRAIN_IN_WSL="1"
uv run robo-garden --mode gym --approved go2_walker__flat_ground
```

When Claude calls the `train` tool, `handle_train` stages the job (robot
XML, environment MJCF, reward source code, hyperparameters) into
`workspace/_wsl_jobs/<run_id>/job.json`, subprocess-launches
`wsl.exe -d Ubuntu-22.04 -- uv run robo-garden --mode train --wsl-worker <job_dir>`,
and streams progress back over stdout (JSONL lines prefixed with
`__RG_PROGRESS__`) which it forwards verbatim to the Isaac Sim training
panel and run history. The worker writes `result.json` on exit; the
Windows side parses it and returns the normal tool-result dict to Claude.

The reward function's source is compiled **twice** inside the WSL worker:
1. As a normal NumPy callable (used by the SB3 / CPU fallback path).
2. As a JAX-traceable callable (used by Brax PPO / CUDA path), by
   re-exec'ing the code with `np`/`numpy` rebound to `jax.numpy` and
   `float`/`int`/`bool` rebound to tracer-safe pass-throughs.

If the JAX compile succeeds, training uses the Brax + MJX GPU path
(~20–50× faster). If it fails (e.g. Python `if` on obs values), the
worker falls back to SB3 with a clear note on stdout explaining the
fix. Most rewards Claude writes — arithmetic, `np.clip`, `np.exp`,
`np.mean`, `np.abs`, `np.where` — trace cleanly.

**Caveats for gym-mode-in-WSL:**
- Live policy rollouts into the Isaac viewport are disabled on this path
  (the trained policy params live in the WSL subprocess; marshaling them
  back each tick isn't wired yet). Run-progress metrics still flow.
- Override the distro with `$env:ROBO_GARDEN_WSL_DISTRO`.
- The staged job dir is preserved on failure — inspect
  `workspace/_wsl_jobs/<run_id>/` and re-run
  `uv run robo-garden --mode train --wsl-worker <job_dir>` from a WSL
  shell to get the full traceback.

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
