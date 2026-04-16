# Isaac Sim Server — Setup Guide

The Isaac Sim server runs in a **separate Python 3.11 environment** alongside robo-garden (Python 3.12). They communicate over a local WebSocket on port 8765.

## System Requirements

- Windows 11 (x64)
- NVIDIA GPU with RTX Cores (RTX 3070+ recommended)
- Latest NVIDIA driver (Game Ready or Studio)
- Python 3.11 installed separately from the project's Python 3.12

## Installation

### 1. Enable Windows long paths (run PowerShell as Administrator)

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

Restart required.

### 2. Install Python 3.11

Download from https://www.python.org/downloads/ and install.

### 3. Create a dedicated Isaac Sim venv

```powershell
python3.11 -m venv C:\isaac-venv
C:\isaac-venv\Scripts\activate
```

### 4. Install Isaac Sim 5.x from NVIDIA PyPI

```powershell
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

> This is a large download (~15-20GB). Be patient.
>
> If `5.1.0` is not available, check the latest version:
> `pip index versions isaacsim --extra-index-url https://pypi.nvidia.com`

### 5. Install websockets inside the Isaac venv

```powershell
pip install websockets
```

### 6. Verify GPU is detected

```powershell
python -c "import omni.isaac.core; print('Isaac Sim OK')"
```

## Launching

```powershell
# From the repo root:
.\isaac_server\launch.ps1
```

Isaac Sim will open a viewport window. The WebSocket server starts on `ws://localhost:8765`.

Then in a separate terminal, start robo-garden:

```powershell
uv run robo-garden --mode chat
# Should print: "Isaac Sim bridge connected at ws://localhost:8765"
```

## VRAM Management (RTX 3070 — 8GB)

| Mode | Isaac Sim VRAM | Remaining for MuJoCo |
|------|---------------|----------------------|
| RayTracedLighting (default) | ~2-3 GB | ~5-6 GB |
| PathTracing | ~4-5 GB | ~3-4 GB |
| Headless | 0 GB | 8 GB |

For active training (MJX at 128 envs), stop the Isaac Sim server or set:

```powershell
$env:ISAAC_BRIDGE_ENABLED = "off"
uv run robo-garden --mode chat
```

## Troubleshooting

**`pip install isaacsim` fails**: Check that Python version is exactly 3.11.x and long paths are enabled.

**Viewport opens but robot doesn't appear**: Check that `workspace\robots\<name>.xml` exists after calling `generate_robot` in chat.

**Port 8765 already in use**: Change with `.\launch.ps1 -Port 8766` and set `$env:ISAAC_BRIDGE_URL = "ws://localhost:8766"` before starting robo-garden.
