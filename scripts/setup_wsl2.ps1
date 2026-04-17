<#
.SYNOPSIS
    One-time Windows bootstrapper for GPU training via WSL2 + MJX/Brax.

.DESCRIPTION
    Robo Garden's training engine takes a JAX/MJX/Brax path on Linux and a
    slower SB3+CPU-physics path on Windows (JAX does not ship CUDA wheels for
    Windows). Running the training backend inside WSL2 gives you native access
    to your NVIDIA GPU through the Windows driver while everything else
    (Isaac Sim viewport, Claude design loop, CLI) stays on Windows.

    This script walks you through the one-time setup:

      1. Verifies wsl.exe is present (Windows feature)
      2. Verifies / installs an Ubuntu-22.04 distro
      3. Runs scripts/setup_wsl2.sh inside that distro (installs jax[cuda12],
         mujoco-mjx, brax; runs a Brax PPO smoke test on cartpole)

    Run it from a regular PowerShell window — it will prompt for elevation
    only if it needs to install the Ubuntu distro.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_wsl2.ps1

.EXAMPLE
    # After setup, kick off GPU training from Windows:
    uv run robo-garden --mode train --wsl --robot cartpole --timesteps 1000000
#>

[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-22.04",
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "OK  $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "!!  $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "XX  $msg" -ForegroundColor Red; exit 1 }

$ProjectDir = Split-Path -Parent $PSScriptRoot
Write-Host ""
Write-Host "Robo Garden - WSL2 GPU Setup" -ForegroundColor White
Write-Host "Project: $ProjectDir"
Write-Host ""

# ---------------------------------------------------------------------------
# 1. wsl.exe available?
# ---------------------------------------------------------------------------
Write-Step "Checking wsl.exe"
$wslExe = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wslExe) {
    Write-Fail @"
wsl.exe was not found on PATH.

WSL2 is a Windows feature. To enable it:
  1. Open an elevated PowerShell (Run as Administrator)
  2. Run:   wsl --install
  3. Reboot when prompted
  4. Re-run this script
"@
}
Write-Ok "wsl.exe: $($wslExe.Path)"

# ---------------------------------------------------------------------------
# 2. Is the requested distro installed?
# ---------------------------------------------------------------------------
# `wsl --list --quiet` emits UTF-16LE with NULs, which garbles string compares
# against ASCII distro names. Force UTF-8 interpretation via /? parsing:
# running `wsl --list --verbose` and capturing raw bytes is simpler.
Write-Step "Checking for WSL distro: $Distro"

# Convert UTF-16 output from wsl.exe to UTF-8 so -match works normally.
$prevEncoding = [Console]::OutputEncoding
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $distroList = & wsl.exe --list --quiet 2>$null
} finally {
    [Console]::OutputEncoding = $prevEncoding
}

$distroList = $distroList -join "`n"
$distroFound = $distroList -match [Regex]::Escape($Distro)

if (-not $distroFound) {
    Write-Warn "$Distro is not installed yet."
    Write-Host ""
    Write-Host "Installing $Distro requires administrator privileges." -ForegroundColor Yellow
    Write-Host "You'll get a UAC prompt, then a new console for Ubuntu's first-run setup"
    Write-Host "(it will ask you to pick a username and password)."
    Write-Host ""
    $ans = Read-Host "Install $Distro now? [y/N]"
    if ($ans -notmatch "^[yY]") {
        Write-Fail "Cannot proceed without a Linux distro. Re-run when you're ready."
    }

    Write-Step "Running: wsl --install -d $Distro (elevated)"
    $installCmd = "wsl --install -d $Distro; Read-Host 'Press Enter after the Ubuntu window finishes first-run setup (username/password)'"
    try {
        Start-Process -FilePath "powershell.exe" `
            -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $installCmd `
            -Verb RunAs -Wait
    } catch {
        Write-Fail "Failed to elevate for WSL install: $_"
    }

    Write-Host ""
    Write-Host "After you finish the Ubuntu first-run setup (username + password prompt" -ForegroundColor Yellow
    Write-Host "in the new console), re-run this script to finish the GPU setup." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\setup_wsl2.ps1"
    Write-Host ""
    exit 0
}
Write-Ok "$Distro is installed."

# ---------------------------------------------------------------------------
# 2a. Preflight: make sure we can actually execute bash in the distro.
# ---------------------------------------------------------------------------
# The first `wsl -d <distro> -- ...` after a fresh install can sometimes exit
# zero without producing any output while the distro is still finishing its
# provisioning. We catch that silent-failure mode by looking for a sentinel
# string in stdout rather than just trusting the exit code.
Write-Step "Preflight: verifying bash is reachable inside $Distro"
$preflight = & wsl.exe -d $Distro -- bash -c "echo __ROBO_GARDEN_PREFLIGHT_OK__; whoami; uname -a" 2>&1 | Out-String
if ($preflight -notmatch "__ROBO_GARDEN_PREFLIGHT_OK__") {
    Write-Host "Preflight output was:" -ForegroundColor DarkGray
    Write-Host $preflight -ForegroundColor DarkGray
    Write-Fail @"
Could not execute bash inside $Distro. Common causes:
  1. The distro was just installed and needs its first-run user setup.
     Open a $Distro shell once (Start menu or `wsl -d $Distro`), finish
     the username/password prompt, then re-run this script.
  2. Your default WSL distro is something else (e.g. docker-desktop) and
     has shadowed the one we're targeting.
     Run:  wsl --set-default $Distro
"@
}
Write-Ok "Preflight OK — $($preflight.Trim().Split("`n")[1])"

# ---------------------------------------------------------------------------
# 2b. Is $Distro the default? If not, offer to make it so — the cli.py
#     _launch_wsl_training path explicitly passes -d, but any tool that calls
#     plain `wsl bash -c "..."` will pick the default distro, and
#     docker-desktop (Alpine) will fail with "bash: not found".
# ---------------------------------------------------------------------------
Write-Step "Checking WSL default distro"
$prevEncoding = [Console]::OutputEncoding
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $verboseList = (& wsl.exe --list --verbose 2>$null) -join "`n"
} finally {
    [Console]::OutputEncoding = $prevEncoding
}
# Default distro is marked with an asterisk in `wsl -l -v` output.
$defaultLine = ($verboseList -split "`n" | Where-Object { $_ -match "^\s*\*" }) -join ""
if ($defaultLine -match [Regex]::Escape($Distro)) {
    Write-Ok "$Distro is already the default distro."
} else {
    Write-Warn "Default WSL distro is not $Distro. That means 'wsl bash -c ...' without -d will NOT use $Distro."
    $ans = Read-Host "Set $Distro as the default WSL distro now? [Y/n]"
    if ($ans -match "^[nN]") {
        Write-Warn "Skipping. You can do it later with:  wsl --set-default $Distro"
    } else {
        & wsl.exe --set-default $Distro 2>&1 | Out-String | Write-Host
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Set default distro to $Distro."
        } else {
            Write-Warn "Could not set default distro (exit $LASTEXITCODE). Continuing anyway."
        }
    }
}

# ---------------------------------------------------------------------------
# 3. Convert project path to /mnt/c/... form for WSL
# ---------------------------------------------------------------------------
Write-Step "Translating project path for WSL"
$projWsl = $ProjectDir -replace '\\', '/'
if ($projWsl -match '^([A-Za-z]):(.*)$') {
    $driveLetter = $Matches[1].ToLower()
    $rest = $Matches[2]
    $projWsl = "/mnt/$driveLetter$rest"
}
Write-Ok "WSL project path: $projWsl"

# ---------------------------------------------------------------------------
# 4. Run setup_wsl2.sh inside the distro
# ---------------------------------------------------------------------------
Write-Step "Running setup_wsl2.sh inside $Distro (this can take 5-15 minutes first time)"
Write-Host ""

$bashScript = "$projWsl/scripts/setup_wsl2.sh"
# Source user profile so uv / cargo-installed binaries are on PATH.
# Keep PYTHONIOENCODING=utf-8 so Rich output doesn't crash on the pipe back to Windows.
# The final 'echo __ROBO_GARDEN_SETUP_DONE__' is a sentinel: we only trust the
# setup if this string makes it back to Windows. Guards against a known
# silent-success mode where wsl.exe exits 0 without running the payload.
$envPrefix = 'export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1; ' +
             'source "$HOME/.profile" 2>/dev/null; ' +
             'source "$HOME/.cargo/env" 2>/dev/null; ' +
             'export PATH="$HOME/.local/bin:$PATH"; '

$skipFlag = if ($SkipSmokeTest) { "ROBO_GARDEN_SKIP_SMOKE=1 " } else { "" }
$fullCmd = $envPrefix + $skipFlag + "bash '$bashScript' && echo __ROBO_GARDEN_SETUP_DONE__"

# Stream output line-by-line (2>&1 merges stderr for live visibility).
# Capture into $setupOutput so we can sentinel-check the tail afterwards.
$setupOutput = New-Object System.Text.StringBuilder
& wsl.exe -d $Distro -- bash -c $fullCmd 2>&1 | ForEach-Object {
    Write-Host $_
    [void]$setupOutput.AppendLine($_)
}
$rc = $LASTEXITCODE
$setupOutputStr = $setupOutput.ToString()

if ($rc -ne 0) {
    Write-Host ""
    Write-Fail "setup_wsl2.sh exited with code $rc. Fix the issue above and re-run this script."
}

if ($setupOutputStr -notmatch "__ROBO_GARDEN_SETUP_DONE__") {
    Write-Host ""
    Write-Fail @"
setup_wsl2.sh exited 0 but did not reach its final step — the expected
sentinel '__ROBO_GARDEN_SETUP_DONE__' was not in the output.

This usually means the wsl.exe subprocess returned without actually running
the bash script (a known issue right after a fresh distro install). Try:

  1. Open a $Distro shell manually (Start menu -> Ubuntu, or 'wsl -d $Distro')
  2. If it prompts for username/password, complete that setup
  3. Exit the shell and re-run this script
"@
}

# Sentinel post-check: uv should now be installed inside $Distro.
Write-Step "Post-check: verifying uv is installed inside $Distro"
$uvCheck = & wsl.exe -d $Distro -- bash -c 'source "$HOME/.profile" 2>/dev/null; source "$HOME/.cargo/env" 2>/dev/null; export PATH="$HOME/.local/bin:$PATH"; command -v uv && uv --version || echo __NO_UV__' 2>&1 | Out-String
if ($uvCheck -match "__NO_UV__") {
    Write-Fail @"
uv is not installed inside $Distro even though the setup script reported
success. Something is wrong with the PATH or uv install inside WSL.

Debug manually with:
  wsl -d $Distro
  bash /mnt/c/Users/aaron/Documents/repositories/robo-garden/scripts/setup_wsl2.sh
"@
}
Write-Ok "uv is installed: $($uvCheck.Trim().Split("`n")[-1])"

Write-Host ""
Write-Ok "GPU setup complete."
Write-Host ""
Write-Host "Next steps (from this Windows PowerShell):" -ForegroundColor White
Write-Host ""
Write-Host "  # Headless GPU training (cartpole smoke run):"
Write-Host "  uv run robo-garden --mode train --wsl --robot cartpole --timesteps 1000000 --envs 128"
Write-Host ""
Write-Host "  # Headless GPU training (your approved robot):"
Write-Host "  uv run robo-garden --mode train --wsl --robot go2_walker --timesteps 5000000 --envs 128"
Write-Host ""
Write-Host "  # Claude-driven training with WSL GPU backend (opt-in):"
Write-Host "  `$env:ROBO_GARDEN_TRAIN_IN_WSL = '1'"
Write-Host "  uv run robo-garden --mode gym --approved <manifest-name>"
Write-Host ""
