# launch.ps1 — Start the robo-garden Isaac Sim server
#
# Prerequisites: Isaac Sim installed in C:\isaac-venv (see README.md)
# Usage: .\isaac_server\launch.ps1

param(
    [string]$VenvPath = "C:/isaac-venv",
    [string]$Port = "8765"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$ServerScript = Join-Path $ScriptDir "server.py"

$PythonExe = Join-Path $VenvPath "Scripts/python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Isaac Sim venv not found at $VenvPath"
    Write-Error "Run the setup steps in isaac_server\README.md first."
    exit 1
}

if (-not (Test-Path $ServerScript)) {
    Write-Error "server.py not found at $ServerScript"
    exit 1
}

Write-Host "Starting Isaac Sim server..."
Write-Host "  Venv:   $VenvPath"
Write-Host "  Script: $ServerScript"
Write-Host "  Port:   $Port"
Write-Host ""

$env:ISAAC_SERVER_PORT = $Port

& $PythonExe $ServerScript
