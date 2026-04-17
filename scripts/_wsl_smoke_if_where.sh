#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/robo-garden/venv}"
cd /mnt/c/Users/aaron/Documents/repositories/robo-garden
uv run python scripts/_smoke_if_where.py
