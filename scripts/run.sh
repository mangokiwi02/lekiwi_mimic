#!/usr/bin/env bash
# Run a python script with the Isaac stack venv (see setup_env.sh). Usage: scripts/run.sh scripts/record_sm.py --headless ...
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${LKM_VENV:-/c/lkm}"
PY="$VENV/Scripts/python.exe"; [ -x "$PY" ] || PY="$VENV/bin/python"
export LEISAAC_ASSETS_ROOT="$ROOT/third_party/leisaac/assets"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
cd "$ROOT" && exec "$PY" "$@"
