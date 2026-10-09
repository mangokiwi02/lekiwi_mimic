#!/usr/bin/env bash
# Installs the LeIsaac-pinned stack: Isaac Sim 5.1, torch 2.7 cu128, Isaac Lab (LeIsaac submodule), LeIsaac, lekiwi_mimic.
#
# Windows: Isaac Sim's extension cache contains paths of ~200 chars, which overflow MAX_PATH (260) under
# <project>/.pixi/envs/default when LongPathsEnabled is off. So the stack goes into a venv at a short prefix
# (default C:/lkm), created from pixi's Python 3.11. On Linux, or with long paths enabled, any prefix works.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${LKM_VENV:-/c/lkm}"
[ -x "$VENV/Scripts/python.exe" ] || [ -x "$VENV/bin/python" ] || pixi run python -m venv "$VENV"
PY="$VENV/Scripts/python.exe"; [ -x "$PY" ] || PY="$VENV/bin/python"
PIP="$PY -m pip"
LAB=third_party/leisaac/dependencies/IsaacLab/source
$PIP install --upgrade pip
$PIP install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$PIP install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
for ext in isaaclab isaaclab_assets isaaclab_tasks isaaclab_mimic; do
  $PIP install -e "$LAB/$ext"
done
$PIP install -e third_party/leisaac/source/leisaac
$PIP install -e source/lekiwi_mimic
$PIP install pytest h5py
# isaaclab extras may override torch; pin back
$PIP install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
