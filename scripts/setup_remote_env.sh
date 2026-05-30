#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.10)"
else
  PYTHON_BIN=""
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if ! command -v micromamba >/dev/null 2>&1; then
    echo "python3.10/python3.11 and micromamba were not found." >&2
    echo "Install a modern Python first, then rerun this script." >&2
    exit 2
  fi
  micromamba create -y -p "$ROOT_DIR/.venv-conda" python=3.11 pip
  # shellcheck disable=SC1091
  source "$(micromamba shell hook --shell=bash)"
  micromamba activate "$ROOT_DIR/.venv-conda"
  python -m pip install --upgrade pip wheel setuptools
else
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
fi

pip install --index-url https://download.pytorch.org/whl/cu124 torch
pip install -e ".[all]"

echo "Environment ready. Activate it with:"
echo "  source $ROOT_DIR/.venv/bin/activate"
