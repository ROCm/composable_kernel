#!/bin/bash
# Run inductor codegen tests
# This script is called from Jenkinsfile to reduce pipeline bytecode size

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CK_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${WORKSPACE:-/tmp}/ck-inductor-venv"
export UV_CACHE_DIR="${WORKSPACE:-/tmp}/.uv-cache"

cd "$CK_DIR"

echo "Setting up Python virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"

echo "Installing uv for faster package installation"
pip install uv

echo "Installing test dependencies"
uv pip install pytest build setuptools setuptools_scm

echo "Installing ck4inductor package"
uv pip install .

echo "Running inductor codegen tests"
python3 -m pytest python/test/test_gen_instances.py -v
