#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/python"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip pybind11 setuptools wheel numpy
.venv/bin/python setup.py build_ext --inplace
echo "Built: $ROOT/python/classicml_py*.so (use .venv/bin/python to import)"
