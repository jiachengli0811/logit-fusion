#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
exec "${PYTHON_BIN}" scripts/bench/visualize.py --latest "$@"
