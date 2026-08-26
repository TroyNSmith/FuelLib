#!/usr/bin/env bash
# Format all Python source code using Black.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

python -m black "$PROJECT_ROOT"
