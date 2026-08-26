#!/usr/bin/env bash
# Build Sphinx documentation for FuelLib.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_ROOT/docs"

echo "Building documentation from $DOCS_DIR"
echo "Command: sphinx-build -M html $DOCS_DIR $DOCS_DIR/_build"
echo

sphinx-build -M html "$DOCS_DIR" "$DOCS_DIR/_build"

echo
echo "================================================================================"
echo "Documentation built successfully!"
echo "View the documentation at: $DOCS_DIR/_build/html/index.html"
echo "================================================================================"
