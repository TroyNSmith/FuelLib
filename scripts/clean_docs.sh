#!/usr/bin/env bash
# Remove Sphinx documentation build artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_ROOT/docs"
BUILD_DIR="$DOCS_DIR/_build"
GENERATED_DIR="$DOCS_DIR/generated"

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
    echo "Removed documentation build directory: $BUILD_DIR"
else
    echo "Build directory does not exist: $BUILD_DIR"
fi

if [ -d "$GENERATED_DIR" ]; then
    rm -rf "$GENERATED_DIR"
    echo "Removed generated documentation directory: $GENERATED_DIR"
else
    echo "Generated directory does not exist: $GENERATED_DIR"
fi
