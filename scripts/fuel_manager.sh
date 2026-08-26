#!/usr/bin/env bash
# List all available fuels in the FuelLib library.
#
# Thin wrapper around the fl-fuels console entry point since listing fuels
# requires the fuellib Python package (metadata parsing, package data lookup).
set -euo pipefail

exec fl-fuels "$@"
