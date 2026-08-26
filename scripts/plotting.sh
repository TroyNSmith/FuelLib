#!/usr/bin/env bash
# Plot fuel composition or mixture properties.
#
# Thin wrapper around the fl-plt-comp / fl-plt-props console entry points
# since plotting requires the fuellib, pandas, and matplotlib Python packages.
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") <comp|props> [args...]"
    echo "  comp   Plot fuel composition (see: fl-plt-comp --help)"
    echo "  props  Plot mixture properties (see: fl-plt-props --help)"
    exit 1
}

[ $# -ge 1 ] || usage

mode="$1"
shift

case "$mode" in
    comp)
        exec fl-plt-comp "$@"
        ;;
    props)
        exec fl-plt-props "$@"
        ;;
    *)
        usage
        ;;
esac
