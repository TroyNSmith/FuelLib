#!/usr/bin/env bash
# Convert transport properties for combustion simulations.
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") eps2K <epsilon_j_per_mol>"
    exit 1
}

[ $# -eq 2 ] || usage

mode="$1"
value="$2"

# Physical constants (must match fuellib/constants.py)
K_B="1.380649e-23"
N_A="6.02214076e23"

case "$mode" in
    eps2K)
        result=$(awk -v e="$value" -v kb="$K_B" -v na="$N_A" \
            'BEGIN { printf "%.3f", (e / na) / kb }')
        echo "Characteristic temperature: $result K"
        ;;
    *)
        usage
        ;;
esac
