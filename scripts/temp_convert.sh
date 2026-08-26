#!/usr/bin/env bash
# Convert temperatures between Celsius, Kelvin, and Fahrenheit.
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") <c2k|k2c|c2f|f2c|f2k|k2f> <temperature>"
    exit 1
}

[ $# -eq 2 ] || usage

mode="$1"
temp="$2"

case "$mode" in
    c2k)
        result=$(awk -v t="$temp" 'BEGIN { printf "%.2f", t + 273.15 }')
        echo "$temp °C = $result K"
        ;;
    k2c)
        result=$(awk -v t="$temp" 'BEGIN { printf "%.2f", t - 273.15 }')
        echo "$temp K = $result °C"
        ;;
    c2f)
        result=$(awk -v t="$temp" 'BEGIN { printf "%.2f", t * 9 / 5 + 32 }')
        echo "$temp °C = $result °F"
        ;;
    f2c)
        result=$(awk -v t="$temp" 'BEGIN { printf "%.2f", (t - 32) * 5 / 9 }')
        echo "$temp °F = $result °C"
        ;;
    f2k)
        result=$(awk -v t="$temp" 'BEGIN { printf "%.2f", (t - 32) * 5 / 9 + 273.15 }')
        echo "$temp °F = $result K"
        ;;
    k2f)
        result=$(awk -v t="$temp" 'BEGIN { printf "%.2f", (t - 273.15) * 9 / 5 + 32 }')
        echo "$temp K = $result °F"
        ;;
    *)
        usage
        ;;
esac
