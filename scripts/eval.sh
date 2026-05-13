#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Ground-truth: palette index → expected color name
declare -a EXPECTED=(
    "Red"     # 0
    "Orange"  # 1
    "Yellow"  # 2
    "Green"   # 3
    "Cyan"    # 4
    "Blue"    # 5
    "Purple"  # 6
    "Pink"    # 7
    "White"   # 8
    "Black"   # 9
    "Gray"    # 10
    "Brown"   # 11
)

PASS=0
FAIL=0
TOTAL=${#EXPECTED[@]}

printf "%-6s %-35s %-10s %-10s %s\n" "Status" "File" "Expected" "Detected" "Confidence"
printf '%0.s─' {1..75}; echo

for i in $(seq 0 $((TOTAL - 1))); do
    FILE="data/color_palette_${i}.png"
    WANT="${EXPECTED[$i]}"

    OUTPUT=$(uv run color_detector_lab.py image "$FILE" 2>/dev/null)
    GOT=$(echo "$OUTPUT"      | grep "Detected Color" | sed 's/.*: //' | tr -d '[:space:]')
    CONF=$(echo "$OUTPUT"     | grep "Confidence"     | sed 's/.*: //' | tr -d '[:space:]')

    if [ "$GOT" = "$WANT" ]; then
        STATUS="PASS"
        ((PASS++)) || true
    else
        STATUS="FAIL"
        ((FAIL++)) || true
    fi

    printf "%-6s %-35s %-10s %-10s %s\n" "$STATUS" "$FILE" "$WANT" "$GOT" "$CONF"
done

printf '%0.s─' {1..75}; echo
printf "Result: %d/%d passed\n" "$PASS" "$TOTAL"
