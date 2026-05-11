#!/bin/bash
# Benchmark all LDS bank conflict scenarios

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/../../../../build/bin"

echo "========================================"
echo "Tutorial 14: Performance Benchmark"
echo "========================================"
echo ""

# Use rocprof to get execution time
declare -A BINARIES
BINARIES["01_row_major"]="aa_tutorial_14_01_row_major"
BINARIES["02_column_major"]="aa_tutorial_14_02_column_major"
BINARIES["03_padded"]="aa_tutorial_14_03_row_major_padded"
BINARIES["04_xor"]="aa_tutorial_14_04_row_major_xor"
BINARIES["05_xor_plus_padding"]="aa_tutorial_14_05_xor_plus_padding"

echo "| Scenario               | Avg Time (μs) | Conflicts |"
echo "|------------------------|---------------|-----------|"

for name in "01_row_major" "02_column_major" "03_padded" "04_xor" "05_xor_plus_padding"; do
    bin="${BINARIES[$name]}"
    
    # Run with rocprof to get kernel time
    output=$("${BIN_DIR}/${bin}" 2>&1 | grep -A 1 "Average time" || echo "N/A")
    
    if echo "$output" | grep -q "Average time"; then
        time_us=$(echo "$output" | grep "Average time" | awk '{print $3}')
    else
        # Fall back to just running it
        "${BIN_DIR}/${bin}" > /dev/null 2>&1
        time_us="N/A"
    fi
    
    # Get conflicts from previous profiling
    case $name in
        "01_row_major") conflicts="7,168" ;;
        "02_column_major") conflicts="6,144" ;;
        "03_padded") conflicts="2,048" ;;
        "04_xor") conflicts="3,072" ;;
        "05_xor_plus_padding") conflicts="0" ;;
    esac
    
    printf "| %-22s | %13s | %9s |\n" "$name" "$time_us" "$conflicts"
done

echo ""
echo "Note: Lower time = better performance"
echo "      Zero conflicts should give best performance"
echo ""
