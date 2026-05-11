#!/bin/bash
# Run all tutorial 14 bank conflict scenarios and display execution times

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/../../../../build/bin"

echo "========================================"
echo "Tutorial 14: Bank Conflict Scenarios"
echo "========================================"
echo ""

# Check if binaries exist
BINARIES=(
    "aa_tutorial_14_01_row_major"
    "aa_tutorial_14_02_column_major"
    "aa_tutorial_14_03_row_major_padded"
    "aa_tutorial_14_04_row_major_xor"
)

for bin in "${BINARIES[@]}"; do
    if [[ ! -f "${BIN_DIR}/${bin}" ]]; then
        echo "Error: ${bin} not found in ${BIN_DIR}"
        echo "Please build first: ninja ${bin}"
        exit 1
    fi
done

echo "All binaries found. Running tests..."
echo ""

# Run each scenario and capture timing
declare -a TIMES
declare -a NAMES

echo "Running 01_row_major..."
OUTPUT1=$("${BIN_DIR}/aa_tutorial_14_01_row_major" 2>&1)
TIME1=$(echo "$OUTPUT1" | grep -oP "Execution time: \K[0-9.]+" | head -1)
NAMES+=("01_row_major (baseline)")
TIMES+=("${TIME1:-N/A}")

echo "Running 02_column_major..."
OUTPUT2=$("${BIN_DIR}/aa_tutorial_14_02_column_major" 2>&1)
TIME2=$(echo "$OUTPUT2" | grep -oP "Execution time: \K[0-9.]+" | head -1)
NAMES+=("02_column_major")
TIMES+=("${TIME2:-N/A}")

echo "Running 03_row_major_padded..."
OUTPUT3=$("${BIN_DIR}/aa_tutorial_14_03_row_major_padded" 2>&1)
TIME3=$(echo "$OUTPUT3" | grep -oP "Execution time: \K[0-9.]+" | head -1)
NAMES+=("03_row_major_padded")
TIMES+=("${TIME3:-N/A}")

echo "Running 04_row_major_xor..."
OUTPUT4=$("${BIN_DIR}/aa_tutorial_14_04_row_major_xor" 2>&1)
PASS_PLAIN=$(echo "$OUTPUT4" | grep "Plain LDS:" | grep -c "PASSED")
PASS_XOR=$(echo "$OUTPUT4" | grep "XOR LDS:" | grep -c "PASSED")
NAMES+=("04_row_major_xor")
TIMES+=("N/A (single-pass)")

echo ""
echo "========================================"
echo "Execution Time Summary"
echo "========================================"
echo ""
printf "%-30s %15s\n" "Scenario" "Time (ms)"
printf "%-30s %15s\n" "------------------------------" "---------------"
for i in "${!NAMES[@]}"; do
    printf "%-30s %15s\n" "${NAMES[$i]}" "${TIMES[$i]}"
done
echo ""

echo "========================================"
echo "Expected Bank Conflict Pattern"
echo "========================================"
echo ""
printf "| %-25s | %-15s | %-12s |\n" "Scenario" "Bank Conflicts" "LDS Overhead"
printf "| %-25s | %-15s | %-12s |\n" "-------------------------" "---------------" "------------"
printf "| %-25s | %-15s | %-12s |\n" "01_row_major" "HIGH (4-way)" "0%"
printf "| %-25s | %-15s | %-12s |\n" "02_column_major" "LOW" "0%"
printf "| %-25s | %-15s | %-12s |\n" "03_row_major_padded" "MEDIUM (9x less)" "6.25%"
printf "| %-25s | %-15s | %-12s |\n" "04_row_major_xor" "LOW (near 0)" "0%"
echo ""

echo "========================================"
echo "Verification Status"
echo "========================================"
echo ""
echo "04_row_major_xor:"
if [[ "$PASS_PLAIN" -eq 1 && "$PASS_XOR" -eq 1 ]]; then
    echo "  Plain LDS: PASSED"
    echo "  XOR LDS:   PASSED"
else
    echo "  Some tests FAILED - check output above"
fi
echo ""

echo "========================================"
echo "To Profile Bank Conflicts"
echo "========================================"
echo ""
echo "Run each with rocprofv3 to measure actual bank conflicts:"
echo ""
echo "rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -d /tmp/t14_01 -- ${BIN_DIR}/aa_tutorial_14_01_row_major"
echo "rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -d /tmp/t14_02 -- ${BIN_DIR}/aa_tutorial_14_02_column_major"
echo "rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -d /tmp/t14_03 -- ${BIN_DIR}/aa_tutorial_14_03_row_major_padded"
echo "rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -d /tmp/t14_04 -- ${BIN_DIR}/aa_tutorial_14_04_row_major_xor"
echo ""
