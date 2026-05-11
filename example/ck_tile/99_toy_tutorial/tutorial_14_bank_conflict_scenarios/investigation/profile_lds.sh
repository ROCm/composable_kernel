#!/bin/bash
# Profile LDS bank conflicts for all tutorial 14 scenarios using rocprofv3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/../../../../build/bin"
RESULTS_DIR="/tmp/tutorial_14_lds_profile"

echo "========================================"
echo "Tutorial 14: LDS Bank Conflict Profiling"
echo "========================================"
echo ""

# Clean up old results
rm -rf "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Binaries to profile
declare -A BINARIES
BINARIES["01_row_major"]="aa_tutorial_14_01_row_major"
BINARIES["02_column_major"]="aa_tutorial_14_02_column_major"
BINARIES["03_padded"]="aa_tutorial_14_03_row_major_padded"
BINARIES["04_xor"]="aa_tutorial_14_04_row_major_xor"
BINARIES["05_xor_plus_padding"]="aa_tutorial_14_05_xor_plus_padding"
BINARIES["04_xor_fp32"]="aa_tutorial_14_04_row_major_xor_fp32"

# Check if binaries exist
for name in "${!BINARIES[@]}"; do
    bin="${BINARIES[$name]}"
    if [[ ! -f "${BIN_DIR}/${bin}" ]]; then
        echo "Error: ${bin} not found in ${BIN_DIR}"
        echo "Please build first: ninja ${bin}"
        exit 1
    fi
done

echo "Profiling with rocprofv3..."
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Arrays to store results
declare -A CONFLICTS
declare -A LDS_INSTS

# Profile each scenario
for name in "01_row_major" "02_column_major" "03_padded" "04_xor" "05_xor_plus_padding" "04_xor_fp32"; do
    bin="${BINARIES[$name]}"
    output_db="${RESULTS_DIR}/${name}.db"

    echo "Profiling ${name}..."

    # Run rocprofv3 with PMC counters
    rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -o "${output_db}" -- "${BIN_DIR}/${bin}" > /dev/null 2>&1

    # Extract results from database
    results_db="${output_db}_results.db"
    if [[ -f "${results_db}" ]]; then
        # Query the pmc_events table for our kernel (not __amd_rocclr_copyBuffer)
        RESULT=$(sqlite3 "${results_db}" "
            SELECT counter_name, SUM(counter_value)
            FROM pmc_events
            WHERE name NOT LIKE '%copyBuffer%'
              AND name NOT LIKE '%fillBuffer%'
            GROUP BY counter_name
        " 2>/dev/null)

        # Parse results
        CONFLICT_VAL=$(echo "$RESULT" | grep "SQ_LDS_BANK_CONFLICT" | cut -d'|' -f2)
        LDS_INST_VAL=$(echo "$RESULT" | grep "SQ_INSTS_LDS" | cut -d'|' -f2)

        CONFLICTS[$name]="${CONFLICT_VAL:-0}"
        LDS_INSTS[$name]="${LDS_INST_VAL:-0}"
    else
        CONFLICTS[$name]="ERROR"
        LDS_INSTS[$name]="ERROR"
    fi
done

echo ""
echo "========================================"
echo "LDS Bank Conflict Analysis Results"
echo "========================================"
echo ""

# Print header
printf "| %-22s | %20s |\n" "Scenario" "Bank Conflicts"
printf "| %-22s | %20s |\n" "----------------------" "--------------------"

# Print results
for name in "01_row_major" "02_column_major" "03_padded" "04_xor" "05_xor_plus_padding" "04_xor_fp32"; do
    conflict="${CONFLICTS[$name]}"

    # Format large numbers with commas
    if [[ "$conflict" != "ERROR" ]]; then
        conflict_fmt=$(printf "%'d" "${conflict%.*}" 2>/dev/null || echo "$conflict")
    else
        conflict_fmt="ERROR"
    fi

    printf "| %-22s | %20s |\n" "$name" "$conflict_fmt"
done

echo ""
echo "========================================"
echo "Analysis"
echo "========================================"
echo ""

# Compare results
c01="${CONFLICTS[01_row_major]}"
c02="${CONFLICTS[02_column_major]}"
c03="${CONFLICTS[03_padded]}"
c04="${CONFLICTS[04_xor]}"

if [[ "$c01" != "ERROR" && "$c03" != "ERROR" && "$c01" != "0" ]]; then
    reduction_03=$(awk "BEGIN {printf \"%.1f\", $c01 / $c03}")
    echo "Padding reduction: ${reduction_03}x fewer conflicts than baseline"
fi

if [[ "$c01" != "ERROR" && "$c04" != "ERROR" && "$c04" != "0" ]]; then
    reduction_04=$(awk "BEGIN {printf \"%.1f\", $c01 / $c04}")
    echo "XOR reduction: ${reduction_04}x fewer conflicts than baseline"
fi

if [[ "$c01" != "ERROR" && "$c02" != "ERROR" && "$c02" != "0" ]]; then
    reduction_02=$(awk "BEGIN {printf \"%.1f\", $c01 / $c02}")
    echo "Column-major reduction: ${reduction_02}x fewer conflicts than baseline"
fi

echo ""
echo "========================================"
echo "Expected Behavior"
echo "========================================"
echo ""
echo "FP16 Tests:"
echo "  01_row_major:        HIGH conflicts (4-way transpose)"
echo "  02_column_major:     LOW conflicts (contiguous access)"
echo "  03_padded:           MEDIUM conflicts (coprime stride)"
echo "  04_xor:              LOW conflicts (XOR swizzle)"
echo "  05_xor_plus_padding: VERY LOW conflicts (XOR + padding combined!)"
echo ""
echo "FP32 Test (data type comparison):"
echo "  04_xor_fp32:         Higher conflicts than FP16 XOR"
echo "                       (demonstrates XOR is less effective for FP32)"
echo ""
echo "Note: All tests use 5 iterations for consistent comparison."
echo ""

# Cleanup temp files but keep results
echo "Results saved to: ${RESULTS_DIR}"
echo ""
