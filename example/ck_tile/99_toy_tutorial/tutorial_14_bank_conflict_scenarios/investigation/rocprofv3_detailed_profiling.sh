#!/bin/bash
# Detailed rocprofv3 profiling for LDS bank conflict analysis

# Set output directory
OUTPUT_DIR="profiling_results"
BINARY="./bin/aa_tutorial_14_04_row_major_xor"

echo "=== ROCm Profiling with rocprofv3 ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create metrics input file for PMC (Performance Monitoring Counters)
cat > pmc_metrics.txt << 'EOF'
pmc: SQ_LDS_BANK_CONFLICT
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU
pmc: SQ_INSTS_LDS
pmc: TA_TA_BUSY[0]
pmc: TCC_HIT[0]
pmc: TCC_MISS[0]
EOF

echo "Created pmc_metrics.txt with hardware counters"
echo ""

# Method 1: Perfetto trace with HIP trace + PMC counters
echo "=== Method 1: Perfetto Trace (Timeline + Counters) ==="
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --stats \
  -i pmc_metrics.txt \
  --output-format pftrace \
  -o ${OUTPUT_DIR}/timeline \
  -- ${BINARY}

echo ""
echo "Perfetto trace saved to: ${OUTPUT_DIR}/timeline_results.pftrace"
echo "Open in: https://ui.perfetto.dev"
echo ""

# Method 2: CSV format for detailed counter analysis
echo "=== Method 2: CSV Output (Detailed Counters) ==="
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --stats \
  -i pmc_metrics.txt \
  --output-format csv \
  -o ${OUTPUT_DIR}/counters \
  -- ${BINARY}

echo ""
echo "CSV results saved to: ${OUTPUT_DIR}/"
echo ""

# Method 3: JSON format for programmatic analysis
echo "=== Method 3: JSON Output (Machine Readable) ==="
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --stats \
  -i pmc_metrics.txt \
  --output-format json \
  -o ${OUTPUT_DIR}/json_output \
  -- ${BINARY}

echo ""
echo "JSON results saved to: ${OUTPUT_DIR}/"
echo ""

# Method 4: All available LDS metrics
cat > lds_all_metrics.txt << 'EOF'
pmc: SQ_LDS_BANK_CONFLICT
pmc: SQ_LDS_ADDR_CONFLICT
pmc: SQ_LDS_UNALIGNED_STALL
pmc: SQ_LDS_MEM_VIOLATIONS
pmc: SQ_WAVES
pmc: SQ_WAVE_CYCLES
pmc: SQ_INSTS_LDS
EOF

echo "=== Method 4: Comprehensive LDS Analysis ==="
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --stats \
  -i lds_all_metrics.txt \
  --output-format csv \
  -o ${OUTPUT_DIR}/lds_comprehensive \
  -- ${BINARY}

echo ""
echo "Comprehensive LDS metrics saved to: ${OUTPUT_DIR}/"
echo ""

# Extract and display key results
echo "=== Quick Results Summary ==="
if [ -f "${OUTPUT_DIR}/counters_results.csv" ]; then
    echo "LDS Bank Conflicts:"
    grep -i "SQ_LDS_BANK_CONFLICT" ${OUTPUT_DIR}/counters_results.csv | head -5
fi

echo ""
echo "=== Analysis Tools ==="
echo "1. Perfetto UI: https://ui.perfetto.dev"
echo "   - Load: ${OUTPUT_DIR}/timeline_results.pftrace"
echo "   - View timeline with GPU kernel execution and counters"
echo ""
echo "2. CSV analysis:"
echo "   - Column-based analysis of all counters"
echo "   - Import into spreadsheet or Python for detailed analysis"
echo ""
echo "3. Database query (if .db files created):"
echo "   sqlite3 ${OUTPUT_DIR}/*_results.db"
echo ""

echo "Done!"
