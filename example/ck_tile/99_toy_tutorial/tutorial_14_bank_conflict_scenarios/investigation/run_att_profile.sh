#!/bin/bash
# AMD Thread Trace (ATT) profiling with assembly-level instruction view

BINARY="./bin/aa_tutorial_14_04_row_major_xor"
OUTPUT_DIR="att_results"

echo "=== Advanced Thread Trace (ATT) Profiling ==="
echo "This will capture EVERY instruction executed with assembly view"
echo ""

# ATT profiling with rocprofv3
echo "Running ATT profiling..."
rocprofv3 \
  --att \
  --hip-trace \
  --kernel-trace \
  --att-target-cu 0 \
  --att-simd-select 0xF \
  --att-buffer-size 512 \
  --output-format pftrace \
  -o ${OUTPUT_DIR}/att_trace \
  -- ${BINARY}

echo ""
echo "=== Results Generated ==="
echo "1. Perfetto trace: ${OUTPUT_DIR}/att_trace_results.pftrace"
echo "   Open in: https://ui.perfetto.dev"
echo ""
echo "2. ATT data files in: ${OUTPUT_DIR}/"
echo ""

# Check for ATT decoder
if [ -f "/opt/rocm-7.2.0/libexec/rocprofiler/att/att.py" ]; then
    echo "=== Decoding ATT trace to assembly view ==="

    # Find the ATT trace files
    ATT_FILES=$(find ${OUTPUT_DIR} -name "*.att" 2>/dev/null)

    if [ ! -z "$ATT_FILES" ]; then
        echo "Found ATT trace files:"
        echo "$ATT_FILES"
        echo ""
        echo "To view assembly-level trace, run:"
        echo "  /opt/rocm-7.2.0/libexec/rocprofiler/att/att.py --help"
        echo "  /opt/rocm-7.2.0/libexec/rocprofiler/att/att.py <att_file>"
    else
        echo "No .att files found - check if ATT capture succeeded"
    fi
else
    echo "ATT decoder not found at expected location"
fi

echo ""
echo "=== To view instruction-level trace with assembly: ==="
echo "1. Find .att files in ${OUTPUT_DIR}/"
echo "2. Run: /opt/rocm-7.2.0/libexec/rocprofiler/att/att.py <file.att>"
echo "3. Or convert to CSV: /opt/rocm-7.2.0/libexec/rocprofiler/att/att_to_csv.py <file.att>"
echo ""

# Also run with LDS metrics
echo "=== Running ATT with LDS metrics ==="
cat > lds_att.txt << 'EOF'
pmc: SQ_LDS_BANK_CONFLICT
pmc: SQ_INSTS_LDS
EOF

rocprofv3 \
  --att \
  --hip-trace \
  --kernel-trace \
  -i lds_att.txt \
  --att-target-cu 0 \
  --output-format pftrace \
  -o ${OUTPUT_DIR}/att_with_lds \
  -- ${BINARY}

echo ""
echo "Done! Check ${OUTPUT_DIR}/ for results"
