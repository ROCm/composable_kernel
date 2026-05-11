#!/bin/bash
# Convenience script for ATT profiling with trace decoder

TRACE_DECODER_LIB=~/rocm-tools/lib

# Check if library exists
if [ ! -f "$TRACE_DECODER_LIB/librocprof-trace-decoder.so" ]; then
    echo "Error: Trace decoder not found at $TRACE_DECODER_LIB"
    echo "Expected: $TRACE_DECODER_LIB/librocprof-trace-decoder.so"
    exit 1
fi

# Default output name
OUTPUT_NAME="${1:-att_profile}"
shift || true

# Run rocprofv3 with ATT and custom library path
rocprofv3 \
  --att \
  --att-library-path "$TRACE_DECODER_LIB" \
  --hip-trace \
  --kernel-trace \
  --att-target-cu 0 \
  --att-simd-select 0xF \
  --output-format pftrace \
  -o "$OUTPUT_NAME" \
  -- "$@"

echo ""
echo "=== ATT Profiling Complete ==="
echo "Results saved to: ${OUTPUT_NAME}_*"
echo ""
echo "Generated files:"
ls -lh ${OUTPUT_NAME}*.att ${OUTPUT_NAME}_results.pftrace 2>/dev/null
echo ""
echo "To view:"
echo "  1. Perfetto timeline: https://ui.perfetto.dev"
echo "     Load: ${OUTPUT_NAME}_results.pftrace"
echo ""
echo "  2. Decode ATT trace:"
echo "     /opt/rocm-7.2.0/libexec/rocprofiler/att/att.py ${OUTPUT_NAME}*.att"
echo ""
