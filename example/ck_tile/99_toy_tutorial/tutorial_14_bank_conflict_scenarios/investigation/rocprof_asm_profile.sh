#!/bin/bash
# Instruction-level profiling with assembly view for rocprofv3

BINARY="./bin/aa_tutorial_14_04_row_major_xor"
OUTPUT_DIR="asm_profile"

echo "=== Instruction-Level Profiling with rocprofv3 ==="
echo ""

# Method 1: PC Sampling with instruction-level granularity
echo "Running PC sampling (instruction-level profiling)..."
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --pc-sampling \
  --pc-sampling-unit instructions \
  --pc-sampling-interval 1000 \
  --stats \
  --output-format pftrace \
  -o ${OUTPUT_DIR}/pc_sample \
  -- ${BINARY}

echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo "Open pftrace in: https://ui.perfetto.dev"
echo ""

# Method 2: With hardware counters for LDS conflicts
cat > lds_metrics.txt << 'EOF'
pmc: SQ_LDS_BANK_CONFLICT
pmc: SQ_INSTS_LDS
pmc: SQ_WAVES
EOF

echo "Running with LDS metrics + PC sampling..."
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --pc-sampling \
  --pc-sampling-unit instructions \
  -i lds_metrics.txt \
  --stats \
  --output-format pftrace \
  -o ${OUTPUT_DIR}/with_counters \
  -- ${BINARY}

echo ""
echo "=== For Assembly View, you also need: ==="
echo ""
echo "1. Extract kernel assembly (already done):"
echo "   Your file: /data0/aghamari/composable_kernel/04_row_major_xor-hip-amdgcn-amd-amdhsa-gfx942.s"
echo ""
echo "2. Use objdump for runtime binary:"
echo "   /opt/rocm/llvm/bin/llvm-objdump -d ${BINARY} > ${OUTPUT_DIR}/binary_disasm.s"
echo ""
echo "3. Correlate PC samples with assembly manually or use CodeXL"
echo ""

# Extract disassembly
echo "Extracting disassembly from binary..."
/opt/rocm/llvm/bin/llvm-objdump -d ${BINARY} > ${OUTPUT_DIR}/binary_disasm.txt 2>/dev/null || echo "Could not disassemble (may need debug symbols)"

echo ""
echo "Done! Check ${OUTPUT_DIR}/ for results"
