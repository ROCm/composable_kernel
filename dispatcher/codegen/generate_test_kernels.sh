#!/bin/bash
# Generate minimal set of CK Tile kernels for dispatcher testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../build/test_kernels"

echo "========================================================================"
echo "Generating Test CK Tile Kernels for Dispatcher"
echo "========================================================================"
echo ""

# Find tile_engine
TILE_ENGINE="$SCRIPT_DIR/../../tile_engine/ops/gemm"

if [ ! -f "$TILE_ENGINE/gemm_instance_builder.py" ]; then
    echo "✗ Error: tile_engine not found at $TILE_ENGINE"
    echo "  Expected: ../../tile_engine/ops/gemm/gemm_instance_builder.py"
    exit 1
fi

echo "Tile Engine: $TILE_ENGINE"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate kernels
echo "Generating FP16 RCR kernels..."
cd "$TILE_ENGINE"

python3 gemm_instance_builder.py \
    --working_path "$OUTPUT_DIR" \
    --gpu_target gfx942 \
    --datatype fp16 \
    --layout rcr \
    --config_json "$SCRIPT_DIR/minimal_test_config.json" \
    --gen_all_individual \
    --num_workers 2

echo ""
echo "✓ Kernels generated in: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR/fp16/rcr/"*.hpp 2>/dev/null || echo "  (No headers found)"
echo ""

# Count kernels
KERNEL_COUNT=$(find "$OUTPUT_DIR" -name "*.hpp" -type f | wc -l)
echo "Total kernels: $KERNEL_COUNT"
echo ""
echo "Next steps:"
echo "  1. Generate registration code:"
echo "     cd $SCRIPT_DIR"
echo "     python3 generate_kernel_registry.py --kernel-dir ../build/test_kernels/fp16/rcr"
echo ""
echo "  2. Build dispatcher with generated kernels"
echo "  3. Run integration example"

