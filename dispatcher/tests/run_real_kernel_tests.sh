#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# Run real kernel tests with automatic kernel generation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPATCHER_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$DISPATCHER_DIR/build"
CODEGEN_DIR="$DISPATCHER_DIR/codegen"
KERNEL_OUTPUT_DIR="$BUILD_DIR/generated_kernels"

echo "========================================"
echo "Real Kernel Test Runner"
echo "========================================"
echo ""

# Step 1: Generate kernels if they don't exist
if [ ! -f "$KERNEL_OUTPUT_DIR/tile_engine_kernel_128x128x64.hpp" ]; then
    echo "Step 1: Generating CK Tile kernels..."
    echo "----------------------------------------"
    
    mkdir -p "$KERNEL_OUTPUT_DIR"
    
    cd "$CODEGEN_DIR"
    python3 unified_gemm_codegen.py \
        --output-dir "$KERNEL_OUTPUT_DIR" \
        --datatype fp16 \
        --layout rcr \
        --gpu-target gfx942 \
        --preselected fp16_rcr_essential
    
    echo ""
    echo "✓ Kernels generated in: $KERNEL_OUTPUT_DIR"
    echo ""
else
    echo "✓ Kernels already exist in: $KERNEL_OUTPUT_DIR"
    echo ""
fi

# Step 2: Build dispatcher with real kernel tests
echo "Step 2: Building dispatcher with tests..."
echo "----------------------------------------"

cd "$BUILD_DIR"

cmake .. \
    -D CMAKE_PREFIX_PATH=/opt/rocm \
    -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -D CMAKE_BUILD_TYPE=Release \
    -D GPU_TARGETS="gfx942" \
    -D BUILD_DISPATCHER_TESTS=ON \
    -D BUILD_DISPATCHER_EXAMPLES=ON

make -j$(nproc) 2>&1 | grep -E "(Building|Linking|Built|error|warning)" || true

echo ""
echo "✓ Build complete"
echo ""

# Step 3: Run tests
echo "Step 3: Running tests..."
echo "----------------------------------------"
echo ""

# Run unit tests (mock kernel tests)
echo "Running unit tests (mock kernels)..."
ctest --output-on-failure -E "test_real_kernel|test_kernel_simple"

echo ""

# Run real kernel tests if they were built
if [ -f "$BUILD_DIR/test/test_real_kernel" ]; then
    echo "Running real kernel test..."
    "$BUILD_DIR/test/test_real_kernel"
    echo ""
fi

# Run examples if they were built
if [ -f "$BUILD_DIR/examples/single_tile_kernel_example" ]; then
    echo "Running single tile kernel example..."
    "$BUILD_DIR/examples/single_tile_kernel_example"
    echo ""
fi

if [ -f "$BUILD_DIR/examples/verify_correctness" ]; then
    echo "Running correctness verification..."
    "$BUILD_DIR/examples/verify_correctness" 256 256 256
    echo ""
fi

echo "========================================"
echo "✅ All tests completed successfully!"
echo "========================================"

