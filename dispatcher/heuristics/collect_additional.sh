#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Generate additional benchmark data for shapes NOT in the original log.
# Runs in background; outputs streaming JSON that can be parsed by data_pipeline.py.

BIN_DIR="/workspace/ck_tile/bin"
OUT_LOG="data/additional_shapes.log"
WARMUP=3
REPEAT=10

mkdir -p data

# Additional shapes: square powers-of-2 and common ML sizes not in original DeepSeek set
SHAPES=(
    "64,64,64"
    "128,128,128"
    "256,256,256"
    "512,512,512"
    "1024,1024,1024"
    "2048,2048,2048"
    "4096,4096,4096"
    "1,4096,4096"
    "8,4096,4096"
    "32,4096,4096"
    "128,4096,4096"
    "1,4096,11008"
    "32,4096,11008"
    "1,8192,8192"
    "32,8192,8192"
    "1,8192,28672"
    "32,8192,28672"
    "256,256,8192"
    "8192,8192,256"
    "1024,4096,1024"
    "4096,1024,4096"
    "2048,8192,2048"
)

echo "CK Tile Additional Shapes Benchmark" > "$OUT_LOG"
echo "GPU ID: 0" >> "$OUT_LOG"
echo "Implementation: gemm_universal" >> "$OUT_LOG"
echo "" >> "$OUT_LOG"

SHAPE_IDX=0
for SHAPE in "${SHAPES[@]}"; do
    IFS=',' read -r M N K <<< "$SHAPE"
    SHAPE_IDX=$((SHAPE_IDX + 1))

    echo "========================================" >> "$OUT_LOG"
    echo "Shape $SHAPE_IDX: M=$M N=$N K=$K dtype=fp8 layout=rcr" >> "$OUT_LOG"
    echo "========================================" >> "$OUT_LOG"

    KERNEL_COUNT=0
    for EXE in "$BIN_DIR"/benchmark_gemm_universal_fp8_rcr_*; do
        KERNEL_COUNT=$((KERNEL_COUNT + 1))
        OUTPUT=$("$EXE" -m="$M" -n="$N" -k="$K" -warmup=$WARMUP -repeat=$REPEAT -verify=0 2>/dev/null)
        # Extract just the JSON block
        echo "$OUTPUT" | sed -n '/{/,/^}/p' >> "$OUT_LOG"
    done

    echo "Found $KERNEL_COUNT kernels" >> "$OUT_LOG"
    echo "Completed shape $SHAPE_IDX: M=$M N=$N K=$K ($KERNEL_COUNT kernels)" >&2
done

echo "Done generating additional data" >&2
