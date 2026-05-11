#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

# Comprehensive bank conflict profiling script for CK Tile Tutorial 11
# Profiles both plain and XOR transpose implementations and compares results

set -e

# Configuration
BUILD_DIR="${1:-relbuild}"
OUTPUT_DIR="${2:-/tmp/bank_conflict_analysis}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔════════════════════════════════════════════════╗"
echo "║  Bank Conflict Analysis Suite                 ║"
echo "║  CK Tile Tutorial 11                           ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Build directory: $BUILD_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found!"
    echo "Usage: $0 [build_dir] [output_dir]"
    echo "Example: $0 relbuild /tmp/my_analysis"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build both tutorials
echo "[1/5] Building tutorials..."
echo "----------------------------------------"
cd "$BUILD_DIR"

echo "Building plain transpose..."
if ! cmake --build . --target aa_tutorial_11_plain_transpose -j$(nproc); then
    echo "Error: Failed to build aa_tutorial_11_plain_transpose"
    exit 1
fi

echo "Building production transpose (XOR)..."
if ! cmake --build . --target aa_tutorial_11_production_transpose -j$(nproc); then
    echo "Error: Failed to build aa_tutorial_11_production_transpose"
    exit 1
fi

echo "✓ Build complete"
echo ""

# Check if binaries exist
if [ ! -f "./bin/aa_tutorial_11_plain_transpose" ]; then
    echo "Error: aa_tutorial_11_plain_transpose binary not found!"
    exit 1
fi

if [ ! -f "./bin/aa_tutorial_11_production_transpose" ]; then
    echo "Error: aa_tutorial_11_production_transpose binary not found!"
    exit 1
fi

# Profile plain transpose
echo "[2/5] Profiling plain transpose (no XOR)..."
echo "----------------------------------------"
if ! rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d "$OUTPUT_DIR/plain" \
          -- ./bin/aa_tutorial_11_plain_transpose; then
    echo "Error: Profiling plain transpose failed!"
    echo "Make sure rocprofv3 is installed and you have GPU access."
    exit 1
fi
echo "✓ Plain transpose profiled"
echo ""

# Profile XOR transpose
echo "[3/5] Profiling XOR transpose..."
echo "----------------------------------------"
if ! rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d "$OUTPUT_DIR/xor" \
          -- ./bin/aa_tutorial_11_production_transpose; then
    echo "Error: Profiling XOR transpose failed!"
    exit 1
fi
echo "✓ XOR transpose profiled"
echo ""

# Find the results.db files
echo "[4/5] Locating profile results..."
echo "----------------------------------------"
PLAIN_DB=$(find "$OUTPUT_DIR/plain" -name "results.db" -type f | head -1)
XOR_DB=$(find "$OUTPUT_DIR/xor" -name "results.db" -type f | head -1)

if [ -z "$PLAIN_DB" ]; then
    echo "Error: Could not find results.db in $OUTPUT_DIR/plain"
    exit 1
fi

if [ -z "$XOR_DB" ]; then
    echo "Error: Could not find results.db in $OUTPUT_DIR/xor"
    exit 1
fi

echo "Plain results: $PLAIN_DB"
echo "XOR results: $XOR_DB"
echo ""

# Analyze results
echo "[5/5] Analyzing results..."
echo "----------------------------------------"
if [ -f "$SCRIPT_DIR/analyze_bank_conflicts.py" ]; then
    python3 "$SCRIPT_DIR/analyze_bank_conflicts.py" \
            "$OUTPUT_DIR/plain" \
            "$OUTPUT_DIR/xor"
else
    # Fallback: manual SQLite query if Python script not available
    echo "Python analysis script not found. Using direct SQLite queries:"
    echo ""

    echo "Plain Transpose Results:"
    sqlite3 "$PLAIN_DB" "
    SELECT
        SUM(CASE WHEN counter_name = 'SQ_LDS_BANK_CONFLICT' THEN counter_value ELSE 0 END) as conflicts,
        SUM(CASE WHEN counter_name = 'SQ_INSTS_LDS' THEN counter_value ELSE 0 END) as lds_insts,
        ROUND(100.0 * conflicts / lds_insts, 2) as conflict_rate_percent
    FROM pmc_events;"

    echo ""
    echo "XOR Transpose Results:"
    sqlite3 "$XOR_DB" "
    SELECT
        SUM(CASE WHEN counter_name = 'SQ_LDS_BANK_CONFLICT' THEN counter_value ELSE 0 END) as conflicts,
        SUM(CASE WHEN counter_name = 'SQ_INSTS_LDS' THEN counter_value ELSE 0 END) as lds_insts,
        ROUND(100.0 * conflicts / lds_insts, 2) as conflict_rate_percent
    FROM pmc_events;"
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Analysis Complete!                           ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view detailed results:"
echo "  sqlite3 $PLAIN_DB"
echo "  sqlite3 $XOR_DB"
echo ""
