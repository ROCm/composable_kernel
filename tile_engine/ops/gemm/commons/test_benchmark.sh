#!/bin/bash

# Test script for tile engine GEMM benchmarks
# This script demonstrates how to run the new individual benchmark executables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find the build directory
if [ -z "$1" ]; then
    # Try to find build directory automatically
    BUILD_DIR=$(find /root/workspace/composable_kernel -name "test_gemm_fix" -type d 2>/dev/null | head -1)
    if [ -z "$BUILD_DIR" ]; then
        echo -e "${RED}Error: Could not find build directory. Please provide it as first argument.${NC}"
        echo "Usage: $0 <build_directory>"
        exit 1
    fi
else
    BUILD_DIR="$1"
fi

echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"

# Check if bin directory exists
if [ ! -d "$BUILD_DIR/bin" ]; then
    echo -e "${RED}Error: bin directory not found in $BUILD_DIR${NC}"
    exit 1
fi

# Find all benchmark executables
echo -e "${YELLOW}Finding benchmark executables...${NC}"
BENCHMARKS=$(find "$BUILD_DIR/bin" -name "benchmark_gemm_*" -type f 2>/dev/null)

if [ -z "$BENCHMARKS" ]; then
    echo -e "${RED}No benchmark executables found in $BUILD_DIR/bin${NC}"
    echo "Please build some benchmarks first with:"
    echo "  cd $BUILD_DIR"
    echo "  make benchmark_gemm_<kernel_name>"
    exit 1
fi

# Count benchmarks
NUM_BENCHMARKS=$(echo "$BENCHMARKS" | wc -l)
echo -e "${GREEN}Found $NUM_BENCHMARKS benchmark executable(s)${NC}"

# Test sizes
SIZES=(512 1024 2048)

# Results file
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"

echo -e "${YELLOW}Running benchmarks...${NC}"
echo "Results will be saved to: $RESULTS_FILE"

# Run each benchmark
COUNTER=0
for BENCH in $BENCHMARKS; do
    COUNTER=$((COUNTER + 1))
    BENCH_NAME=$(basename "$BENCH")
    echo -e "\n${GREEN}[$COUNTER/$NUM_BENCHMARKS] Running: $BENCH_NAME${NC}"
    
    for SIZE in "${SIZES[@]}"; do
        echo -e "  Testing size: ${SIZE}x${SIZE}x${SIZE}"
        
        # Run with verification
        "$BENCH" -m=$SIZE -n=$SIZE -k=$SIZE -verify=2 -warmup=10 -repeat=20 \
                 -csv_filename="$RESULTS_FILE" -csv_format=simple \
                 2>&1 | grep -E "(Time:|Performance:|Verification:|Error)"
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo -e "  ${RED}Benchmark failed!${NC}"
        fi
    done
done

echo -e "\n${GREEN}Benchmark testing complete!${NC}"
echo "Results saved to: $RESULTS_FILE"

# Show summary if CSV file exists
if [ -f "$RESULTS_FILE" ]; then
    echo -e "\n${YELLOW}Summary of results:${NC}"
    echo "Number of tests: $(tail -n +2 "$RESULTS_FILE" | wc -l)"
    echo "Successful tests: $(grep -c "true" "$RESULTS_FILE")"
    echo "Failed tests: $(grep -c "false" "$RESULTS_FILE")"
fi

# Example of running a specific benchmark with different options
echo -e "\n${YELLOW}Example commands for manual testing:${NC}"
echo "# Basic run:"
echo "$BUILD_DIR/bin/benchmark_gemm_fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16 -m=1024 -n=1024 -k=1024"
echo ""
echo "# With CPU verification:"
echo "$BUILD_DIR/bin/benchmark_gemm_fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16 -m=1024 -n=1024 -k=1024 -verify=1"
echo ""
echo "# JSON output for parsing:"
echo "$BUILD_DIR/bin/benchmark_gemm_fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16 -m=1024 -n=1024 -k=1024 -json_output=true"
echo ""
echo "# Performance testing with TFLOPS metric:"
echo "$BUILD_DIR/bin/benchmark_gemm_fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16 -m=4096 -n=4096 -k=4096 -warmup=100 -repeat=200 -metric=1"
