#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


# Test script for tile engine benchmarks (GEMM and Pooling)
# This script demonstrates how to run the individual benchmark executables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default operation type (gemm, pool, or all)
OP_TYPE="all"

# Parse command line arguments
show_help() {
    echo "Usage: $0 [OPTIONS] [build_directory]"
    echo ""
    echo "Options:"
    echo "  --gemm        Test only GEMM benchmarks"
    echo "  --pool        Test only Pooling benchmarks"
    echo "  --all         Test all benchmarks (default)"
    echo "  --verify      Enable verification"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Test all benchmarks, auto-detect build dir"
    echo "  $0 --pool                    # Test only pooling benchmarks"
    echo "  $0 --gemm /path/to/build     # Test GEMM with specific build dir"
    echo "  $0 --verify --pool           # Test pooling with verification"
}

VERIFY_FLAG=""
BUILD_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gemm)
            OP_TYPE="gemm"
            shift
            ;;
        --pool)
            OP_TYPE="pool"
            shift
            ;;
        --all)
            OP_TYPE="all"
            shift
            ;;
        --verify)
            VERIFY_FLAG="-verify=1"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            BUILD_DIR="$1"
            shift
            ;;
    esac
done

# Find the build directory
if [ -z "$BUILD_DIR" ]; then
    # Try common build directory locations
    for dir in "/root/workspace/composable_kernel/build" "$HOME/composable_kernel/build" "$(pwd)/build"; do
        if [ -d "$dir/bin" ]; then
            BUILD_DIR="$dir"
            break
        fi
    done
    
    if [ -z "$BUILD_DIR" ]; then
        echo -e "${RED}Error: Could not find build directory. Please provide it as argument.${NC}"
        echo "Usage: $0 [--gemm|--pool|--all] <build_directory>"
        exit 1
    fi
fi

echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"
echo -e "${GREEN}Operation type: $OP_TYPE${NC}"

# Check if bin directory exists
if [ ! -d "$BUILD_DIR/bin" ]; then
    echo -e "${RED}Error: bin directory not found in $BUILD_DIR${NC}"
    exit 1
fi

# Results file
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
echo "Results will be saved to: $RESULTS_FILE"

# ============================================================================
# GEMM Benchmark Functions
# ============================================================================

run_gemm_benchmarks() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}       GEMM BENCHMARKS${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Find all GEMM benchmark executables
    GEMM_BENCHMARKS=$(find "$BUILD_DIR/bin" -name "benchmark_gemm_*" -type f -executable 2>/dev/null | sort)
    
    if [ -z "$GEMM_BENCHMARKS" ]; then
        echo -e "${YELLOW}No GEMM benchmark executables found in $BUILD_DIR/bin${NC}"
        echo "Build with: make benchmark_gemm_<kernel_name>"
        return 0
    fi
    
    NUM_GEMM=$(echo "$GEMM_BENCHMARKS" | wc -l)
    echo -e "${GREEN}Found $NUM_GEMM GEMM benchmark executable(s)${NC}"
    
    # Test sizes for GEMM
    GEMM_SIZES=(512 1024 2048)
    
    COUNTER=0
    GEMM_PASSED=0
    GEMM_FAILED=0
    
    for BENCH in $GEMM_BENCHMARKS; do
        COUNTER=$((COUNTER + 1))
        BENCH_NAME=$(basename "$BENCH")
        echo -e "\n${GREEN}[GEMM $COUNTER/$NUM_GEMM] Running: $BENCH_NAME${NC}"
        
        for SIZE in "${GEMM_SIZES[@]}"; do
            echo -e "  Testing size: ${SIZE}x${SIZE}x${SIZE}"
            
            # Run benchmark
            if "$BENCH" -m=$SIZE -n=$SIZE -k=$SIZE $VERIFY_FLAG -warmup=5 -repeat=10 2>&1 | \
               grep -E "(Time:|Performance:|Verification:|Error|TFLOPS|latency)" | head -5; then
                GEMM_PASSED=$((GEMM_PASSED + 1))
            else
                echo -e "  ${RED}Benchmark failed or no output!${NC}"
                GEMM_FAILED=$((GEMM_FAILED + 1))
            fi
        done
    done
    
    echo -e "\n${GREEN}GEMM Summary: $GEMM_PASSED passed, $GEMM_FAILED failed${NC}"
}

# ============================================================================
# Pooling Benchmark Functions
# ============================================================================

run_pool_benchmarks() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}       POOLING BENCHMARKS${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Find all Pooling benchmark executables
    POOL_BENCHMARKS=$(find "$BUILD_DIR/bin" -name "benchmark_pool*" -type f -executable 2>/dev/null | sort)
    
    if [ -z "$POOL_BENCHMARKS" ]; then
        echo -e "${YELLOW}No Pooling benchmark executables found in $BUILD_DIR/bin${NC}"
        echo "Build with: make benchmark_pool_all or make benchmark_pool2d or make benchmark_pool3d"
        return 0
    fi
    
    NUM_POOL=$(echo "$POOL_BENCHMARKS" | wc -l)
    echo -e "${GREEN}Found $NUM_POOL Pooling benchmark executable(s)${NC}"
    
    COUNTER=0
    POOL_PASSED=0
    POOL_FAILED=0
    
    # Test configurations for pooling
    # Format: "description|args"
    POOL2D_TESTS=(
        "Small 2D (64x64)|      -N=1 -H=64  -W=64  -C=32  -Y=3 -X=3 -Sy=2 -Sx=2"
        "Medium 2D (224x224)|   -N=2 -H=224 -W=224 -C=64  -Y=3 -X=3 -Sy=2 -Sx=2"
        "Large 2D (512x512)|    -N=1 -H=512 -W=512 -C=128 -Y=3 -X=3 -Sy=2 -Sx=2"
    )
    
    POOL3D_TESTS=(
        "Small 3D (32x32x32)|   -N=1 -D=32 -H=32 -W=32 -C=32  -Z=3 -Y=3 -X=3 -Sz=2 -Sy=2 -Sx=2"
        "Medium 3D (56x56x32)|  -N=2 -D=32 -H=56 -W=56 -C=64  -Z=3 -Y=3 -X=3 -Sz=2 -Sy=2 -Sx=2"
        "Large 3D (64x64x64)|   -N=1 -D=64 -H=64 -W=64 -C=128 -Z=3 -Y=3 -X=3 -Sz=2 -Sy=2 -Sx=2"
    )
    
    for BENCH in $POOL_BENCHMARKS; do
        COUNTER=$((COUNTER + 1))
        BENCH_NAME=$(basename "$BENCH")
        echo -e "\n${GREEN}[Pool $COUNTER/$NUM_POOL] Running: $BENCH_NAME${NC}"
        
        # Determine if 2D or 3D based on name
        if [[ "$BENCH_NAME" == *"pool2d"* ]]; then
            TESTS=("${POOL2D_TESTS[@]}")
            POOL_TYPE="2D"
        elif [[ "$BENCH_NAME" == *"pool3d"* ]]; then
            TESTS=("${POOL3D_TESTS[@]}")
            POOL_TYPE="3D"
        else
            echo -e "  ${YELLOW}Unknown pool type, skipping...${NC}"
            continue
        fi
        
        for TEST in "${TESTS[@]}"; do
            # Parse test description and args
            DESC=$(echo "$TEST" | cut -d'|' -f1 | xargs)
            ARGS=$(echo "$TEST" | cut -d'|' -f2 | xargs)
            
            echo -e "  Testing: ${DESC}"
            
            # Run benchmark
            OUTPUT=$("$BENCH" $ARGS $VERIFY_FLAG -warmup=5 -repeat=10 2>&1)
            EXIT_CODE=$?
            
            if [ $EXIT_CODE -eq 0 ]; then
                # Try to extract and display key metrics
                echo "$OUTPUT" | grep -iE "(latency|bandwidth|tflops|time|performance|pass)" | head -3
                if [ -z "$(echo "$OUTPUT" | grep -i "error\|fail\|wrong")" ]; then
                    echo -e "    ${GREEN}PASS${NC}"
                    POOL_PASSED=$((POOL_PASSED + 1))
                else
                    echo -e "    ${RED}FAIL (verification error)${NC}"
                    POOL_FAILED=$((POOL_FAILED + 1))
                fi
            else
                echo -e "    ${RED}FAIL (exit code: $EXIT_CODE)${NC}"
                # Show error output
                echo "$OUTPUT" | grep -iE "(error|fail|wrong|unsupported)" | head -3
                POOL_FAILED=$((POOL_FAILED + 1))
            fi
        done
    done
    
    echo -e "\n${GREEN}Pooling Summary: $POOL_PASSED passed, $POOL_FAILED failed${NC}"
}

# ============================================================================
# Main Execution
# ============================================================================

echo -e "${YELLOW}Starting ckTileEngine benchmark tests...${NC}"
echo "=============================================="

TOTAL_START=$(date +%s)

case $OP_TYPE in
    gemm)
        run_gemm_benchmarks
        ;;
    pool)
        run_pool_benchmarks
        ;;
    all)
        run_gemm_benchmarks
        run_pool_benchmarks
        ;;
esac

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}       FINAL SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total execution time: ${TOTAL_TIME} seconds"
echo -e "Operation type tested: $OP_TYPE"

# ============================================================================
# Example Commands
# ============================================================================

echo -e "\n${YELLOW}Example commands for manual testing:${NC}"

if [[ "$OP_TYPE" == "gemm" ]] || [[ "$OP_TYPE" == "all" ]]; then
    echo -e "\n${GREEN}GEMM Examples:${NC}"
    SAMPLE_GEMM=$(find "$BUILD_DIR/bin" -name "benchmark_gemm_*" -type f 2>/dev/null | head -1)
    if [ -n "$SAMPLE_GEMM" ]; then
        echo "# Basic GEMM run:"
        echo "$SAMPLE_GEMM -m=1024 -n=1024 -k=1024"
        echo ""
        echo "# GEMM with CPU verification:"
        echo "$SAMPLE_GEMM -m=1024 -n=1024 -k=1024 -verify=1"
    fi
fi

if [[ "$OP_TYPE" == "pool" ]] || [[ "$OP_TYPE" == "all" ]]; then
    echo -e "\n${GREEN}Pooling Examples:${NC}"
    SAMPLE_POOL2D=$(find "$BUILD_DIR/bin" -name "benchmark_pool2d_*" -type f 2>/dev/null | head -1)
    SAMPLE_POOL3D=$(find "$BUILD_DIR/bin" -name "benchmark_pool3d_*" -type f 2>/dev/null | head -1)
    
    if [ -n "$SAMPLE_POOL2D" ]; then
        echo "# 2D Pooling (NHWC format):"
        echo "$SAMPLE_POOL2D -N=2 -H=224 -W=224 -C=64 -Y=3 -X=3 -Sy=2 -Sx=2"
        echo ""
        echo "# 2D Pooling with verification:"
        echo "$SAMPLE_POOL2D -N=2 -H=224 -W=224 -C=64 -Y=3 -X=3 -Sy=2 -Sx=2 -verify=1"
    fi
    
    if [ -n "$SAMPLE_POOL3D" ]; then
        echo ""
        echo "# 3D Pooling (NDHWC format):"
        echo "$SAMPLE_POOL3D -N=2 -D=32 -H=56 -W=56 -C=64 -Z=3 -Y=3 -X=3 -Sz=2 -Sy=2 -Sx=2"
        echo ""
        echo "# 3D Pooling with verification:"
        echo "$SAMPLE_POOL3D -N=2 -D=32 -H=56 -W=56 -C=64 -Z=3 -Y=3 -X=3 -Sz=2 -Sy=2 -Sx=2 -verify=1"
    fi
fi

echo -e "\n${GREEN}Testing complete!${NC}"
