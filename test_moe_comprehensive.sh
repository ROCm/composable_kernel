#!/bin/bash

# Comprehensive MoE GEMM Test Script
# Goal: Test all combinations to find which ones crash (memory errors, etc.)
# Unsupported configurations are logged separately, not counted as crashes

# Fixed parameters
N=4096
K=256
NUM_EXPERTS=128
TOPK=8
VALIDATE=0  # Disable validation for speed

# Test parameters
DATA_TYPES=("fp16" "bf16" "fp8" "bf8")
GEMM_KINDS=("gemm1_gate_only" "gemm1_gate_up" "gemm2")
WARP_TILES=(0 1 2 3)
NUM_TOKENS_VALUES=(32 1024)

# Binary path
BINARY="./build/bin/tile_example_moe_flatmm"

# Output files
RESULTS_FILE="moe_test_results.txt"
CRASH_FILE="moe_test_crashes.txt"
UNSUPPORTED_FILE="moe_test_unsupported.txt"
SUCCESS_FILE="moe_test_success.txt"

# Initialize output files
echo "MoE GEMM Comprehensive Test Results" > $RESULTS_FILE
echo "Started at: $(date)" >> $RESULTS_FILE
echo "Test Parameters: N=$N, K=$K, experts=$NUM_EXPERTS, topk=$TOPK" >> $RESULTS_FILE
echo "======================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

echo "ACTUAL CRASHES (Memory Errors, HIP Errors, Aborts)" > $CRASH_FILE
echo "Started at: $(date)" >> $CRASH_FILE
echo "======================================" >> $CRASH_FILE
echo "" >> $CRASH_FILE

echo "Unsupported Configurations (Not Crashes)" > $UNSUPPORTED_FILE
echo "Started at: $(date)" >> $UNSUPPORTED_FILE
echo "======================================" >> $UNSUPPORTED_FILE
echo "" >> $UNSUPPORTED_FILE

echo "Successful Test Configurations" > $SUCCESS_FILE
echo "Started at: $(date)" >> $SUCCESS_FILE
echo "======================================" >> $SUCCESS_FILE
echo "" >> $SUCCESS_FILE

# Counter
total_tests=0
passed_tests=0
unsupported_tests=0
crashed_tests=0

# Function to run a single test
run_test() {
    local prec=$1
    local gemm_kind=$2
    local warp_tile=$3
    local num_tokens=$4
    
    total_tests=$((total_tests + 1))
    
    test_name="prec=${prec} gemm_kind=${gemm_kind} warp_tile=${warp_tile} NumTokens=${num_tokens}"
    
    echo "[$total_tests] Testing: $test_name"
    
    # Build command
    cmd="$BINARY -experts=$NUM_EXPERTS -TopK=$TOPK -N=$N -K=$K -prec=$prec -NumTokens=$num_tokens -gemm_kind=$gemm_kind -warp_tile=$warp_tile -validate=$VALIDATE -warmup=1 -repeat=1"
    
    # Run test and capture output
    output=$($cmd 2>&1)
    exit_code=$?
    
    # Determine test status
    has_unsupported=$(echo "$output" | grep -q "Can't support\|Arguments not supported" && echo "yes" || echo "no")
    has_crash=$(echo "$output" | grep -q -i "illegal memory\|HIP.*error\|abort\|terminate\|segmentation\|core dumped" && echo "yes" || echo "no")
    
    if [ "$has_crash" = "yes" ]; then
        # ACTUAL CRASH
        crashed_tests=$((crashed_tests + 1))
        result="⚠ CRASH"
        
        echo "  → CRASHED" >> $CRASH_FILE
        echo "  Test: $test_name" >> $CRASH_FILE
        echo "  Exit code: $exit_code" >> $CRASH_FILE
        echo "  Crash Details:" >> $CRASH_FILE
        
        # Extract crash details
        echo "$output" | grep -i "illegal memory\|HIP.*error\|abort\|terminate\|segmentation\|core dumped" | while IFS= read -r line; do
            echo "    $line" >> $CRASH_FILE
        done
        
        echo "" >> $CRASH_FILE
        
    elif [ "$has_unsupported" = "yes" ]; then
        # UNSUPPORTED CONFIGURATION (Not a crash)
        unsupported_tests=$((unsupported_tests + 1))
        result="○ UNSUPPORTED"
        
        echo "  → Configuration Not Supported" >> $UNSUPPORTED_FILE
        echo "  Test: $test_name" >> $UNSUPPORTED_FILE
        echo "  Reason:" >> $UNSUPPORTED_FILE
        
        echo "$output" | grep "Can't support\|Arguments not supported" | while IFS= read -r line; do
            echo "    $line" >> $UNSUPPORTED_FILE
        done
        
        echo "" >> $UNSUPPORTED_FILE
        
    elif [ $exit_code -eq 0 ]; then
        # SUCCESS
        passed_tests=$((passed_tests + 1))
        result="✓ PASS"
        
        echo "  → PASSED" >> $SUCCESS_FILE
        echo "  Test: $test_name" >> $SUCCESS_FILE
        
        # Extract performance if available
        if echo "$output" | grep -q "Perf:"; then
            perf=$(echo "$output" | grep "Perf:" | tail -1)
            echo "  $perf" >> $SUCCESS_FILE
        fi
        echo "" >> $SUCCESS_FILE
        
    else
        # UNKNOWN FAILURE
        crashed_tests=$((crashed_tests + 1))
        result="✗ FAIL"
        
        echo "  → UNKNOWN FAILURE" >> $CRASH_FILE
        echo "  Test: $test_name" >> $CRASH_FILE
        echo "  Exit code: $exit_code" >> $CRASH_FILE
        echo "  Last output lines:" >> $CRASH_FILE
        echo "$output" | tail -5 | while IFS= read -r line; do
            echo "    $line" >> $CRASH_FILE
        done
        echo "" >> $CRASH_FILE
    fi
    
    # Log to main results file
    echo "Test #$total_tests: $result" >> $RESULTS_FILE
    echo "  Configuration: $test_name" >> $RESULTS_FILE
    echo "" >> $RESULTS_FILE
    
    echo "  Result: $result"
    echo ""
}

# Main test loop
echo "Starting comprehensive MoE GEMM testing..."
echo "Total combinations: $((${#DATA_TYPES[@]} * ${#GEMM_KINDS[@]} * ${#WARP_TILES[@]} * ${#NUM_TOKENS_VALUES[@]}))"
echo ""

for prec in "${DATA_TYPES[@]}"; do
    for gemm_kind in "${GEMM_KINDS[@]}"; do
        for warp_tile in "${WARP_TILES[@]}"; do
            for num_tokens in "${NUM_TOKENS_VALUES[@]}"; do
                run_test "$prec" "$gemm_kind" "$warp_tile" "$num_tokens"
            done
        done
    done
done

# Summary
echo "======================================" >> $RESULTS_FILE
echo "Test Summary" >> $RESULTS_FILE
echo "======================================" >> $RESULTS_FILE
echo "Total tests run: $total_tests" >> $RESULTS_FILE
echo "Passed: $passed_tests" >> $RESULTS_FILE
echo "Unsupported (not crashes): $unsupported_tests" >> $RESULTS_FILE
echo "Actual crashes: $crashed_tests" >> $RESULTS_FILE
echo "Success rate: $(awk "BEGIN {printf \"%.2f\", ($passed_tests/$total_tests)*100}")%" >> $RESULTS_FILE
echo "Crash rate: $(awk "BEGIN {printf \"%.2f\", ($crashed_tests/$total_tests)*100}")%" >> $RESULTS_FILE
echo "Completed at: $(date)" >> $RESULTS_FILE

echo ""
echo "========================================"
echo "COMPREHENSIVE TEST COMPLETED"
echo "========================================"
echo "Total tests run: $total_tests"
echo "Passed: $passed_tests"
echo "Unsupported configs: $unsupported_tests"
echo "Actual crashes: $crashed_tests"
echo "Success rate: $(awk "BEGIN {printf \"%.2f\", ($passed_tests/$total_tests)*100}")%"
echo "Crash rate: $(awk "BEGIN {printf \"%.2f\", ($crashed_tests/$total_tests)*100}")%"
echo ""
echo "Results saved to:"
echo "  - Full results: $RESULTS_FILE"
echo "  - Actual crashes: $CRASH_FILE"
echo "  - Unsupported configs: $UNSUPPORTED_FILE"
echo "  - Successful runs: $SUCCESS_FILE"
