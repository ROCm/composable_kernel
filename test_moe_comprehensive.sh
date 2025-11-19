#!/bin/bash

# ============================================================================
# CONFIGURABLE MoE GEMM TEST SCRIPT
# ============================================================================
# Edit the parameters below to run different test scenarios
# The script will generate a CSV file with results

# ============================================================================
# TEST CONFIGURATION - MODIFY THESE FOR DIFFERENT TESTS
# ============================================================================

# Output file name (CSV format, Excel-compatible)
OUTPUT_CSV="moe_comprehensive_all_types_test.csv"

# COMPREHENSIVE TEST: All data types, gemm kinds, and configurations
# Goal: Verify validity flag solution works universally across all supported combinations

# Test parameters - COMPLETE COVERAGE
N_K_PAIRS=("256 4096" "4096 256")
NUM_EXPERTS_VALUES=(32 64 128)           # Multiple expert counts
TOPK_VALUES=(2 4 8)                        # Multiple TopK values
DATA_TYPES=("fp8" "bf8" "bf16" "fp16")        # ALL supported types
GEMM_KINDS=("gemm1_gate_only" "gemm1_gate_up" "gemm2")  # ALL gemm kinds
WARP_TILES=(0 1 2 3)                               
NUM_TOKENS_VALUES=(32 128 256 512 1024)  # Critical values including previously crashing ones

# Execution parameters
VALIDATE=0                                    # 0=no validation, 1=validate with CPU
MAX_PARALLEL_JOBS=4                           # Reduce parallelism for stability
WARMUP=0                                      # Warmup iterations
REPEAT=1                                      # Repeat iterations

# Binary path
BINARY="./build/bin/tile_example_moe_flatmm"

# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

# Initialize CSV
CSV_FILE="$OUTPUT_CSV"

# Initialize CSV with header
echo "N,K,Num_Experts,TopK,Precision,GEMM_Kind,Warp_Tile,NumTokens,Result,Crash_Reason" > $CSV_FILE

# Counter
total_tests=0
passed_tests=0
unsupported_tests=0
crashed_tests=0

# Function to run a single test
run_test() {
    local num_experts=$1
    local topk=$2
    local prec=$3
    local gemm_kind=$4
    local warp_tile=$5
    local num_tokens=$6
    local test_id=$7
    
    test_name="N=${N} K=${K} experts=${num_experts} topk=${topk} prec=${prec} gemm_kind=${gemm_kind} warp_tile=${warp_tile} tokens=${num_tokens}"
    
    echo "[${test_id}/${TOTAL_COMBOS}] Testing: $test_name"
    
    # Build command
    cmd="$BINARY -experts=$num_experts -TopK=$topk -N=$N -K=$K -prec=$prec -NumTokens=$num_tokens -gemm_kind=$gemm_kind -warp_tile=$warp_tile -validate=$VALIDATE -warmup=$WARMUP -repeat=$REPEAT"
    
    # Run test and capture output
    output=$($cmd 2>&1)
    exit_code=$?
    
    # Determine test status using more stable logic:
    # 1. If "Perf:" appears in output → PASSED
    # 2. If "Arguments not supported" or "Can't support" → UNSUPPORTED
    # 3. Otherwise → CRASHED
    
    local result_status=""
    local crash_reason=""
    
    if echo "$output" | grep -q "Perf:"; then
        # Test ran and completed successfully
        result_status="PASSED"
        crash_reason=""
        echo "  Result: ✓ PASS"
        
    elif echo "$output" | grep -q "Can't support\|Arguments not supported"; then
        # Configuration not supported by kernel
        result_status="UNSUPPORTED"
        crash_reason=""
        echo "  Result: ○ UNSUPPORTED"
        
    else
        # Test crashed or failed
        result_status="CRASHED"
        # Try to extract crash reason
        if echo "$output" | grep -q -i "illegal memory"; then
            crash_reason="Illegal memory access"
        elif echo "$output" | grep -q -i "HIP.*error"; then
            crash_reason="HIP error"
        elif echo "$output" | grep -q -i "abort\|terminate"; then
            crash_reason="Aborted/Terminated"
        else
            crash_reason="Unknown (exit $exit_code)"
        fi
        echo "  Result: ⚠ CRASH"
    fi
    
    # Write to CSV (thread-safe append)
    echo "$N,$K,$num_experts,$topk,$prec,$gemm_kind,$warp_tile,$num_tokens,$result_status,$crash_reason" >> $CSV_FILE
    
    echo ""
}

# Main test loop
TOTAL_COMBOS=$((${#N_K_PAIRS[@]} * ${#NUM_EXPERTS_VALUES[@]} * ${#TOPK_VALUES[@]} * ${#DATA_TYPES[@]} * ${#GEMM_KINDS[@]} * ${#WARP_TILES[@]} * ${#NUM_TOKENS_VALUES[@]}))
echo "Starting MoE GEMM testing..."
echo "Total combinations: $TOTAL_COMBOS"
echo "Running with up to $MAX_PARALLEL_JOBS parallel jobs"
echo "Output: $CSV_FILE"
echo ""

test_counter=0

for n_k_pair in "${N_K_PAIRS[@]}"; do
    N=$(echo $n_k_pair | awk '{print $1}')
    K=$(echo $n_k_pair | awk '{print $2}')
    
    echo "========================================"
    echo "Testing with N=$N, K=$K"
    echo "========================================"
    
    for num_experts in "${NUM_EXPERTS_VALUES[@]}"; do
        for topk in "${TOPK_VALUES[@]}"; do
            for prec in "${DATA_TYPES[@]}"; do
                for gemm_kind in "${GEMM_KINDS[@]}"; do
                    for warp_tile in "${WARP_TILES[@]}"; do
                        for num_tokens in "${NUM_TOKENS_VALUES[@]}"; do
                            test_counter=$((test_counter + 1))
                            
                            # Wait if we've reached max parallel jobs
                            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
                                sleep 0.1
                            done
                            
                            # Run test in background
                            run_test "$num_experts" "$topk" "$prec" "$gemm_kind" "$warp_tile" "$num_tokens" "$test_counter" &
                        done
                    done
                done
            done
        done
    done
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all tests to complete..."
wait

# Count final results from CSV
total_tests=$(tail -n +2 $CSV_FILE | wc -l)
passed_tests=$(grep -c ",PASSED," $CSV_FILE)
unsupported_tests=$(grep -c ",UNSUPPORTED," $CSV_FILE)
crashed_tests=$(grep -c ",CRASHED," $CSV_FILE)

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
echo "Results saved to CSV (Excel-compatible): $CSV_FILE"
echo ""
echo "To view in Excel: Open $CSV_FILE"
echo "To view in terminal: column -t -s',' $CSV_FILE | less -S"
