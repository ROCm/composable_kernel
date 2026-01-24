#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Stress test script for Split-K BQuant GEMM
# Tests only valid configurations where KRead % 128 == 0
# Tests both fp8 and bf8 precisions

BINARY="./build/bin/tile_example_gemm_quant"
QUANT_MODE="bquant"
INIT=0
REPEAT=1
WARMUP=0

# Arrays to track results
declare -a PASSED_TESTS=()
declare -a FAILED_TESTS=()
declare -a SKIPPED_TESTS=()

echo "=============================================="
echo "Split-K BQuant GEMM Stress Test"
echo "=============================================="
echo "Binary: $BINARY"
echo "Quant Mode: $QUANT_MODE"
echo "Init: $INIT (random)"
echo "Testing precisions: fp8, bf8"
echo "=============================================="
echo ""

# M values to test
M_VALUES=(16 32 64 128 256)

# N values to test  
N_VALUES=(64 128 256)

# Valid (K, split_k) combinations where KRead % 128 == 0
# Format: "K:split_k1,split_k2,..."
declare -A VALID_K_SPLITS
VALID_K_SPLITS[256]="1,2"
VALID_K_SPLITS[384]="1,3"
VALID_K_SPLITS[512]="1,2,4,5"
VALID_K_SPLITS[640]="1,5,6"
VALID_K_SPLITS[768]="1,2,3,6,7"
VALID_K_SPLITS[896]="1,7,8"
VALID_K_SPLITS[1024]="1,2,4,8"
VALID_K_SPLITS[1152]="1,3,5"
VALID_K_SPLITS[1280]="1,2,5"
VALID_K_SPLITS[1536]="1,2,3,4,6"
VALID_K_SPLITS[1792]="1,2,5,7"
VALID_K_SPLITS[2048]="1,2,4,8"
VALID_K_SPLITS[2560]="1,2,4,5,7"
VALID_K_SPLITS[3072]="1,2,3,4,5,6,8"
VALID_K_SPLITS[4096]="1,2,4,8"

# Precisions to test
PREC_VALUES=("fp8" "bf8")

TOTAL_TESTS=0
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for PREC in "${PREC_VALUES[@]}"; do
  echo ""
  echo "############################################"
  echo "Testing precision: $PREC"
  echo "############################################"
  echo ""
  
  for M in "${M_VALUES[@]}"; do
    for N in "${N_VALUES[@]}"; do
      for K in "${!VALID_K_SPLITS[@]}"; do
        # Parse valid split_k values for this K
        IFS=',' read -ra SPLIT_K_ARRAY <<< "${VALID_K_SPLITS[$K]}"
        
        for SPLIT_K in "${SPLIT_K_ARRAY[@]}"; do
          ((TOTAL_TESTS++))
          
          echo "----------------------------------------------"
          echo "Test #$TOTAL_TESTS: prec=$PREC M=$M N=$N K=$K split_k=$SPLIT_K"
          echo "----------------------------------------------"
          
          OUTPUT=$($BINARY -quant_mode=$QUANT_MODE -repeat=$REPEAT -warmup=$WARMUP \
                           -prec=$PREC -split_k=$SPLIT_K -m=$M -n=$N -init=$INIT -k=$K 2>&1)
          
          # Print kernel output (grid size and verification result)
          echo "$OUTPUT" | grep -E "(grid:|verification)" | head -2
          
          # Check result
          if echo "$OUTPUT" | grep -q "verification result is:correct"; then
            echo "Result: PASS"
            ((PASS_COUNT++))
            PASSED_TESTS+=("prec=$PREC M=$M N=$N K=$K split_k=$SPLIT_K")
          elif echo "$OUTPUT" | grep -q "verification result is:fail"; then
            echo "Result: FAIL (numerical error)"
            ((FAIL_COUNT++))
            FAILED_TESTS+=("prec=$PREC M=$M N=$N K=$K split_k=$SPLIT_K")
            # Show error details
            echo "$OUTPUT" | grep -E "max err:|wrong values" | head -2
          elif echo "$OUTPUT" | grep -q "not supported\|Skipping\|Arguments not supported"; then
            echo "Result: SKIPPED (configuration not supported)"
            ((SKIP_COUNT++))
            SKIPPED_TESTS+=("prec=$PREC M=$M N=$N K=$K split_k=$SPLIT_K")
            ((TOTAL_TESTS--))
          else
            echo "Result: FAIL (unknown error)"
            ((FAIL_COUNT++))
            FAILED_TESTS+=("prec=$PREC M=$M N=$N K=$K split_k=$SPLIT_K")
            echo "$OUTPUT" | tail -5
          fi
          echo ""
        done
      done
    done
  done
done

echo ""
echo "=============================================="
echo "                  SUMMARY"
echo "=============================================="
echo ""
echo "Total Tests Run: $TOTAL_TESTS"
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Skipped: $SKIP_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
  echo "✓ ALL TESTS PASSED!"
else
  echo "✗ SOME TESTS FAILED!"
  echo ""
  echo "Failed test cases:"
  for test in "${FAILED_TESTS[@]}"; do
    echo "  - $test"
  done
fi

if [ $SKIP_COUNT -gt 0 ]; then
  echo ""
  echo "Skipped test cases (not supported):"
  for test in "${SKIPPED_TESTS[@]}"; do
    echo "  - $test"
  done
fi

echo ""
echo "=============================================="
echo "Test completed at $(date)"
echo "=============================================="

# Exit with error code if any tests failed
if [ $FAIL_COUNT -gt 0 ]; then
  exit 1
fi
exit 0
