#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Dispatcher FMHA smoke test - mirrors the 01_fmha smoke_test_fwd.sh matrix.
# Run from the dispatcher build directory.

set -euo pipefail

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

GPU_ARCH=${GPU_ARCH:-gfx950}
if [ -z "${GPU_ARCH}" ]; then
    GPU_ARCH=$(rocminfo 2>/dev/null | grep -E 'Name:\s+gfx' | head -n1 | awk '{print $2}' || echo "gfx950")
fi

BUILD_DIR=${BUILD_DIR:-"${SCRIPT_DIR}/../build"}
PASS=0
FAIL=0
TOTAL=0

run_example() {
    local name=$1
    shift
    local exe="${BUILD_DIR}/examples/${name}"

    if [ ! -x "$exe" ]; then
        echo "[SKIP] $name (not built)"
        return
    fi

    TOTAL=$((TOTAL + 1))
    if "$exe" --arch "$GPU_ARCH" "$@" >/dev/null 2>&1; then
        echo "[PASS] $name $*"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name $*"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== FMHA Dispatcher Smoke Test ==="
echo "GPU_ARCH=$GPU_ARCH"
echo "BUILD_DIR=$BUILD_DIR"
echo ""

echo "--- Basic FMHA ---"
run_example fmha_01_basic
run_example fmha_02_splitkv
run_example fmha_03_kvcache
run_example fmha_04_bwd
run_example fmha_05_appendkv
run_example fmha_06_batch_prefill

echo ""
echo "--- Profile FMHA ---"
run_example fmha_07_profile_pytorch
run_example fmha_08_profile_flash
run_example fmha_09_profile_aiter
run_example fmha_10_profile_fp32_fp8
run_example fmha_11_receipt_aliases
run_example fmha_12_registry_json

echo ""
echo "--- Feature Coverage ---"
run_example fmha_13_feature_coverage

echo ""
echo "--- GPU Execution (new) ---"
run_example fmha_14_benchmark_validation --verify
run_example fmha_15_multi_shape
run_example fmha_16_heuristics
run_example fmha_17_autofill_autocorrect
run_example fmha_18_gpu_splitkv
run_example fmha_19_gpu_masks
run_example fmha_20_gpu_bias
run_example fmha_21_gpu_features
run_example fmha_22_gpu_bwd
run_example fmha_23_multi_registry
run_example fmha_24_per_receipt_registries
run_example fmha_25_gpu_appendkv_prefill
run_example fmha_26_dtypes_hdims
run_example fmha_27_padding_permutation

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $TOTAL total ==="

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
