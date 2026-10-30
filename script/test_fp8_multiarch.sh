#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Build these examples with GPU_TARGETS="gfx942;gfx950", then run the same
# binaries on each GPU (select one with HIP_VISIBLE_DEVICES when necessary).
# Usage: bash script/test_fp8_multiarch.sh /path/to/build/bin
set -euo pipefail

binary_dir=${1:?Provide the CK build/bin directory}
test_log=$(mktemp)
trap 'rm -f "$test_log"' EXIT

run() {
    local example=$1
    shift
    printf 'Checking %s\n' "$example"
    "$binary_dir/$example" "$@" 2>&1 | tee "$test_log"
    # Some example runners report validation failures without a nonzero exit.
    if grep -Eq 'wrong values|Error: Incorrect' "$test_log"; then
        return 1
    fi
}

run example_gemm_xdl_fp8 1 2 0 256 128 128 -1 -1 -1
run example_gemm_xdl_fp8_bf8 1 2 0 256 128 128 -1 -1 -1
run example_grouped_conv_bwd_weight_xdl_fp16_comp_bf8_fp8 1 2 0 2 1 2 256 256 3 3 8 8 1 1 1 1 1 1 1 1
run example_batched_gemm_xdl_fp8_rowwise_v3 1 2 0 256 128 256 2
run elementwise_scale_permute_amax_2D_fp16_fp8 1 0 128 128
run example_convnd_fwd_xdl_convinvscale_fp8 1 2 0 2 1 1 128 128 3 3 8 8 1 1 1 1 1 1 1 1
run example_convnd_fwd_xdl_convscale_amax_fp8 1 2 0 2 1 1 128 128 3 3 8 8 1 1 1 1 1 1 1 1
run example_convnd_fwd_xdl_convscale_relu_amax_fp8 1 2 0 2 1 1 128 128 3 3 8 8 1 1 1 1 1 1 1 1
run example_gemm_multiply_multiply_xdl_fp8_blockscale_bpreshuffle 1 2 0 128 128 128 0
run example_moe_gemm1_xdl_fp8 1 2 0 128 128 64 8 8
run example_moe_gemm1_xdl_fp8_blockscale_splitk 1 2 0 128 128 64 8 8
run example_moe_gemm2_xdl_fp8_blockscale 1 2 0 128 128 64
run example_moe_gemm1_xdl_pk_i4 1 2 0 128 128 64
run example_moe_gemm2_xdl_pk_i4 1 2 0 128 128 64
run example_moe_gemm1_xdl_fp8_blockscale 1 2 0 128 128 64 8 8
run example_moe_gemm2_xdl_fp8 1 2 0 256 128 64 8 8
run tile_example_gemm_quant -m=256 -n=256 -k=256 -prec=fp8 -v=1 -warmup=0 -repeat=1
run tile_example_flatmm_basic -m=128 -n=128 -k=128 -v=1 -prec=fp8 -warmup=0 -repeat=1
run tile_example_grouped_flatmm -Ms=128,128 -Ns=128,128 -Ks=128,128 -group_count=2 -prec=fp8 -v=2 -warmup=0 -repeat=1

printf 'All 19 FP8 examples passed.\n'
