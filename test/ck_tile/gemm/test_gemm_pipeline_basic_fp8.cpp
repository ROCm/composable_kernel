// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "test_gemm_pipeline_basic_run_test.inc"

int main()
{
    bool is_success = true;
    is_success =
        run_gemm_combinations<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>() && is_success;
    is_success =
        run_gemm_combinations<ck_tile::fp8_t, ck_tile::pk_int4_t, ck_tile::half_t>() && is_success;
    return is_success ? EXIT_SUCCESS : EXIT_FAILURE;
}
