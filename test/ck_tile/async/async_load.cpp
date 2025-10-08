// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "test_gemm_pipeline_basic_run_test.inc"

int main()
{
    return run_load_tile<ck_tile::pk_fp4_t>() ? EXIT_SUCCESS : EXIT_FAILURE;
}
