// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "test_gemm_pipeline_prec_types.hpp"
#include "test_gemm_pipeline_basic_run_test.inc"

int main()
{
    bool is_success = true;
    is_success      = run_gemm_combinations<F16>() && is_success;
#if 0
    is_success =
        run_gemm_combinations<F16, I4, F16>() && is_success;
#endif
    return is_success ? EXIT_SUCCESS : EXIT_FAILURE;
}
