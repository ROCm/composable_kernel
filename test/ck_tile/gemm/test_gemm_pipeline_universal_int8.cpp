// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_smoke_util.hpp"
#include "test_gemm_pipeline_smoke_run_test.inc"
#include "test_gemm_pipeline_universal_run_test.inc"

int main()
{
    bool is_success = true;
    is_success =
        run_gemm_combinations<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t>() && is_success;
    return is_success ? EXIT_SUCCESS : EXIT_FAILURE;
}
