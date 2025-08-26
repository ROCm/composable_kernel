// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_smoke_util.hpp"
#include "test_gemm_pipeline_smoke_run_test.inc"
#include "test_gemm_pipeline_universal_run_test.inc"

int main() { return run_gemm_combinations<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(); }
