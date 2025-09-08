// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_smoke_util.hpp"
#include "test_gemm_pipeline_smoke_run_test.inc"
#include "test_gemm_pipeline_universal_run_test.inc"

int main() { return run_gemm_combinations<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(); }
