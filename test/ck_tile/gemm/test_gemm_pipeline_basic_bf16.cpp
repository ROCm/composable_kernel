// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_basic_run_test.inc"

int main() { return run_gemm_combinations<ck_tile::bf16_t>(); }
