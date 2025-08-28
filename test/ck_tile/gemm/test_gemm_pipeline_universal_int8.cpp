// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstddef>
#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <string>

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_smoke_util.hpp"
#include "test_gemm_pipeline_smoke_run_test.inc"
#include "test_gemm_pipeline_universal_run_test.inc"

int main() { return run_gemm_combinations<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t>(); }
