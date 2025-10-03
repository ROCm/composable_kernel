// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "gtest/gtest.h"
#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_smoke_util.hpp"
#include "test_gemm_pipeline_smoke_run_test.inc"
#include "test_gemm_pipeline_prec_types.hpp"
#include "test_gemm_pipeline_universal_run_test.inc"

using PrecTypes = ::testing::Types<std::tuple<F16, F16, F16>, std::tuple<F16, I4, F16>>;

#include "test_gemm_pipeline_universal_cases.hpp"
