// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "gtest/gtest.h"
#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_prec_types.hpp"
#include "test_gemm_pipeline_basic_run_test.inc"
#include "test_gemm_pipeline_type_param_product.hpp"

// Test each combination of GEMM config and precision type tuple by forming a cartesian product
using PrecTypes      = ::testing::Types<std::tuple<F16, F16, F16>, std::tuple<F16, I4, F16>>;
using BasicTestTypes = CartesianProduct_t<GemmConfigs, PrecTypes>;

#include "test_gemm_pipeline_basic_cases.hpp"
