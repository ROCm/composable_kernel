// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_gemm_streamk_types_fp8_bf8.hpp"
#include "test_gemm_streamk_util_fp8_bf8.hpp"
#include "gtest/gtest.h"

#define TEST_SUITE_NAME TestCkTileStreamK

TYPED_TEST_SUITE(TestCkTileStreamK, KernelTypesStreamK);

#include "test_gemm_streamk_cases_fp8_bf8.inc"

#undef TEST_SUITE_NAME
