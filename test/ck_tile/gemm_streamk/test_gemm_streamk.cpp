// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_gemm_streamk_types.hpp"
#include "test_gemm_streamk_util.hpp"
#include "gtest/gtest.h"

#define TEST_SUITE_NAME TestCkTileStreamK

TYPED_TEST_SUITE(TestCkTileStreamK, KernelTypesStreamK);

#include "test_gemm_streamk_cases.inc"

#undef TEST_SUITE_NAME
