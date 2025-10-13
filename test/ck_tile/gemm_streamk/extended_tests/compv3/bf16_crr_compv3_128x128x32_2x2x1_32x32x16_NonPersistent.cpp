// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_gemm_streamk_common_includes.hpp"

#define TEST_SUITE_PARAMS BF16_CRR_CompV3_128x128x32_2x2x1_32x32x16_NonPersistent
#define TEST_SUITE_NAME MAKE_TEST_SUITE_NAME(TEST_SUITE_PARAMS)

DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS);

#include "test_gemm_streamk_cases.inc"
