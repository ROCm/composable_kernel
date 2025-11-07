// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "test_warp_gemm_attr_compose.hpp"

using WGDispatcherTypesList = ::testing::Types<
    // clang-format off

    // fp16 2:4 structural sparsity
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, false, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false, false, true>>;
// clang-format on

TYPED_TEST_SUITE(WGCompileTimeTest, WGDispatcherTypesList);
TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGCompileTimeTest, Instantiate) { this->RunTest(); }

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG) { this->RunTest(); }
