// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "test_warp_gemm_attr_compose.hpp"

using WGDispatcherTypesList = ::testing::Types<
    // clang-format off

    // int8
    WGDispCase<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 32, 32, 16, false>,
    WGDispCase<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 32, 32, 16, true>,
    WGDispCase<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 16, 16, 32, false>,
    WGDispCase<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 16, 16, 32, true>>;
// clang-format on

TYPED_TEST_SUITE(WGCompileTimeTest, WGDispatcherTypesList);
TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGCompileTimeTest, Instantiate) { this->RunTest(); }

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG) { this->RunTest(); }
