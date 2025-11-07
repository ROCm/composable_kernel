// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "test_warp_gemm_attr_compose.hpp"

using WGDispatcherTypesList = ::testing::Types<
    // clang-format off

    // fp16
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 8, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 8, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, false, false, ck_tile::WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true, false, false, ck_tile::WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float,16, 16, 32, false, false, false, ck_tile::WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, true, false, false, ck_tile::WGAttrNumAccessEnum::Double>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 4, 64, 16, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 64, 4, 16, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 16, false>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 16, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 8, false, true>,
    // WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 8, true, true>,
    WGDispCase<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true, true>>;
// clang-format on

TYPED_TEST_SUITE(WGCompileTimeTest, WGDispatcherTypesList);
TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGCompileTimeTest, Instantiate) { this->RunTest(); }

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG) { this->RunTest(); }
