// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "test_warp_gemm_attr_compose.hpp"

using WGDispatcherTypesList = ::testing::Types<
    // clang-format off

    // mixed fp8/bf8
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32, 16, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32, 16, true>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 16, 16, 128, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 16, 16, 128, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32, 64, false>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32, 64, false>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32, 64, false, false, false, ck_tile::WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32, 64, false, false, false, ck_tile::WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::fp8_t, ck_tile::bf8_t, float, 16, 16, 128, false, false, false, ck_tile::WGAttrNumAccessEnum::Quad>,
    WGDispCase<ck_tile::bf8_t, ck_tile::fp8_t, float, 16, 16, 128, false, false, false, ck_tile::WGAttrNumAccessEnum::Quad>>;
// clang-format on

TYPED_TEST_SUITE(WGCompileTimeTest, WGDispatcherTypesList);
TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGCompileTimeTest, Instantiate) { this->RunTest(); }

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG) { this->RunTest(); }
