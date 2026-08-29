// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <type_traits>

#include "test_gemm_util.hpp"
#include "gtest/gtest.h"

// Compile-time dispatch coverage for the scaled-MFMA f8f6f4 warp GEMM. The 32x32x64
// dispatcher must accept the supported MX input types (fp8/bf8/fp6/fp4) and reject
// unsupported types (e.g. half), so an unsupported pair fails to compile rather than
// silently routing into the f8f6f4 MFMA implementation.
template <typename A,
          typename B,
          ck_tile::index_t M,
          ck_tile::index_t N,
          ck_tile::index_t K,
          ck_tile::WGAttrNumAccessEnum NA,
          typename = void>
struct IsWarpGemmDispatchable : std::false_type
{
};

template <typename A,
          typename B,
          ck_tile::index_t M,
          ck_tile::index_t N,
          ck_tile::index_t K,
          ck_tile::WGAttrNumAccessEnum NA>
struct IsWarpGemmDispatchable<
    A,
    B,
    M,
    N,
    K,
    NA,
    std::void_t<ck_tile::WarpGemmDispatcher<A, B, float, M, N, K, false, false, false, NA>>>
    : std::true_type
{
};

static_assert(IsWarpGemmDispatchable<ck_tile::pk_fp4_t,
                                     ck_tile::pk_fp4_t,
                                     32,
                                     32,
                                     64,
                                     ck_tile::WGAttrNumAccessEnum::Single>::value);
static_assert(IsWarpGemmDispatchable<ck_tile::pk_fp6x16_t,
                                     ck_tile::pk_fp4_t,
                                     32,
                                     32,
                                     64,
                                     ck_tile::WGAttrNumAccessEnum::Single>::value);
static_assert(!IsWarpGemmDispatchable<ck_tile::half_t,
                                      ck_tile::half_t,
                                      32,
                                      32,
                                      64,
                                      ck_tile::WGAttrNumAccessEnum::Single>::value);

using WGDispatcherTypesList =
    ::testing::Types<ck_tile::test::warp_gemm::WGDispCase<ck_tile::pk_fp4_t,
                                                          ck_tile::pk_fp4_t,
                                                          float,
                                                          false,
                                                          false,
                                                          false,
                                                          ck_tile::WGAttrNumAccessEnum::Single>>;

template <typename T>
class WGRuntimeTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(WGRuntimeTest, WGDispatcherTypesList);

TYPED_TEST(WGRuntimeTest, Compare_Dispatcher_MakeWG)
{
    ck_tile::test::warp_gemm::
        RunCompareDispatcherAndReference<TypeParam, 16, 16, 128, true, false>();
}

// The 32x32x64 scaled-MFMA "unscaled" overload must use a unity e8m0 default scale
// (0x7F per byte), so its result matches a plain (unscaled) reference GEMM. A zero
// default scale would multiply by 2^(0-127) and zero the result.
TYPED_TEST(WGRuntimeTest, DefaultScaleMatchesUnity_32x32x64)
{
    ck_tile::test::warp_gemm::
        RunCompareDispatcherAndReference<TypeParam, 32, 32, 64, false, false>();
}
