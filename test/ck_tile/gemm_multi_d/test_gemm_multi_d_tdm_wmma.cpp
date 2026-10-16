// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Multi-D GEMM with the TDM pipelines (CompTDMV1 / CompTDMV2) and TdmMultiDEpilogue (gfx1250).

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_multi_d_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using NoTransC = std::false_type;
using TransC   = std::true_type;

using Tile64  = ck_tile::sequence<64, 64, 32>;
using Tile128 = ck_tile::sequence<128, 128, 64>;

// clang-format off
using KernelTypes = ::testing::Types<
    //          ALayout, BLayout, D0Layout, D1Layout, ELayout, ADataType, BDataType, D0DataType, D1DataType, AccDataType, EDataType, CDElementWiseFn,   Pipeline,    TransposeC, Tile
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV1, TransC,     Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV2, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV2, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV2, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       BF16,       BF16,       F32,        F16,  ElementWiseAddAdd, MultiDTdmV2, TransC,     Tile128>,

    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV1, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV1, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV1, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV1, TransC,     Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV2, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV2, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV2, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F16,        F16,        F32,        F16,  MultiplyMultiply,  MultiDTdmV2, TransC,     Tile128>,

    // fp32 E also instantiates the (unreachable) atomic-add epilogue branch of the kernel.
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV1, TransC,     Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV2, NoTransC,   Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV2, NoTransC,   Tile128>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV2, TransC,     Tile64>,
    std::tuple<    Row,     Col,     Row,      Row,      Row,      F16,       F16,       F32,        F32,        F32,        F32,  MultiplyMultiply,  MultiDTdmV2, TransC,     Tile128>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmMultiD, KernelTypes);

// Aligned shapes.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_256x512x256)
{
    EXPECT_TRUE(this->Run(256, 512, 256, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_512x256x256)
{
    EXPECT_TRUE(this->Run(512, 256, 256, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_512x512x256)
{
    EXPECT_TRUE(this->Run(512, 512, 256, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_64x64x32)
{
    EXPECT_TRUE(this->Run(64, 64, 32, 1));
}

// Unaligned shapes. They depend on the plain TDM GEMM handling the same N (the E row pitch is not
// a multiple of 8 elements); filter with --gtest_filter=-*Unaligned* if that is not yet the case.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_253x381x256)
{
    EXPECT_TRUE(this->Run(253, 381, 256, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_255x257x96)
{
    EXPECT_TRUE(this->Run(255, 257, 96, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_65x63x32)
{
    EXPECT_TRUE(this->Run(65, 63, 32, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_1x64x32)
{
    EXPECT_TRUE(this->Run(1, 64, 32, 1));
}

// Non-packed D0 and E row pitches.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_StrideD0_256x512x256)
{
    constexpr int M = 256;
    constexpr int N = 512;
    constexpr int K = 256;
    EXPECT_TRUE(this->Run(M, N, K, 1, 0, 0, N + 8, 0, 0));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_StrideE_256x512x256)
{
    constexpr int M = 256;
    constexpr int N = 512;
    constexpr int K = 256;
    EXPECT_TRUE(this->Run(M, N, K, 1, 0, 0, 0, 0, N + 16));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_StrideD0E_512x256x256)
{
    constexpr int M = 512;
    constexpr int N = 256;
    constexpr int K = 256;
    EXPECT_TRUE(this->Run(M, N, K, 1, 0, 0, N + 8, 0, N + 16));
}

// Ramp D0 / identity D1: every (m, n) sees a distinct D0 value and D1 is the identity of the
// elementwise op, so E depends on D0 alone. Most elements must differ from the GEMM without Ds
// and E must not be all zero, so a dropped or misplaced D0 fusion cannot pass.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_RampD0IdentityD1_256x256x128)
{
    EXPECT_TRUE(this->Run(256, 256, 128, 1, 0, 0, 0, 0, 0, MultiDFill::RampD0IdentityD1, true));
}

// Host-side argument check on gfx125: the TDM kernel is accepted for k_batch = 1.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_KBatch1Supported)
{
    EXPECT_TRUE(this->IsTdmSupported(256, 256, 256, 1));
}

// TDM requires exactly one split: its store overwrites E and cannot accumulate split-K.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_InvalidKBatchNotSupported)
{
    for(int k_batch : {0, -1, 2})
    {
        SCOPED_TRACE(k_batch);
        EXPECT_FALSE(this->IsTdmSupported(256, 256, 256, k_batch));
    }
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_NonPositiveKNotSupported)
{
    for(int K : {0, -1})
    {
        SCOPED_TRACE(K);
        EXPECT_FALSE(this->IsTdmSupported(256, 256, K, 1));
    }
}
