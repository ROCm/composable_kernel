// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize1x1x128   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using GroupSize1x128x128 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for ABQuant split-K tests - Decode shape
// GemmConfigDecode: M_Tile=16, N_Tile=64, K_Tile=256, kPadK=false
// Constraints: M % 16 == 0, N % 64 == 0, K % (k_batch * 256) == 0
//
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
//                QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantSplitKDecodeTypes = ::testing::Types<
    // GroupSize 1x1x128 (kK=128 for both A and B, kN=1)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigDecode, GroupSize1x1x128, GroupSize1x1x128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigDecode, GroupSize1x1x128, GroupSize1x1x128, ColumnMajor>,
    // GroupSize 1x128x128 for B (kK=128, kN=128)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigDecode, GroupSize1x1x128, GroupSize1x128x128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigDecode, GroupSize1x1x128, GroupSize1x128x128, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant split-K Decode
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantSplitKDecodeTypes);

// ---- k_batch=2 ----------------------------------------------------------------
// Note: K=512 (= 2*K_Tile) is excluded because KRead=K_Tile=256, giving
// per_batch_num_loop=1 which the software-pipelined kernel cannot handle.

TYPED_TEST(TestCkTileGemmABQuant, SplitK2_MedK_BaseShape)
{
    // K=1024=4*256: standard decode decode shape
    this->run_test_with_validation(32, 64, 1024, 2);
}

TYPED_TEST(TestCkTileGemmABQuant, SplitK2_LargeK_WideN)
{
    // K=2048, larger N (multiple of N_Tile=64)
    this->run_test_with_validation(32, 256, 2048, 2);
}

TYPED_TEST(TestCkTileGemmABQuant, SplitK2_LargeK_TallM)
{
    // K=4096, larger M (multiple of M_Tile=16)
    this->run_test_with_validation(64, 64, 4096, 2);
}

// ---- k_batch=3 ----------------------------------------------------------------
// Note: K=768 (= 3*K_Tile) excluded: per_batch_num_loop=1.

TYPED_TEST(TestCkTileGemmABQuant, SplitK3_MedK_BaseShape)
{
    // K=1536=6*256
    this->run_test_with_validation(32, 64, 1536, 3);
}

TYPED_TEST(TestCkTileGemmABQuant, SplitK3_LargeK_BaseShape)
{
    // K=3072=12*256
    this->run_test_with_validation(32, 64, 3072, 3);
}

// ---- k_batch=4 ----------------------------------------------------------------
// Note: K=1024 (= 4*K_Tile) excluded: per_batch_num_loop=1.

TYPED_TEST(TestCkTileGemmABQuant, SplitK4_MedK_BaseShape)
{
    // K=2048=8*256
    this->run_test_with_validation(32, 64, 2048, 4);
}

TYPED_TEST(TestCkTileGemmABQuant, SplitK4_LargeK_WideN)
{
    // K=4096, wider N
    this->run_test_with_validation(32, 128, 4096, 4);
}

// ---- k_batch=5 ----------------------------------------------------------------
// Note: K=1280 (= 5*K_Tile) excluded: per_batch_num_loop=1.

TYPED_TEST(TestCkTileGemmABQuant, SplitK5_MedK_BaseShape)
{
    // K=2560=10*256
    this->run_test_with_validation(32, 64, 2560, 5);
}

// ---- k_batch=6 ----------------------------------------------------------------
// Note: K=1536 (= 6*K_Tile) excluded: per_batch_num_loop=1.

TYPED_TEST(TestCkTileGemmABQuant, SplitK6_LargeK_BaseShape)
{
    // K=3072=12*256
    this->run_test_with_validation(32, 64, 3072, 6);
}

// ---- k_batch=8 ----------------------------------------------------------------
// Note: K=2048 (= 8*K_Tile) excluded: per_batch_num_loop=1.

TYPED_TEST(TestCkTileGemmABQuant, SplitK8_LargeK_BaseShape)
{
    // K=4096=16*256
    this->run_test_with_validation(32, 64, 4096, 8);
}

TYPED_TEST(TestCkTileGemmABQuant, SplitK8_LargeK_LargeMN)
{
    // K=4096, larger M and N
    this->run_test_with_validation(48, 192, 4096, 8);
}
