// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Cases for the gfx1250 emulator smoke stage, where each gtest binary must finish well
 *        under a minute.
 * @details Apart from SingleRow (M = 1), each problem spans several tiles in M and N. For the
 *          64x64 tiles of the type lists that include this file, none of these shapes is in
 *          test_mx_gemm_pipeline_ut_cases.inc (which uses N = N_Tile, apart from its 512^3 case),
 *          so a type list shared with a ut_cases binary is not tested twice.
 */

#pragma once

#include <stdexcept>
#include <type_traits>

TYPED_TEST(TEST_SUITE_NAME, KLoopPaths)
{
    // 1..4 K tiles cover the four K-loop paths of the TDM pipelines: no hot loop with tail One
    // and with tail Two, then the hot loop with tail One and with tail Two.
    for(int k_tiles : {1, 2, 3, 4})
    {
        this->Run(4 * TestFixture::M_Tile, 4 * TestFixture::N_Tile, k_tiles * TestFixture::K_Tile);
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleRow)
{
    using ADataType = typename TestFixture::ADataType;
    using ALayout   = typename TestFixture::ALayout;
    if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor> &&
                 ck_tile::numeric_traits<ADataType>::PackedSize > 1)
    {
        // M is the contiguous dimension of a column-major A, and M = 1 is not a whole number of
        // packs, so IsSupportedArgument rejects the argument and Run throws.
        EXPECT_THROW(this->Run(1, 4 * TestFixture::N_Tile, 4 * TestFixture::K_Tile),
                     std::runtime_error);
    }
    else
    {
        this->Run(1, 4 * TestFixture::N_Tile, 4 * TestFixture::K_Tile);
    }
}

TYPED_TEST(TEST_SUITE_NAME, PartialTiles)
{
    // 3.25 x 3.5 tiles at 64x64: partial last tiles in M and N.
    this->Run(3 * TestFixture::M_Tile + 16, 3 * TestFixture::N_Tile + 32, 2 * TestFixture::K_Tile);
}
