// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test TDM V2 with atomic barriers.
// This target is compiled with BARRIER_ATOMIC_IN_TDM=1.

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

// clang-format off
using KernelTypesCompTDMAtomic = ::testing::Types<
    // 128x128x32
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I32, I16, I16, Intrawave, CompTDMV2>,
    // 256x256x32
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I256, I256, I32, I16, I16, Intrawave, CompTDMV2>
>;
// clang-format on

template <typename T>
class TestCkTileGemmPipelineCompTDMAtomic
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMAtomic<T>>
{
    protected:
    // Regular non-cluster launch
    void SetUp() override
    {
        if(!ck_tile::is_gfx125_supported())
        {
            GTEST_SKIP() << "TDM atomic barriers require gfx1250";
        }
        this->k_batches_ = {1};
    }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMAtomic, KernelTypesCompTDMAtomic);

// Problem is exactly one block tile.
TYPED_TEST(TestCkTileGemmPipelineCompTDMAtomic, ExactTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

// Problem is 2x3=6 output tiles, with 8 K loops (K=8*K_Tile).
TYPED_TEST(TestCkTileGemmPipelineCompTDMAtomic, MultipleKLoops)
{
    this->Run(2 * TestFixture::M_Tile, 3 * TestFixture::N_Tile, 8 * TestFixture::K_Tile);
}
