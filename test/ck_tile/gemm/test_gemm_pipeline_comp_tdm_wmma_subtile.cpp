// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm / comp_tdm_v2 at 4 waves (2x2x1) with tiles whose
// VGPR estimate forces sub_tile_num > 1 in the TDM policy.
#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

// clang-format off
using KernelTypesCompTDMSubTile = ::testing::Types<
    // sub_tile_num = 2: the block GEMM hot loop must cover one K sub-tile only
    std::tuple<Row, Col, Row, F16,  F16,  F32, F16, I128, I128, I256, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F16,  F16,  F32, F16, I128, I128, I256, I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, BF16, BF16, F32, F16, I128, I128, I256, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, BF16, BF16, F32, F16, I128, I128, I256, I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, F16,  F16,  F32, F16, I128, I256, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, BF16, BF16, F32, F16, I128, I256, I128, I16, I16, Intrawave, CompTDMV2>,
    // sub_tile_num = 3: a 192-deep FP8 K tile holds three 64-deep warp tiles; the register
    // estimate asks for 2 sub-tiles (4 would also become 3), which the policy rounds up to 3
    // because 2 does not divide 3. K = 512 and 2048 end in a partial K tile; WholeKTiles does not.
    std::tuple<Row, Col, Row, F8,   F8,   F32, F16, I256, I256, I192, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F8,   F8,   F32, F16, I256, I256, I192, I16, I16, Intrawave, CompTDMV2>,
    // sub_tile_num = 1 controls
    std::tuple<Row, Col, Row, F16,  F16,  F32, F16, I128, I128, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, BF16, BF16, F32, F16, I128, I128, I128, I16, I16, Intrawave, CompTDMV2>
>;
// clang-format on

template <typename T>
class TestCkTileGemmPipelineCompTDMSubTile
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMSubTile<T>>
{
    protected:
    // These instances use the regular (non-cluster) launch, which also works on
    // asicRevision=0, so the base-class TDM revision skip does not apply here.
    // That also bypasses the base-class data-type skip, and Run() does nothing for an
    // unsupported combination, so every instance here must be supported.
    void SetUp() override
    {
        static_assert(TestCkTileGemmPipelineCompTDMSubTile<T>::check_data_type(),
                      "unsupported data type combination: this case would pass without running");
        this->k_batches_ = {1};
    }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMSubTile, KernelTypesCompTDMSubTile);

TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, Regular) { this->Run(512, 1024, 512); }
// K = 768 is a whole number of K tiles for every row (128, 192 and 256 deep).
TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, WholeKTiles) { this->Run(512, 1024, 768); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, LargeMatrix) { this->Run(2048, 2048, 2048); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, Llama) { this->Run(5608, 4096, 2048); }
