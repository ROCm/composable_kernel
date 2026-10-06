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
    void SetUp() override { this->k_batches_ = {1}; }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMSubTile, KernelTypesCompTDMSubTile);

TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, Regular) { this->Run(512, 1024, 512); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, LargeMatrix) { this->Run(2048, 2048, 2048); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, Llama) { this->Run(5608, 4096, 2048); }
