// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Every gfx1250 GEMM pipeline (mem, compv3, compv4, comp_async, comp_tdm v1/v2) for each quant
// type. Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType,
// CDataType, QuantType, GemmConfig, QuantGroupSize>
template <typename QuantT, typename Config>
using Fp8Rcr = std::tuple<RowMajor,
                          ColumnMajor,
                          RowMajor,
                          RowMajor,
                          FP8,
                          FP8,
                          float,
                          Half,
                          QuantT,
                          Config,
                          GroupSize1D_128>;

template <typename QuantT>
using AllPipelines = ::testing::Types<Fp8Rcr<QuantT, GemmConfigMem>,
                                      Fp8Rcr<QuantT, GemmConfigCompV3>,
                                      Fp8Rcr<QuantT, GemmConfigCompV4>,
                                      Fp8Rcr<QuantT, GemmConfigCompAsync>,
                                      Fp8Rcr<QuantT, GemmConfigCompTDMV1>,
                                      Fp8Rcr<QuantT, GemmConfigCompTDMV2>>;

TYPED_TEST_SUITE(TestCkTileGemmRowColQuant, AllPipelines<RowColQuant>);
TYPED_TEST(TestCkTileGemmRowColQuant, AllPipelines)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

TYPED_TEST_SUITE(TestCkTileGemmTensorQuant, AllPipelines<TensorQuant>);
TYPED_TEST(TestCkTileGemmTensorQuant, AllPipelines)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

// Group-quant ops on the gfx1250 pipelines. Tuple format as above; the quant layout is B's.
template <typename Config>
using BQuantFp8Rcr = std::tuple<RowMajor,
                                ColumnMajor,
                                RowMajor,
                                ColumnMajor,
                                FP8,
                                FP8,
                                float,
                                Half,
                                BQuantGrouped,
                                Config,
                                GroupSize1D_128>;

using BQuantPipelines = ::testing::Types<BQuantFp8Rcr<GemmConfigMem>,
                                         BQuantFp8Rcr<GemmConfigCompV3>,
                                         BQuantFp8Rcr<GemmConfigCompV4>,
                                         BQuantFp8Rcr<GemmConfigCompAsync>,
                                         BQuantFp8Rcr<GemmConfigCompTDMV1>,
                                         BQuantFp8Rcr<GemmConfigCompTDMV2>>;

TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuantPipelines);
TYPED_TEST(TestCkTileGemmBQuant, Pipelines) { this->run_test_with_validation(1024, 1024, 1024); }

template <typename Config>
using AQuantFp8Rcr = std::tuple<RowMajor,
                                ColumnMajor,
                                RowMajor,
                                RowMajor,
                                FP8,
                                FP8,
                                float,
                                Half,
                                AQuantGrouped,
                                Config,
                                GroupSize1D_128>;

using AQuantPipelines = ::testing::Types<AQuantFp8Rcr<GemmConfigMem>,
                                         AQuantFp8Rcr<GemmConfigCompV3>,
                                         AQuantFp8Rcr<GemmConfigCompV4>,
                                         AQuantFp8Rcr<GemmConfigCompAsync>,
                                         AQuantFp8Rcr<GemmConfigCompTDMV1>,
                                         AQuantFp8Rcr<GemmConfigCompTDMV2>>;

TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantPipelines);
TYPED_TEST(TestCkTileGemmAQuant, Pipelines) { this->run_test_with_validation(1024, 1024, 1024); }

// ABQuant tuple appends <BQuantGroupSize, BQLayout>.
template <typename Config>
using ABQuantFp8Rcr = std::tuple<RowMajor,
                                 ColumnMajor,
                                 RowMajor,
                                 RowMajor,
                                 FP8,
                                 FP8,
                                 float,
                                 Half,
                                 ABQuantGrouped,
                                 Config,
                                 GroupSize1D_128,
                                 GroupSize1D_128,
                                 ColumnMajor>;

using ABQuantPipelines = ::testing::Types<ABQuantFp8Rcr<GemmConfigMem>,
                                          ABQuantFp8Rcr<GemmConfigCompV3>,
                                          ABQuantFp8Rcr<GemmConfigCompV4>,
                                          ABQuantFp8Rcr<GemmConfigCompAsync>,
                                          ABQuantFp8Rcr<GemmConfigCompTDMV1>,
                                          ABQuantFp8Rcr<GemmConfigCompTDMV2>>;

TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantPipelines);
TYPED_TEST(TestCkTileGemmABQuant, Pipelines) { this->run_test_with_validation(1024, 1024, 1024); }
