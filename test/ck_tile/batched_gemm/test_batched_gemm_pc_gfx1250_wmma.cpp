// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Batched GEMM with the gfx1250 producer/consumer TDM pipeline
// (GemmPipelineAgBgCrCompTDMProducerConsumer) and TdmEpilogue. k_batch == 1 only.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_batched_gemm_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using PC = BatchedGemmPipelineTypeConstant<BatchedGemmPipelineType::CompTDMPC>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, Pipeline
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       PC>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,       PC>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,       PC>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,       PC>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,      BF16,       PC>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileBatchedGemmPCWmma : public TestCkTileBatchedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileBatchedGemmPCWmma, KernelTypes);

#define TEST_CKTILE_BGEMM_SUITE_NAME TestCkTileBatchedGemmPCWmma

#include "test_batched_gemm_ut_cases.inc"

#undef TEST_CKTILE_BGEMM_SUITE_NAME
