// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_batched_gemm_util.hpp"

using INT8  = ck_tile::int8_t;
using INT32 = ck_tile::int32_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    //             ALayout,     BLayout,     CLayout,   ADataType,   BDataType,   AccDataType,    CDataType
    std::tuple<    Row,         Row,         Row,       INT8,        INT8,        INT32,          INT32>,
    std::tuple<    Col,         Row,         Row,       INT8,        INT8,        INT32,          INT32>,
    std::tuple<    Row,         Col,         Row,       INT8,        INT8,        INT32,          INT32>,
    std::tuple<    Col,         Col,         Row,       INT8,        INT8,        INT32,          INT32>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileBatchedGemmInt8 : public TestCkTileBatchedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileBatchedGemmInt8, KernelTypes);

#define TEST_CKTILE_BGEMM_SUITE_NAME TestCkTileBatchedGemmInt8

#include "test_batched_gemm_ut_cases.inc"
