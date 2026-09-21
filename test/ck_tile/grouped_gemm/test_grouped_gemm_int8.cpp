// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util.hpp"

using INT8  = ck_tile::int8_t;
using INT32 = ck_tile::int32_t;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using True  = ck_tile::bool_constant<true>;
using False = ck_tile::bool_constant<false>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, Persistent
    std::tuple<    Row,     Col,     Row,      INT8,      INT8,       INT32,     INT32,       True>,
    std::tuple<    Row,     Col,     Row,      INT8,      INT8,       INT32,     INT32,      False>,
    std::tuple<    Col,     Col,     Row,      INT8,      INT8,       INT32,     INT32,       True>,
    std::tuple<    Col,     Col,     Row,      INT8,      INT8,       INT32,     INT32,      False>,
    std::tuple<    Row,     Row,     Row,      INT8,      INT8,       INT32,     INT32,       True>,
    std::tuple<    Row,     Row,     Row,      INT8,      INT8,       INT32,     INT32,      False>,
    std::tuple<    Col,     Row,     Row,      INT8,      INT8,       INT32,     INT32,       True>,
    std::tuple<    Col,     Row,     Row,      INT8,      INT8,       INT32,     INT32,      False>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileGroupedGemmInt8 : public TestCkTileGroupedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGroupedGemmInt8, KernelTypes);

#define TEST_CKTILE_GGEMM_SUITE_NAME TestCkTileGroupedGemmInt8

#include "test_grouped_gemm_ut_cases.inc"
