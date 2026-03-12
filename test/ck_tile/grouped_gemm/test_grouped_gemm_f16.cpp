// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util.hpp"

using F16   = ck_tile::half_t;
using F32   = float;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using True  = ck_tile::bool_constant<true>;
using False = ck_tile::bool_constant<false>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, Persistent
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       True>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,      False>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,       True>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,      False>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,       True>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,      False>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,       True>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,      False>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileGroupedGemmF16 : public TestCkTileGroupedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGroupedGemmF16, KernelTypes);

#define TEST_CKTILE_GGEMM_SUITE_NAME TestCkTileGroupedGemmF16

#include "test_grouped_gemm_ut_cases.inc"
