// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util.hpp"

using BF16  = ck_tile::bf16_t;
using F16   = ck_tile::half_t;
using F32   = float;
using F8    = ck_tile::fp8_t;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using True  = ck_tile::bool_constant<true>;
using False = ck_tile::bool_constant<false>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, Persistent
    std::tuple<    Row,     Col,     Row,        F8,        F8,         F32,       F16,       True>,
    std::tuple<    Row,     Col,     Row,        F8,        F8,         F32,       F16,      False>,
    std::tuple<    Col,     Col,     Row,        F8,        F8,         F32,       F16,       True>,
    std::tuple<    Col,     Col,     Row,        F8,        F8,         F32,       F16,      False>,
    std::tuple<    Row,     Row,     Row,        F8,        F8,         F32,       F16,       True>,
    std::tuple<    Row,     Row,     Row,        F8,        F8,         F32,       F16,      False>,
    std::tuple<    Col,     Row,     Row,        F8,        F8,         F32,       F16,       True>,
    std::tuple<    Col,     Row,     Row,        F8,        F8,         F32,       F16,      False>,
    std::tuple<    Row,     Col,     Row,        F8,        F8,         F32,      BF16,       True>,
    std::tuple<    Row,     Col,     Row,        F8,        F8,         F32,      BF16,      False>,
    std::tuple<    Col,     Col,     Row,        F8,        F8,         F32,      BF16,       True>,
    std::tuple<    Col,     Col,     Row,        F8,        F8,         F32,      BF16,      False>,
    std::tuple<    Row,     Row,     Row,        F8,        F8,         F32,      BF16,       True>,
    std::tuple<    Row,     Row,     Row,        F8,        F8,         F32,      BF16,      False>,
    std::tuple<    Col,     Row,     Row,        F8,        F8,         F32,      BF16,       True>,
    std::tuple<    Col,     Row,     Row,        F8,        F8,         F32,      BF16,      False>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileGroupedGemmF8 : public TestCkTileGroupedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGroupedGemmF8, KernelTypes);

#define TEST_CKTILE_GGEMM_SUITE_NAME TestCkTileGroupedGemmF8

#include "test_grouped_gemm_ut_cases.inc"
