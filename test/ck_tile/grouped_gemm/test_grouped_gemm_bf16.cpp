// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util.hpp"

using F32   = float;
using BF16  = ck_tile::bf16_t;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using True  = ck_tile::bool_constant<true>;
using False = ck_tile::bool_constant<false>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, Persistent
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,       True>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,      False>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,       True>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,      False>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,       True>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,      False>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,       True>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,      False>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileGroupedGemmBF16 : public TestCkTileGroupedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGroupedGemmBF16, KernelTypes);

#define TEST_CKTILE_GGEMM_SUITE_NAME TestCkTileGroupedGemmBF16

#include "test_grouped_gemm_ut_cases.inc"
