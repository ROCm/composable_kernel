// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_batched_gemm_util.hpp"

using BF16 = ck_tile::bf16_t;
using F16  = ck_tile::half_t;
using F32  = float;
using F8   = ck_tile::fp8_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    //             ALayout,     BLayout,     CLayout,   ADataType,   BDataType,   AccDataType,    CDataType
    std::tuple<    Row,         Row,         Row,       F8,          F8,          F32,            F16>,
    std::tuple<    Col,         Row,         Row,       F8,          F8,          F32,            F16>,
    std::tuple<    Row,         Col,         Row,       F8,          F8,          F32,            F16>,
    std::tuple<    Col,         Col,         Row,       F8,          F8,          F32,            F16>,
    std::tuple<    Row,         Row,         Row,       F8,          F8,          F32,            BF16>,
    std::tuple<    Col,         Row,         Row,       F8,          F8,          F32,            BF16>,
    std::tuple<    Row,         Col,         Row,       F8,          F8,          F32,            BF16>,
    std::tuple<    Col,         Col,         Row,       F8,          F8,          F32,            BF16>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileBatchedGemmF8 : public TestCkTileBatchedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileBatchedGemmF8, KernelTypes);

#define TEST_CKTILE_BGEMM_SUITE_NAME TestCkTileBatchedGemmF8

#include "test_batched_gemm_ut_cases.inc"
