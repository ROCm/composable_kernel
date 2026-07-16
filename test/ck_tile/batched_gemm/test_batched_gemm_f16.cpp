// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_batched_gemm_util.hpp"

using F16 = ck_tile::half_t;
using F32 = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    //             ALayout,     BLayout,     CLayout,   ADataType,   BDataType,   AccDataType,    CDataType
    std::tuple<    Row,         Row,         Row,       F16,         F16,         F32,            F16>,
    std::tuple<    Col,         Row,         Row,       F16,         F16,         F32,            F16>,
    std::tuple<    Row,         Col,         Row,       F16,         F16,         F32,            F16>,
    std::tuple<    Col,         Col,         Row,       F16,         F16,         F32,            F16>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileBatchedGemmF16 : public TestCkTileBatchedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileBatchedGemmF16, KernelTypes);

#define TEST_CKTILE_BGEMM_SUITE_NAME TestCkTileBatchedGemmF16

#include "test_batched_gemm_ut_cases.inc"
