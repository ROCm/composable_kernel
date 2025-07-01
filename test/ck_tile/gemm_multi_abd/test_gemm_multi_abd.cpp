// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_multi_abd_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F8   = ck_tile::fp8_t;
using F32  = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    //          A0Layout, A0Layout, B0Layout, B0Layout, CLayout, D0Layout, D1Layout, A0DataType, A1DataType, B0DataType, B1DataType, D0DataType,  D1DataType, AccDataType, CDataType, CDElementWiseFn
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F16,         F16,        F16,       F16,       BF16,         BF16,     F32,      F32,     AddScale, AddScale, MultiplyMultiply>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F16,         F16,        F16,       F16,       F32,          F32,      F32,      F32,     AddScale, AddScale, MultiplyMultiply>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F16,         F16,        F16,       F16,       F32,          F32,      F32,      F32,     AddScale, AddScale, MultiplyMultiply>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F8,          F8,         F8,        F8,        BF16,         BF16,     F32,      F32,     AddScale, AddScale, MultiplyMultiply>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F8,          F8,         F8,        F8,        F8,           F8,       F32,      F32,     AddScale, AddScale, MultiplyMultiply>,

    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F16,         F16,        F16,       F16,       BF16,         BF16,     F32,      F32,     AlphaBetaAdd, AddScale, MultiplyMultiply>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F16,         F16,        F16,       F16,       F32,          F32,      F32,      F32,     AlphaBetaAdd, AddScale, MultiplyMultiply>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F16,         F16,        F16,       F16,       F32,          F32,      F32,      F32,     AlphaBetaAdd, AddScale, MultiplyMultiply>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F8,          F8,         F8,        F8,        BF16,         BF16,     F32,      F32,     AlphaBetaAdd, AddScale, MultiplyMultiply>,
    std::tuple<    Row,     Row,     Col,     Col,      Row,      Row,       Row,      F8,          F8,         F8,        F8,        F8,           F8,       F32,      F32,     AlphaBetaAdd, AddScale, MultiplyMultiply>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmMultiABD, KernelTypes);

#include "test_gemm_multi_abd_ut_cases.inc"
