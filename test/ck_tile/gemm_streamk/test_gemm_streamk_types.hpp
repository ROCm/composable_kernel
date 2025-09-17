// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"

using F16  = ck_tile::half_t;
using F32  = float;
using BF16 = ck_tile::bf16_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypesStreamK = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16>
>;

// clang-format on
