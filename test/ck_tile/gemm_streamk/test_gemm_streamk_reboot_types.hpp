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

using Persistent    = std::true_type;
using NonPersistent = std::false_type;

// clang-format off
using KernelTypesStreamKFp16 = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  Persistent
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,     Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,     Persistent>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,     Persistent>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,     Persistent>
>;

using KernelTypesStreamKBf16 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,    Persistent>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,    Persistent>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       BF16,    Persistent>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       BF16,    Persistent>
>;
// clang-format on
