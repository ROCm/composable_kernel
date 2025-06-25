// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using BF16_Tuple = ck::Tuple<BF16>;

using ADataType        = BF16;
using BDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = BF16;
using DsDataType       = BF16_Tuple;
using EDataType        = BF16;

using Row_Tuple = ck::Tuple<Row>;

using ALayout  = Row;
using BLayout  = Row;
using DLayout  = Row;
using DsLayout = Row_Tuple;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffleV3<
    Row,
    Row,
    Row_Tuple,
    Row,
    BF16,
    BF16,
    BF16_Tuple,
    BF16,
    F32,
    F32,
    PassThrough,
    PassThrough,
    Add,
    GemmSpec,
    128,
    128,
    64,
    64,
    8,
    8,
    16,
    16,
    4,
    2,
    S<4, 32, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    0,
    S<4, 32, 1>,
    S<0, 2, 1>,
    S<0, 2, 1>,
    1,
    1,
    8,
    0,
    1,
    1,
    S<1, 32, 1, 4>,
    S<8, 8, 8>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1>;
}