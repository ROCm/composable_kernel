// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// aliasing, for commonly used type
using F64  = double;
using F32  = float;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;

using F16_TUPLE     = ck::Tuple<F16>;
using F16_F16_TUPLE = ck::Tuple<F16, F16>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough    = ck::tensor_operation::element_wise::PassThrough;
using Bilinear       = ck::tensor_operation::element_wise::Bilinear;
using AddAddFastGelu = ck::tensor_operation::element_wise::AddAddFastGelu;

template <typename DeviceOp>
struct DeviceOperationInstanceFactory;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
