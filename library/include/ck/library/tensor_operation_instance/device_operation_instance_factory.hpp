// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// aliasing, for commonly used data type
using F64  = double;
using F32  = float;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;

using Empty_Tuple = ck::Tuple<>;

using F16_Tuple     = ck::Tuple<F16>;
using F16_F16_Tuple = ck::Tuple<F16, F16>;

using F32_Tuple = ck::Tuple<F32>;

// GEMM layout
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Row_Tuple     = ck::Tuple<Row>;
using Row_Row_Tuple = ck::Tuple<Row, Row>;

// Conv layout
//
using NWC   = ck::tensor_layout::convolution::NWC;
using NHWC  = ck::tensor_layout::convolution::NHWC;
using NDHWC = ck::tensor_layout::convolution::NDHWC;

using KXC   = ck::tensor_layout::convolution::KXC;
using KYXC  = ck::tensor_layout::convolution::KYXC;
using KZYXC = ck::tensor_layout::convolution::KZYXC;

using NWK   = ck::tensor_layout::convolution::NWK;
using NHWK  = ck::tensor_layout::convolution::NHWK;
using NDHWK = ck::tensor_layout::convolution::NDHWK;

//
using GNWC   = ck::tensor_layout::convolution::GNWC;
using GNHWC  = ck::tensor_layout::convolution::GNHWC;
using GNDHWC = ck::tensor_layout::convolution::GNDHWC;

using GKXC   = ck::tensor_layout::convolution::GKXC;
using GKYXC  = ck::tensor_layout::convolution::GKYXC;
using GKZYXC = ck::tensor_layout::convolution::GKZYXC;

using GNWK   = ck::tensor_layout::convolution::GNWK;
using GNHWK  = ck::tensor_layout::convolution::GNHWK;
using GNDHWK = ck::tensor_layout::convolution::GNDHWK;

//
using NWGC   = ck::tensor_layout::convolution::NWGC;
using NHWGC  = ck::tensor_layout::convolution::NHWGC;
using NDHWGC = ck::tensor_layout::convolution::NDHWGC;

using KXGC   = ck::tensor_layout::convolution::KXGC;
using KYXGC  = ck::tensor_layout::convolution::KYXGC;
using KZYXGC = ck::tensor_layout::convolution::KZYXGC;

using NWGK   = ck::tensor_layout::convolution::NWGK;
using NHWGK  = ck::tensor_layout::convolution::NHWGK;
using NDHWGK = ck::tensor_layout::convolution::NDHWGK;

// pointwise functor
using PassThrough    = ck::tensor_operation::element_wise::PassThrough;
using Scale          = ck::tensor_operation::element_wise::Scale;
using Bilinear       = ck::tensor_operation::element_wise::Bilinear;
using AddAddFastGelu = ck::tensor_operation::element_wise::AddAddFastGelu;

template <typename DeviceOp>
struct DeviceOperationInstanceFactory;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
