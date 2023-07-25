// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using NHWGC = ck::tensor_layout::convolution::NHWGC;
using GNHWC = ck::tensor_layout::convolution::GNHWC;

using GKYXC = ck::tensor_layout::convolution::GKYXC;

using NHWGK = ck::tensor_layout::convolution::NHWGK;
using GNHWK = ck::tensor_layout::convolution::GNHWK;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

static constexpr auto ConvFwdOddC =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
