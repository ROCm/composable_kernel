// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_scale_impl.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Pass    = ck::tensor_operation::element_wise::PassThrough;
using UnaryOp = ck::tensor_operation::element_wise::UnarySquare;
using Scale   = ck::tensor_operation::element_wise::Scale;

// clang-format off
template <index_t NDims>
using device_permute_scale_f16_instances =
    std::tuple <
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, NDims, 1, ck::Sequence<1>, ck::Sequence<1>>,
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, NDims, 8, ck::Sequence<8>, ck::Sequence<1>>,
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, NDims, 4, ck::Sequence<4>, ck::Sequence<1>>,
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, NDims, 2, ck::Sequence<2>, ck::Sequence<1>>
    >;

template <index_t NDims>
using device_permute_scale_f32_instances = std::tuple<
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, Pass, UnaryOp, Scale, NDims, 1, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, Pass, UnaryOp, Scale, NDims, 8, ck::Sequence<8>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, Pass, UnaryOp, Scale, NDims, 4, ck::Sequence<4>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, Pass, UnaryOp, Scale, NDims, 2, ck::Sequence<2>, ck::Sequence<1>>
    >;
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
