// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_scale_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

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
using device_permute_scale_f16_instances =
    std::tuple <
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, 4, 1, ck::Sequence<1>, ck::Sequence<1>>,
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, 4, 8, ck::Sequence<1>, ck::Sequence<1>>,
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, 4, 8, ck::Sequence<8>, ck::Sequence<1>>,
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, 4, 8, ck::Sequence<1>, ck::Sequence<8>>,
        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, 4, 2, ck::Sequence<1>, ck::Sequence<1>>
    >;

using device_permute_scale_f32_instances = std::tuple<
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, Pass, UnaryOp, Scale, 4, 1, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, Pass, UnaryOp, Scale, 4, 8, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, Pass, UnaryOp, Scale, 4, 2, ck::Sequence<1>, ck::Sequence<1>>
    >;
// clang-format on

void add_device_permute_scale_f16_instances(
    std::vector<std::unique_ptr<
        DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, Pass, UnaryOp, Scale, 4>>>& instances)
{
    add_device_operation_instances(instances, device_permute_scale_f16_instances{});
}

void add_device_permute_scale_f32_instances(
    std::vector<std::unique_ptr<
        DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, Pass, UnaryOp, Scale, 4>>>& instances)
{
    add_device_operation_instances(instances, device_permute_scale_f32_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
