// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_3d_impl.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using device_transpose_f16_instances = std::tuple<
    // FOR 16, 32, 16, 32, 16
    // clang-format off
    DeviceElementwise3dImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 2, 2, 1, 8, 8, 8, ck::Sequence<1>, ck::Sequence<1>>,
    DeviceElementwise3dImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 2, 2, 1, 8, 1, 1, ck::Sequence<1>, ck::Sequence<1>>,
    DeviceElementwise3dImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 2, 2, 1, 8, 4, 4, ck::Sequence<1>, ck::Sequence<1>>
    // clang-format on
    >;

using device_transpose_f32_instances = std::tuple<
    // for 16, 8, 16, 32, 8 -> test with instances for fp16
    // clang-format off
    DeviceElementwise3dImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 2, 2, 1, 4, 4, 4, ck::Sequence<1>, ck::Sequence<1>>,
    DeviceElementwise3dImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 2, 2, 1, 4, 8, 4, ck::Sequence<1>, ck::Sequence<1>>,
    DeviceElementwise3dImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 2, 2, 1, 4, 8, 8, ck::Sequence<1>, ck::Sequence<1>>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
