// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool2d_fwd_impl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool3d_fwd_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I32 = int32_t;
using F16 = ck::half_t;
using F32 = float;

template <typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          typename ComputeDataType,
          ReduceTensorOp ReduceOpId,
          bool OutputIndex>
using device_pool2d_fwd_nhwc_instances =
    // clang-format off
    std::tuple <
        DevicePool2dFwdImpl<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 1, 1, 1, false>,
        DevicePool2dFwdImpl<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 2, 1, 2, false>,
        DevicePool2dFwdImpl<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 4, 1, 4, false>
                 // clang-format on
                 >;

template <typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          typename ComputeDataType,
          ReduceTensorOp ReduceOpId,
          bool OutputIndex>
using device_pool3d_fwd_ndhwc_instances =
    // clang-format off
    std::tuple <
        DevicePool3dFwdImpl<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 1, 1, 1, false>,
        DevicePool3dFwdImpl<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 2, 1, 2, false>,
        DevicePool3dFwdImpl<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 4, 1, 4, false>
                 // clang-format on
                 >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
