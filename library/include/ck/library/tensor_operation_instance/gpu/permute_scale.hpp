// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_permute_scale_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>,
                                                  ck::Tuple<F16>,
                                                  PassThrough,
                                                  element_wise::UnarySquare,
                                                  Scale,
                                                  4>>>&);

void add_device_permute_scale_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>,
                                                  ck::Tuple<F32>,
                                                  PassThrough,
                                                  element_wise::UnarySquare,
                                                  Scale,
                                                  4>>>&);

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          typename UnaryOperation,
          typename Scale,
          index_t NumDim>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceElementwise<InDataTypeTuple,
                                                    OutDataTypeTuple,
                                                    ElementwiseOperation,
                                                    UnaryOperation,
                                                    Scale,
                                                    NumDim>>
{
    using DeviceOp = DeviceElementwise<InDataTypeTuple,
                                       OutDataTypeTuple,
                                       ElementwiseOperation,
                                       UnaryOperation,
                                       Scale,
                                       NumDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                     is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
        {
            add_device_permute_scale_f32_instances(op_ptrs);
        }
        else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                          is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
        {
            add_device_permute_scale_f16_instances(op_ptrs);
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
