// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_3d_impl.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::transpose_op;

void add_device_transpose_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise3dImpl<F16, F16, NCDHW, 3>>>& instances);

void add_device_transpose_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise3dImpl<F32, F32, NCDHW, 3>>>& instances);

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          index_t NumDim>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::
        DeviceElementwise3dImpl<InDataTypeTuple, OutDataTypeTuple, ElementwiseOperation, NumDim>>
{
    using DeviceOp = DeviceElementwise3dImpl<InDataTypeTuple,
                                             OutDataTypeTuple,
                                             ElementwiseOperation,
                                             NumDim_m, // choose how to set dims
                                             NumDim_n,
                                             NumDim_k,
                                             MPerThread,
                                             NPerThread,
                                             KPerThread,
                                             InScalarPerVectorSeq,
                                             OutScalarPerVectorSeq>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
        {
            add_device_transpose_f32_instances(op_ptrs);
        }
        else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
        {
            add_device_transpose_f16_instances(op_ptrs);
        }
    }
    return op_ptrs;
}
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
