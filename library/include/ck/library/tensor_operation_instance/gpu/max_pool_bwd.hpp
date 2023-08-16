// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_max_pool_bwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef __fp16__
void add_device_maxpool_bwd_f16_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<F16, I32, F16>>>&);
#endif
#ifdef __bf16__
void add_device_maxpool_bwd_bf16_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<BF16, I32, BF16>>>&);
#endif
#ifdef __fp32__
void add_device_maxpool_bwd_f32_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<F32, I32, F32>>>&);
#endif
template <typename DOutDataType, typename IndexDataType, typename DInDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceMaxPoolBwd<DOutDataType, IndexDataType, DInDataType>>
{
    using DeviceOp = DeviceMaxPoolBwd<DOutDataType, IndexDataType, DInDataType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef __fp16__
        if constexpr(is_same_v<DOutDataType, F16> && is_same_v<DInDataType, F16> &&
                     is_same_v<IndexDataType, I32>)
            add_device_maxpool_bwd_f16_instances(op_ptrs);
#endif
#ifdef __bf16__
        else if constexpr(is_same_v<DOutDataType, BF16> && is_same_v<DInDataType, BF16> &&
                          is_same_v<IndexDataType, I32>)
            add_device_maxpool_bwd_bf16_instances(op_ptrs);
#endif
#ifdef __fp32__
        else if constexpr(is_same_v<DOutDataType, F32> && is_same_v<DInDataType, F32> &&
                          is_same_v<IndexDataType, I32>)
            add_device_maxpool_bwd_f32_instances(op_ptrs);
#endif

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
