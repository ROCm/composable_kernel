// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_pool_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr auto InOutRank  = 5;
static constexpr auto WindowRank = 3;

static constexpr auto MaxOp = ck::ReduceTensorOp::MAX;
static constexpr auto AvgOp = ck::ReduceTensorOp::AVG;

// FP16
void add_device_pool3d_fwd_ndhwc_f16_instances(
    std::vector<
        std::unique_ptr<DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_f16_instances(
    std::vector<
        std::unique_ptr<DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, AvgOp, false>>>&);

// FP16 - return index
void add_device_pool3d_fwd_ndhwc_index_f16_instances(
    std::vector<
        std::unique_ptr<DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, MaxOp, true>>>&);

// FP32
void add_device_pool3d_fwd_ndhwc_f32_instances(
    std::vector<
        std::unique_ptr<DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_f32_instances(
    std::vector<
        std::unique_ptr<DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, AvgOp, false>>>&);

// FP32 - return index
void add_device_pool3d_fwd_ndhwc_index_f32_instances(
    std::vector<
        std::unique_ptr<DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, MaxOp, true>>>&);

template <typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          ck::ReduceTensorOp ReduceOpId,
          bool OutputIndex>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DevicePoolFwd<InOutRank,
                                                                                  WindowRank,
                                                                                  InDataType,
                                                                                  OutDataType,
                                                                                  IndexDataType,
                                                                                  ReduceOpId,
                                                                                  OutputIndex>>
{
    using DeviceOp = DevicePoolFwd<InOutRank,
                                   WindowRank,
                                   InDataType,
                                   OutDataType,
                                   IndexDataType,
                                   ReduceOpId,
                                   OutputIndex>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<InDataType, F16> && is_same_v<OutDataType, F16> &&
                     is_same_v<IndexDataType, I32>)
        {
            if constexpr(OutputIndex && ReduceOpId == MaxOp)
            {
                add_device_pool3d_fwd_ndhwc_index_f16_instances(op_ptrs);
            }
            else
            {
                add_device_pool3d_fwd_ndhwc_f16_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<InDataType, F32> && is_same_v<OutDataType, F32> &&
                          is_same_v<IndexDataType, I32>)
        {
            if constexpr(OutputIndex && ReduceOpId == MaxOp)
            {
                add_device_pool3d_fwd_ndhwc_index_f32_instances(op_ptrs);
            }
            else
            {
                add_device_pool3d_fwd_ndhwc_f32_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
