// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_layernorm_f16_rank2_instances(
    std::vector<DeviceLayernormPtr<F16, F16, F16, F32, F16, PassThrough, 2, 1>>&);

void add_device_layernorm_f16_rank4_instances(
    std::vector<DeviceLayernormPtr<F16, F16, F16, F32, F16, PassThrough, 4, 3>>&);

void add_device_layernorm_f32_rank2_instances(
    std::vector<DeviceLayernormPtr<F32, F32, F32, F32, F32, PassThrough, 2, 1>>&);

void add_device_layernorm_f32_rank4_instances(
    std::vector<DeviceLayernormPtr<F32, F32, F32, F32, F32, PassThrough, 4, 3>>&);

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceLayernorm<XDataType,
                                                  GammaDataType,
                                                  BetaDataType,
                                                  F32,
                                                  YDataType,
                                                  ck::tensor_operation::element_wise::PassThrough,
                                                  Rank,
                                                  NumReduceDim>>
{
    using DeviceOp = DeviceLayernorm<XDataType,
                                     GammaDataType,
                                     BetaDataType,
                                     F32,
                                     YDataType,
                                     ck::tensor_operation::element_wise::PassThrough,
                                     Rank,
                                     NumReduceDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<XDataType, F16> && is_same_v<GammaDataType, F16> &&
                     is_same_v<BetaDataType, F16> && is_same_v<YDataType, F16>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
                add_device_layernorm_f16_rank2_instances(op_ptrs);
            else if constexpr(Rank == 4 && NumReduceDim == 3)
                add_device_layernorm_f16_rank4_instances(op_ptrs);
        }
        else if constexpr(is_same_v<XDataType, F32> && is_same_v<GammaDataType, F32> &&
                          is_same_v<BetaDataType, F32> && is_same_v<YDataType, F32>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
                add_device_layernorm_f32_rank2_instances(op_ptrs);
            else if constexpr(Rank == 4 && NumReduceDim == 3)
                add_device_layernorm_f32_rank4_instances(op_ptrs);
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
