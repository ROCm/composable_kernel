// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_mha_bwd_qloop_light_v1_casual_f16_f16_instances(
    std::vector<std::unique_ptr<DeviceBatchedMultiheadAttentionBackwardQloopLightV1<
        2,
        1,
        1,
        1,
        1,
        F16,
        F16,
        unsigned short,
        F32,
        F32,
        ck::Tuple<>,
        ck::Tuple<>,
        PassThrough,
        PassThrough,
        Scale,
        PassThrough,
        PassThrough,
        MaskingSpecialization::MaskUpperTriangleFromTopLeft>>>& instances);

void add_device_batched_mha_bwd_qloop_light_v1_noncasual_f16_f16_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedMultiheadAttentionBackwardQloopLightV1<2,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            F16,
                                                            F16,
                                                            unsigned short,
                                                            F32,
                                                            F32,
                                                            ck::Tuple<>,
                                                            ck::Tuple<>,
                                                            PassThrough,
                                                            PassThrough,
                                                            Scale,
                                                            PassThrough,
                                                            PassThrough,
                                                            MaskingSpecialization::MaskDisabled>>>&
        instances);

void add_device_batched_mha_bwd_qloop_light_v1_casual_bf16_bf16_instances(
    std::vector<std::unique_ptr<DeviceBatchedMultiheadAttentionBackwardQloopLightV1<
        2,
        1,
        1,
        1,
        1,
        BF16,
        BF16,
        unsigned short,
        F32,
        F32,
        ck::Tuple<>,
        ck::Tuple<>,
        PassThrough,
        PassThrough,
        Scale,
        PassThrough,
        PassThrough,
        MaskingSpecialization::MaskUpperTriangleFromTopLeft>>>& instances);

void add_device_batched_mha_bwd_qloop_light_v1_noncasual_bf16_bf16_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedMultiheadAttentionBackwardQloopLightV1<2,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            BF16,
                                                            BF16,
                                                            unsigned short,
                                                            F32,
                                                            F32,
                                                            ck::Tuple<>,
                                                            ck::Tuple<>,
                                                            PassThrough,
                                                            PassThrough,
                                                            Scale,
                                                            PassThrough,
                                                            PassThrough,
                                                            MaskingSpecialization::MaskDisabled>>>&
        instances);

template <typename InputDataType,
          typename OutputDataType,
          typename ZDataType,
          typename LSEDataType,
          typename DDataType,
          MaskingSpecialization MaskingSpec>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackwardQloopLightV1<
        2,
        1,
        1,
        1,
        1,
        InputDataType,
        OutputDataType,
        ZDataType,
        LSEDataType,
        DDataType,
        ck::Tuple<>,
        ck::Tuple<>,
        PassThrough,
        PassThrough,
        Scale,
        PassThrough,
        PassThrough,
        MaskingSpec>>
{
    using DeviceOp = DeviceBatchedMultiheadAttentionBackwardQloopLightV1<2,
                                                                         1,
                                                                         1,
                                                                         1,
                                                                         1,
                                                                         InputDataType,
                                                                         OutputDataType,
                                                                         ZDataType,
                                                                         LSEDataType,
                                                                         DDataType,
                                                                         ck::Tuple<>,
                                                                         ck::Tuple<>,
                                                                         PassThrough,
                                                                         PassThrough,
                                                                         Scale,
                                                                         PassThrough,
                                                                         PassThrough,
                                                                         MaskingSpec>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<InputDataType, half_t> && is_same_v<OutputDataType, half_t> &&
                     is_same_v<ZDataType, unsigned short> && is_same_v<LSEDataType, float> &&
                     is_same_v<DDataType, float>)
        {
            if constexpr(MaskingSpec == MaskingSpecialization::MaskUpperTriangleFromTopLeft)
            {
                add_device_batched_mha_bwd_qloop_light_v1_casual_f16_f16_instances(op_ptrs);
            }
            else if(MaskingSpec == MaskingSpecialization::MaskDisabled)
            {
                add_device_batched_mha_bwd_qloop_light_v1_noncasual_f16_f16_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<InputDataType, BF16> && is_same_v<OutputDataType, BF16> &&
                          is_same_v<ZDataType, unsigned short> && is_same_v<LSEDataType, float> &&
                          is_same_v<DDataType, float>)
        {
            if constexpr(MaskingSpec == MaskingSpecialization::MaskUpperTriangleFromTopLeft)
            {
                add_device_batched_mha_bwd_qloop_light_v1_casual_bf16_bf16_instances(op_ptrs);
            }
            else if(MaskingSpec == MaskingSpecialization::MaskDisabled)
            {
                add_device_batched_mha_bwd_qloop_light_v1_noncasual_bf16_bf16_instances(op_ptrs);
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
