// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// void add_device_permute_scale_1d_f16_instances(
//     std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>,
//                                                   ck::Tuple<F16>,
//                                                   element_wise::UnaryScaleSquare,
//                                                   1>>>&);

// void add_device_permute_scale_1d_f32_instances(
//     std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>,
//                                                   ck::Tuple<F32>,
//                                                   element_wise::UnaryScaleSquare,
//                                                   1>>>&);

// void add_device_permute_scale_2d_f16_instances(
//     std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>,
//                                                   ck::Tuple<F16>,
//                                                   element_wise::UnaryScaleSquare,
//                                                   2>>>&);

// void add_device_permute_scale_2d_f32_instances(
//     std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>,
//                                                   ck::Tuple<F32>,
//                                                   element_wise::UnaryScaleSquare,
//                                                   2>>>&);

void add_device_permute_scale_3d_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>,
                                                  ck::Tuple<F16>,
                                                  element_wise::UnaryScaleSquare,
                                                  3>>>&);

void add_device_permute_scale_3d_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>,
                                                  ck::Tuple<F32>,
                                                  element_wise::UnaryScaleSquare,
                                                  3>>>&);

void add_device_permute_scale_4d_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>,
                                                  ck::Tuple<F16>,
                                                  element_wise::UnaryScaleSquare,
                                                  4>>>&);

void add_device_permute_scale_4d_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>,
                                                  ck::Tuple<F32>,
                                                  element_wise::UnaryScaleSquare,
                                                  4>>>&);

void add_device_permute_scale_5d_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>,
                                                  ck::Tuple<F16>,
                                                  element_wise::UnaryScaleSquare,
                                                  5>>>&);

void add_device_permute_scale_5d_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>,
                                                  ck::Tuple<F32>,
                                                  element_wise::UnaryScaleSquare,
                                                  5>>>&);

void add_device_permute_scale_6d_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>,
                                                  ck::Tuple<F16>,
                                                  element_wise::UnaryScaleSquare,
                                                  6>>>&);

void add_device_permute_scale_6d_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>,
                                                  ck::Tuple<F32>,
                                                  element_wise::UnaryScaleSquare,
                                                  6>>>&);

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          index_t NumDim>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceElementwise<InDataTypeTuple,
                                                    OutDataTypeTuple,
                                                    ElementwiseOperation,
                                                    NumDim>>
{
    using DeviceOp = DeviceElementwise<InDataTypeTuple,
                                       OutDataTypeTuple,
                                       ElementwiseOperation,
                                       NumDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(NumDim == 1)
        {
            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
            {
                // add_device_permute_scale_1d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
            {
                // add_device_permute_scale_1d_f16_instances(op_ptrs);
            }
        }
        else if constexpr(NumDim == 2)
        {
            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
            {
                // add_device_permute_scale_2d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
            {
                // add_device_permute_scale_2d_f16_instances(op_ptrs);
            }
        }
        else if constexpr(NumDim == 3)
        {
            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
            {
                add_device_permute_scale_3d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
            {
                add_device_permute_scale_3d_f16_instances(op_ptrs);
            }
        }
        else if constexpr(NumDim == 4)
        {
            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
            {
                add_device_permute_scale_4d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
            {
                add_device_permute_scale_4d_f16_instances(op_ptrs);
            }
        }
        else if constexpr(NumDim == 5)
        {
            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
            {
                add_device_permute_scale_5d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
            {
                add_device_permute_scale_5d_f16_instances(op_ptrs);
            }
        }
        else if constexpr(NumDim == 6)
        {
            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
            {
                add_device_permute_scale_6d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
            {
                add_device_permute_scale_6d_f16_instances(op_ptrs);
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
