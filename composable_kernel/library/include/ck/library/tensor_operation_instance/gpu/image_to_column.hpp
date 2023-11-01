// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_image_to_column.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// nhwc, 1d
void add_device_image_to_column_nhwc_1d_bf16_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<1, GNWC, BF16, BF16>>>& instances);

void add_device_image_to_column_nhwc_1d_f16_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<1, GNWC, F16, F16>>>& instances);

void add_device_image_to_column_nhwc_1d_f32_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<1, GNWC, F32, F32>>>& instances);

void add_device_image_to_column_nhwc_1d_i8_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<1, GNWC, int8_t, int8_t>>>& instances);
// nhwc, 2d
void add_device_image_to_column_nhwc_2d_bf16_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<2, GNHWC, BF16, BF16>>>& instances);

void add_device_image_to_column_nhwc_2d_f16_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<2, GNHWC, F16, F16>>>& instances);

void add_device_image_to_column_nhwc_2d_f32_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<2, GNHWC, F32, F32>>>& instances);

void add_device_image_to_column_nhwc_2d_i8_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<2, GNHWC, int8_t, int8_t>>>& instances);
// nhwc, 3d
void add_device_image_to_column_nhwc_3d_bf16_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<3, GNDHWC, BF16, BF16>>>& instances);

void add_device_image_to_column_nhwc_3d_f16_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<3, GNDHWC, F16, F16>>>& instances);

void add_device_image_to_column_nhwc_3d_f32_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<3, GNDHWC, F32, F32>>>& instances);

void add_device_image_to_column_nhwc_3d_i8_instances(
    std::vector<std::unique_ptr<DeviceImageToColumn<3, GNDHWC, int8_t, int8_t>>>& instances);

template <ck::index_t NumDimSpatial, typename InLayout, typename InDataType, typename OutDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::
        DeviceImageToColumn<NumDimSpatial, InLayout, InDataType, OutDataType>>
{
    using DeviceOp = DeviceImageToColumn<NumDimSpatial, InLayout, InDataType, OutDataType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(NumDimSpatial == 1 && is_same_v<InLayout, GNWC>)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
            {
                add_device_image_to_column_nhwc_1d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
            {
                add_device_image_to_column_nhwc_1d_f16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                              is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_image_to_column_nhwc_1d_bf16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, int8_t> && is_same_v<OutDataType, int8_t>)
            {
                add_device_image_to_column_nhwc_1d_i8_instances(op_ptrs);
            }
        }
        else if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, GNHWC>)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
            {
                add_device_image_to_column_nhwc_2d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
            {
                add_device_image_to_column_nhwc_2d_f16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                              is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_image_to_column_nhwc_2d_bf16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, int8_t> && is_same_v<OutDataType, int8_t>)
            {
                add_device_image_to_column_nhwc_2d_i8_instances(op_ptrs);
            }
        }
        else if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, GNDHWC>)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
            {
                add_device_image_to_column_nhwc_3d_f32_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
            {
                add_device_image_to_column_nhwc_3d_f16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                              is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_image_to_column_nhwc_3d_bf16_instances(op_ptrs);
            }
            else if constexpr(is_same_v<InDataType, int8_t> && is_same_v<OutDataType, int8_t>)
            {
                add_device_image_to_column_nhwc_3d_i8_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
