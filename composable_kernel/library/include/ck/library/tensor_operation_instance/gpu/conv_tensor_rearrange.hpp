// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_tensor_rearrange.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/conv_tensor_rearrange_op.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::conv_tensor_rearrange_op;

// Image to Column
// nhwc, 1d
void add_device_image_to_column_nwc_1d_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, BF16, BF16, ImageToColumn>>>&
        instances);

void add_device_image_to_column_nwc_1d_f16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, F16, F16, ImageToColumn>>>&
        instances);

void add_device_image_to_column_nwc_1d_f32_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, F32, F32, ImageToColumn>>>&
        instances);

void add_device_image_to_column_nwc_1d_i8_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, int8_t, int8_t, ImageToColumn>>>&
        instances);
// nhwc, 2d
void add_device_image_to_column_nhwc_2d_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<2, GNHWC, BF16, BF16, ImageToColumn>>>&
        instances);

void add_device_image_to_column_nhwc_2d_f16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<2, GNHWC, F16, F16, ImageToColumn>>>&
        instances);

void add_device_image_to_column_nhwc_2d_f32_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<2, GNHWC, F32, F32, ImageToColumn>>>&
        instances);

void add_device_image_to_column_nhwc_2d_i8_instances(
    std::vector<
        std::unique_ptr<DeviceConvTensorRearrange<2, GNHWC, int8_t, int8_t, ImageToColumn>>>&
        instances);
// nhwc, 3d
void add_device_image_to_column_ndhwc_3d_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, GNDHWC, BF16, BF16, ImageToColumn>>>&
        instances);

void add_device_image_to_column_ndhwc_3d_f16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, GNDHWC, F16, F16, ImageToColumn>>>&
        instances);

void add_device_image_to_column_ndhwc_3d_f32_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, GNDHWC, F32, F32, ImageToColumn>>>&
        instances);

void add_device_image_to_column_ndhwc_3d_i8_instances(
    std::vector<
        std::unique_ptr<DeviceConvTensorRearrange<3, GNDHWC, int8_t, int8_t, ImageToColumn>>>&
        instances);

// Column to Image
// nhwc, 1d
void add_device_column_to_image_nwc_1d_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, BF16, BF16, ColumnToImage>>>&
        instances);

void add_device_column_to_image_nwc_1d_f16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, F16, F16, ColumnToImage>>>&
        instances);

void add_device_column_to_image_nwc_1d_f32_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, F32, F32, ColumnToImage>>>&
        instances);

void add_device_column_to_image_nwc_1d_i8_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<1, GNWC, int8_t, int8_t, ColumnToImage>>>&
        instances);
// nhwc, 2d
void add_device_column_to_image_nhwc_2d_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<2, GNHWC, BF16, BF16, ColumnToImage>>>&
        instances);

void add_device_column_to_image_nhwc_2d_f16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<2, GNHWC, F16, F16, ColumnToImage>>>&
        instances);

void add_device_column_to_image_nhwc_2d_f32_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<2, GNHWC, F32, F32, ColumnToImage>>>&
        instances);

void add_device_column_to_image_nhwc_2d_i8_instances(
    std::vector<
        std::unique_ptr<DeviceConvTensorRearrange<2, GNHWC, int8_t, int8_t, ColumnToImage>>>&
        instances);
// nhwc, 3d
void add_device_column_to_image_ndhwc_3d_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, GNDHWC, BF16, BF16, ColumnToImage>>>&
        instances);

void add_device_column_to_image_ndhwc_3d_f16_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, GNDHWC, F16, F16, ColumnToImage>>>&
        instances);

void add_device_column_to_image_ndhwc_3d_f32_instances(
    std::vector<std::unique_ptr<DeviceConvTensorRearrange<3, GNDHWC, F32, F32, ColumnToImage>>>&
        instances);

void add_device_column_to_image_ndhwc_3d_i8_instances(
    std::vector<
        std::unique_ptr<DeviceConvTensorRearrange<3, GNDHWC, int8_t, int8_t, ColumnToImage>>>&
        instances);

template <ck::index_t NumDimSpatial,
          typename ImageLayout,
          typename InDataType,
          typename OutDataType,
          typename ConvTensorRearrangeOp>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceConvTensorRearrange<NumDimSpatial,
                                                            ImageLayout,
                                                            InDataType,
                                                            OutDataType,
                                                            ConvTensorRearrangeOp>>
{
    using DeviceOp = DeviceConvTensorRearrange<NumDimSpatial,
                                               ImageLayout,
                                               InDataType,
                                               OutDataType,
                                               ConvTensorRearrangeOp>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ConvTensorRearrangeOp, ImageToColumn>)
        {
            if constexpr(NumDimSpatial == 1 && is_same_v<ImageLayout, GNWC>)
            {
                if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
                {
                    add_device_image_to_column_nwc_1d_f32_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
                {
                    add_device_image_to_column_nwc_1d_f16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
                    add_device_image_to_column_nwc_1d_bf16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, int8_t> && is_same_v<OutDataType, int8_t>)
                {
                    add_device_image_to_column_nwc_1d_i8_instances(op_ptrs);
                }
            }
            else if constexpr(NumDimSpatial == 2 && is_same_v<ImageLayout, GNHWC>)
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
            else if constexpr(NumDimSpatial == 3 && is_same_v<ImageLayout, GNDHWC>)
            {
                if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
                {
                    add_device_image_to_column_ndhwc_3d_f32_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
                {
                    add_device_image_to_column_ndhwc_3d_f16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
                    add_device_image_to_column_ndhwc_3d_bf16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, int8_t> && is_same_v<OutDataType, int8_t>)
                {
                    add_device_image_to_column_ndhwc_3d_i8_instances(op_ptrs);
                }
            }
        }
        else if constexpr(is_same_v<ConvTensorRearrangeOp, ColumnToImage>)
        {
            if constexpr(NumDimSpatial == 1 && is_same_v<ImageLayout, GNWC>)
            {
                if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
                {
                    add_device_column_to_image_nwc_1d_f32_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
                {
                    add_device_column_to_image_nwc_1d_f16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
                    add_device_column_to_image_nwc_1d_bf16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, int8_t> && is_same_v<OutDataType, int8_t>)
                {
                    add_device_column_to_image_nwc_1d_i8_instances(op_ptrs);
                }
            }
            else if constexpr(NumDimSpatial == 2 && is_same_v<ImageLayout, GNHWC>)
            {
                if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
                {
                    add_device_column_to_image_nhwc_2d_f32_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
                {
                    add_device_column_to_image_nhwc_2d_f16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
                    add_device_column_to_image_nhwc_2d_bf16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, int8_t> && is_same_v<OutDataType, int8_t>)
                {
                    add_device_column_to_image_nhwc_2d_i8_instances(op_ptrs);
                }
            }
            else if constexpr(NumDimSpatial == 3 && is_same_v<ImageLayout, GNDHWC>)
            {
                if constexpr(is_same_v<InDataType, float> && is_same_v<OutDataType, float>)
                {
                    add_device_column_to_image_ndhwc_3d_f32_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, half_t> && is_same_v<OutDataType, half_t>)
                {
                    add_device_column_to_image_ndhwc_3d_f16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                                  is_same_v<OutDataType, ck::bhalf_t>)
                {
                    add_device_column_to_image_ndhwc_3d_bf16_instances(op_ptrs);
                }
                else if constexpr(is_same_v<InDataType, int8_t> && is_same_v<OutDataType, int8_t>)
                {
                    add_device_column_to_image_ndhwc_3d_i8_instances(op_ptrs);
                }
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
