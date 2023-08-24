// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Image to column:
//   input : input image [N, Di, Hi, Wi, C],
//   output : output image [N * Do * Ho * Wo, Z *  Y * X * C]
template <index_t NDimSpatial,
          typename InputLayout,
          typename InputDataType,
          typename OutputDataType>
struct DeviceImageToColumn : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in, // input image
                        void* p_out,      // output image
                        const ck::index_t N,
                        const ck::index_t C,
                        const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                        const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                        const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                        const std::array<index_t, NDimSpatial + 3>& input_g_n_c_wis_strides,
                        const std::array<index_t, 2>& output_m_k_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
