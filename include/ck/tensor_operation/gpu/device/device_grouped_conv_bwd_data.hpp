// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <ck::index_t NDimSpatial,
          typename InputLayout,
          typename WeightLayout,
          typename OutputLayout,
          typename InputDataType,
          typename WeightDataType,
          typename OutputDataType,
          typename InputElementwiseOperation,
          typename WeightElementwiseOperation,
          typename OutputElementwiseOperation>
struct DeviceGroupedConvBwdData : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(void* p_input,
                        const void* p_weight,
                        const void* p_output,
                        const std::array<index_t, NDimSpatial + 3>& input_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& input_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& weight_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& weight_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& output_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& output_g_n_k_wos_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const InputElementwiseOperation& input_element_op,
                        const WeightElementwiseOperation& weight_element_op,
                        const OutputElementwiseOperation& output_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
