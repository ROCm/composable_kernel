// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DsLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename DsDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
struct DeviceGroupedConvBwdWeightMultipleD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsLayout::Size();

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_in_grid,
        void* p_wei_grid,
        const void* p_out_grid,
        const std::array<const void*, NumDTensor>& p_ds,
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
        InElementwiseOperation in_element_op,
        WeiElementwiseOperation wei_element_op,
        OutElementwiseOperation out_element_op,
        const ck::index_t split_k) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
