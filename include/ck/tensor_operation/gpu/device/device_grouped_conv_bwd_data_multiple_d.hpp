// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Conv backward data multiple D:
//   input : output image A[G, N, K, Ho, Wo]
//   input : weight B[G, K, C, Y, X],
//   input : D0[G, N, K, Ho, Wo], D1[G, N, K, Ho, Wo], ...
//   output : input image E[G, N, C, Hi, Wi],
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGroupedConvBwdDataMultipleD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(NumDTensor == DsLayout::Size(), "wrong! Inconsistent NumDTensor");

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,                                                 // output image
        const void* p_b,                                                 // weight
        const std::array<const void*, NumDTensor>& p_ds,                 // bias
        void* p_e,                                                       // input image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_k_wos_lengths, // bias
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_k_wos_strides,                                        // bias
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                         ALayout,
                                         BLayout,
                                         Tuple<>,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         Tuple<>,
                                         EDataType,
                                         AElementwiseOperation,
                                         BElementwiseOperation,
                                         CDEElementwiseOperation>
    : public DeviceGroupedConvBwdData<NDimSpatial,
                                      ELayout,
                                      BLayout,
                                      ALayout,
                                      EDataType,
                                      BDataType,
                                      ADataType,
                                      CDEElementwiseOperation,
                                      BElementwiseOperation,
                                      AElementwiseOperation>
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,                                                 // output image
        const void* p_b,                                                 // weight
        const std::array<const void*, 0>&,                               // bias
        void* p_e,                                                       // input image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
        const std::array<std::array<index_t, NDimSpatial + 3>, 0>&,      // bias
        const std::array<std::array<index_t, NDimSpatial + 3>, 0>&,      // bias
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op) = 0;

    std::unique_ptr<BaseArgument>
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
                        const CDEElementwiseOperation& input_element_op,
                        const BElementwiseOperation& weight_element_op,
                        const AElementwiseOperation& output_element_op) override final
    {
        return MakeArgumentPointer(p_output,
                                   p_weight,
                                   std::array<const void*, 0>{},
                                   p_input,
                                   output_g_n_k_wos_lengths,
                                   output_g_n_k_wos_strides,
                                   weight_g_k_c_xs_lengths,
                                   weight_g_k_c_xs_strides,
                                   std::array<std::array<index_t, NDimSpatial + 3>, 0>{},
                                   std::array<std::array<index_t, NDimSpatial + 3>, 0>{},
                                   input_g_n_c_wis_lengths,
                                   input_g_n_c_wis_strides,
                                   conv_filter_strides,
                                   conv_filter_dilations,
                                   input_left_pads,
                                   input_right_pads,
                                   output_element_op,
                                   weight_element_op,
                                   input_element_op);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
