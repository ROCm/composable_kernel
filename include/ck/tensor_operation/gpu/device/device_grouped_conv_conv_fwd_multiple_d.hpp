// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Convolution Forward:
//   input : input image A0[G, N, C0, Hi0, Wi0]
//   input : weight B0[G, K0, C0, Y0, X0]
//   input : residual D00[G, N, K0, Ho0, Wo0], D01[G, N, K0, Ho0, Wo0], ...
//   input : weight B1[G, K1, C1, 1, 1], (1x1 only)
//   input : residual D10[G, N, K1, Ho1, Wo1], D11[G, N, K1, Ho1, Wo1], ...
//   output : output image E1[G, N, K1, Ho1, Wo1]
//   C0 = a0_op(A0) * b0_op(B0)
//   E0 = cde0_op(C0, D00, D01, ...)
//   C1 = E0 * b1_op(B1)
//   E1 = cde1_op(C1, D10, D11, ...)
template <index_t NDimSpatial,
          typename A0Layout,
          typename B0Layout,
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename E1Layout,
          typename A0DataType,
          typename B0DataType,
          typename D0sDataType,
          typename B1DataType,
          typename D1sDataType,
          typename E1DataType,
          typename A0ElementwiseOperation,
          typename B0ElementwiseOperation,
          typename CDE0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation>
struct DeviceGroupedConvConvFwdMultipleD : public BaseOperator
{
    static constexpr index_t NumD0Tensor = D0sDataType::Size();
    static constexpr index_t NumD1Tensor = D1sDataType::Size();

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a0,
        const void* p_b0,
        const std::array<const void*, NumD0Tensor>& p_d0s,
        const void* p_b1,
        const std::array<const void*, NumD1Tensor>& p_d1s,
        void* p_e1,
        const std::array<index_t, NDimSpatial + 3>& a0_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a0_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b0_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b0_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumD0Tensor>& d0s_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumD0Tensor>& d0s_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& b1_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b1_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumD1Tensor>& d1s_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumD1Tensor>& d1s_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e1_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e1_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv0_filter_strides,
        const std::array<index_t, NDimSpatial>& conv0_filter_dilations,
        const std::array<index_t, NDimSpatial>& input0_left_pads,
        const std::array<index_t, NDimSpatial>& input0_right_pads,
        const std::array<index_t, NDimSpatial>& conv1_filter_strides,
        const std::array<index_t, NDimSpatial>& conv1_filter_dilations,
        const std::array<index_t, NDimSpatial>& input1_left_pads,
        const std::array<index_t, NDimSpatial>& input1_right_pads,
        const A0ElementwiseOperation& a0_element_op,
        const B0ElementwiseOperation& b0_element_op,
        const CDE0ElementwiseOperation& cde0_element_op,
        const B1ElementwiseOperation& b1_element_op,
        const CDE1ElementwiseOperation& cde1_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
