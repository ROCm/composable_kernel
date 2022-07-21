// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Convolution Forward:
//   input : input image A[N, Hi, Wi, C],
//   input : weight B[K, Y, X, C],
//   input : D0[N, Ho, Wo, K], D1[N, Ho, Wo, K], ...
//   output : output image E[N, Ho, Wo, K]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <index_t NDimSpatial,
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
struct DeviceConvFwdMultipleD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
        std::array<const void*, NumDTensor> p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 2>& a_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 2>& a_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 2>& b_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 2>& b_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 2>, NumDTensor>& ds_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 2>, NumDTensor>& ds_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 2>& e_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 2>& e_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
