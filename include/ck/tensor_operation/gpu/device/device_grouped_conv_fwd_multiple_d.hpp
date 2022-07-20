// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Grouped Convolution Forword
//   input : input image A[G, C, N, Hi, Wi],
//   input : weight B[G, K, C, Y, X],
//   input : D0[G, N, K, Ho, Wo], D1[G, N, K, Ho, Wo], ...
//   output : output image E[G, N, K, Ho, Wo]
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
struct DeviceGroupedConvFwdMultipleD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const std::vector<ck::index_t>& a_g_n_c_wis_lengths,
                        const std::vector<ck::index_t>& a_g_n_c_wis_strides,
                        const std::vector<ck::index_t>& b_g_k_c_xs_lengths,
                        const std::vector<ck::index_t>& b_g_k_c_xs_strides,
                        std::array<std::vector<ck::index_t>, NumDTensor> ds_g_n_k_wos_lengths;
                        std::array<std::vector<ck::index_t>, NumDTensor> ds_g_n_k_wos_strides;
                        const std::vector<ck::index_t>& e_g_n_k_wos_lengths,
                        const std::vector<ck::index_t>& e_g_n_k_wos_strides,
                        const std::vector<ck::index_t>& conv_filter_strides,
                        const std::vector<ck::index_t>& conv_filter_dilations,
                        const std::vector<ck::index_t>& input_left_pads,
                        const std::vector<ck::index_t>& input_right_pads,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CDEElementwiseOperation& cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
