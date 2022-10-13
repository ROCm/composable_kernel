// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using CPermuteNumDims_G_M_O =
    S<2, 1, 1>; // "using CLayout = Row" has been replaced by CPermuteNumDims_G_M_O

void add_device_batched_gemm_masking_scale_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<Row,
                                                                    Col,
                                                                    Row,
                                                                    CPermuteNumDims_G_M_O,
                                                                    F16,
                                                                    F16,
                                                                    F16,
                                                                    F16,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    Scale,
                                                                    PassThrough,
                                                                    PassThrough>>>& instances);

template <typename ALayout,
          typename B0Layout,
          typename B1Layout,
          typename CPermuteNumDims_G_M_Gemm1N,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute<ALayout,
                                                                      B0Layout,
                                                                      B1Layout,
                                                                      CPermuteNumDims_G_M_Gemm1N,
                                                                      ADataType,
                                                                      B0DataType,
                                                                      B1DataType,
                                                                      CDataType,
                                                                      PassThrough,
                                                                      PassThrough,
                                                                      Scale,
                                                                      PassThrough,
                                                                      PassThrough>>
{
    using DeviceOp = DeviceBatchedGemmSoftmaxGemmPermute<ALayout,
                                                         B0Layout,
                                                         B1Layout,
                                                         CPermuteNumDims_G_M_Gemm1N,
                                                         ADataType,
                                                         B0DataType,
                                                         B1DataType,
                                                         CDataType,
                                                         PassThrough,
                                                         PassThrough,
                                                         Scale,
                                                         PassThrough,
                                                         PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<B0DataType, half_t> &&
                     is_same_v<B1DataType, half_t> && is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                         is_same_v<B1Layout, Row> &&
                         is_same_v<CPermuteNumDims_G_M_Gemm1N, CPermuteNumDims_G_M_O>)
            {
                add_device_batched_gemm_masking_scale_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
                    op_ptrs);
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
