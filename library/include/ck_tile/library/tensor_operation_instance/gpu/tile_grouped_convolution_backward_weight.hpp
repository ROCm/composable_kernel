// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

// Add these missing includes:
// #include "ck_tile/core/tensor_layout.hpp"
// #include "ck_tile/ops/common/tensor_layout.hpp"
// #include "ck_tile/ops/common/element_wise_operation.hpp"

#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp"

namespace ck_tile {
namespace ops {

template <typename DeviceOp>
struct DeviceOperationInstanceFactory;

template <ck_tile::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<ck_tile::GroupedConvolutionBackwardWeightInvoker<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    OutLayout,
    InDataType,
    WeiDataType,
    OutDataType,
    ck_tile::element_wise::PassThrough,
    ck_tile::element_wise::PassThrough,
    ck_tile::element_wise::PassThrough,
    ComputeTypeA,
    ComputeTypeB>>
{
    using DeviceOp = GroupedConvolutionBackwardWeightInvoker<NumDimSpatial,
                                                InLayout,
                                                WeiLayout,
                                                OutLayout,
                                                InDataType,
                                                WeiDataType,
                                                OutDataType,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                ComputeTypeA,
                                                ComputeTypeB>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

//         if constexpr(NumDimSpatial == 2)
//         {
//             if constexpr(is_same_v<InLayout, GNHWC> && is_same_v<WeiLayout, GKYXC> &&
//                          is_same_v<OutLayout, GNHWK>)
//             {
// #ifdef CK_ENABLE_FP32
//                 if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
//                              is_same_v<ComputeTypeB, float>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_FP16
//                 if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
//                              is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
//                              is_same_v<ComputeTypeB, half_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_BF16
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
// #endif
//             }
//             if constexpr(is_same_v<InLayout, NHWGC> && is_same_v<WeiLayout, GKYXC> &&
//                          is_same_v<OutLayout, NHWGK>)
//             {
// #ifdef CK_ENABLE_FP32
//                 if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, float>)
//                 {
//                     static_assert(is_same_v<ComputeTypeA, ComputeTypeB>,
//                                   "Error: ComputeTypeA and ComputeTypeB should be the same");
//                     if constexpr(is_same_v<ComputeTypeA, float>)
//                     {
                        
//                     }
//                 }
// #endif
// #ifdef CK_ENABLE_FP16
//                 if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
//                              is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
//                              is_same_v<ComputeTypeB, half_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_BF16
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
//                              is_same_v<WeiDataType, ck::bhalf_t> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
// #endif
//             }
//             if constexpr(is_same_v<InLayout, NGCHW> && is_same_v<WeiLayout, GKCYX> &&
//                          is_same_v<OutLayout, NGKHW>)
//             {
// #ifdef CK_ENABLE_FP16
//                 if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
//                              is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
//                              is_same_v<ComputeTypeB, half_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_BF16
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
//                              is_same_v<WeiDataType, ck::bhalf_t> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_FP32
//                 if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
//                              is_same_v<ComputeTypeB, float>)
//                 {
//                 }
// #endif
//             }
//             if constexpr(is_same_v<InLayout, NGCHW> && is_same_v<WeiLayout, GKYXC> &&
//                          is_same_v<OutLayout, NGKHW>)
//             {
// #ifdef CK_ENABLE_FP16
//                 if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
//                              is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
//                              is_same_v<ComputeTypeB, half_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_BF16
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
//                              is_same_v<WeiDataType, ck::bhalf_t> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_FP32
//                 if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
//                              is_same_v<ComputeTypeB, float>)
//                 {
//                 }
// #endif
//             }
//         }

        return op_ptrs;
    }
};

} // namespace ops
} // namespace ck_tile
