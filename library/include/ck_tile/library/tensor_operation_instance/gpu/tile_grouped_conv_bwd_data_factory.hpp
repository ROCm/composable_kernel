// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>
#include <type_traits>

#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_bwd_data_invoker.hpp"

namespace ck_tile {
namespace ops {

using DeviceOp2DF16 = GroupedConvolutionBackwardDataBaseInvoker<2,
                                                NHWGC,
                                                GKYXC,
                                                NHWGK,
                                                ck_tile::half_t,
                                                ck_tile::half_t,
                                                ck_tile::half_t,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::half_t,
                                                ck_tile::half_t>;

using DeviceOp2DBF16 = GroupedConvolutionBackwardDataBaseInvoker<2,
                                                 NHWGC,
                                                 GKYXC,
                                                 NHWGK,
                                                 ck_tile::bfloat16_t,
                                                 ck_tile::bfloat16_t,
                                                 ck_tile::bfloat16_t,
                                                 ck_tile::element_wise::PassThrough,
                                                 ck_tile::element_wise::PassThrough,
                                                 ck_tile::element_wise::PassThrough,
                                                 ck_tile::bfloat16_t,
                                                 ck_tile::bfloat16_t>;

using DeviceOp2DF32 = GroupedConvolutionBackwardDataBaseInvoker<2,
                                                NHWGC,
                                                GKYXC,
                                                NHWGK,
                                                float,
                                                float,
                                                float,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                float,
                                                float>;

// Forward declarations for instance factory functions
// void add_grouped_conv2d_bwd_weight_f16_instances(std::vector<std::unique_ptr<DeviceOp2DF16>>& instances);
void add_grouped_conv2d_bwd_data_bf16_instances(std::vector<std::unique_ptr<DeviceOp2DBF16>>& instances);
// void add_grouped_conv2d_bwd_weight_bf16_instances_opt(std::vector<std::unique_ptr<DeviceOp2DBF16>>& instances);

template <ck_tile::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<GroupedConvolutionBackwardDataBaseInvoker<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    OutLayout,
    InDataType,
    WeiDataType,
    OutDataType,
    PassThrough,
    PassThrough,
    PassThrough,
    ComputeTypeA,
    ComputeTypeB>>
{
    using DeviceOp = GroupedConvolutionBackwardDataBaseInvoker<NumDimSpatial,
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

        if constexpr(NumDimSpatial == 2)
        {
            if constexpr(std::is_same_v<InLayout, NHWGC> && std::is_same_v<WeiLayout, GKYXC> &&
                         std::is_same_v<OutLayout, NHWGK>)
            {
                if constexpr(std::is_same_v<InDataType, ck_tile::half_t> && 
                             std::is_same_v<WeiDataType, ck_tile::half_t> &&
                             std::is_same_v<OutDataType, ck_tile::half_t> && 
                             std::is_same_v<ComputeTypeA, ck_tile::half_t> &&
                             std::is_same_v<ComputeTypeB, ck_tile::half_t>)
                {
                    // add_grouped_conv2d_bwd_weight_f16_instances(op_ptrs);
                }
                if constexpr(std::is_same_v<InDataType, ck_tile::bfloat16_t> &&
                             std::is_same_v<WeiDataType, ck_tile::bfloat16_t> &&
                             std::is_same_v<OutDataType, ck_tile::bfloat16_t> &&
                             std::is_same_v<ComputeTypeA, ck_tile::bfloat16_t> &&
                             std::is_same_v<ComputeTypeB, ck_tile::bfloat16_t>)
                {
                    add_grouped_conv2d_bwd_data_bf16_instances(op_ptrs);
                    // add_grouped_conv2d_bwd_weight_bf16_instances_opt(op_ptrs);
                }
            }
        }

        return op_ptrs;
    }
};

} // namespace ops
} // namespace ck_tile
