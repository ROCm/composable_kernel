// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>
#include <type_traits>

#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_fwd_invoker.hpp"

namespace ck_tile {
namespace ops {

using DeviceOpFwd2DBF16 = GroupedConvolutionForwardBaseInvoker<2,
                                                 NHWGC,
                                                 GKYXC,
                                                 NHWGK,
                                                 BF16,
                                                 BF16,
                                                 BF16,
                                                 PassThrough,
                                                 PassThrough,
                                                 PassThrough,
                                                 BF16,
                                                 BF16>;

using DeviceOpFwd2DF16 = GroupedConvolutionForwardBaseInvoker<2,
                                                NHWGC,
                                                GKYXC,
                                                NHWGK,
                                                F16,
                                                F16,
                                                F16,
                                                PassThrough,
                                                PassThrough,
                                                PassThrough,
                                                F16,
                                                F16>;

using DeviceOpFwd2DINT8 = GroupedConvolutionForwardBaseInvoker<2,
                                                NHWGC,
                                                GKYXC,
                                                NHWGK,
                                                INT8,
                                                INT8,
                                                INT8,
                                                PassThrough,
                                                PassThrough,
                                                PassThrough,
                                                INT8,
                                                INT8>;

// BF16 instances 
void add_grouped_conv2d_fwd_bf16_instances(std::vector<std::unique_ptr<DeviceOpFwd2DBF16>>& instances);
void add_grouped_conv2d_fwd_bf16_instances_2(std::vector<std::unique_ptr<DeviceOpFwd2DBF16>>& instances);
void add_grouped_conv2d_fwd_bf16_instances_3(std::vector<std::unique_ptr<DeviceOpFwd2DBF16>>& instances);
void add_grouped_conv2d_fwd_bf16_instances_4(std::vector<std::unique_ptr<DeviceOpFwd2DBF16>>& instances);
void add_grouped_conv2d_fwd_bf16_instances_5(std::vector<std::unique_ptr<DeviceOpFwd2DBF16>>& instances);
void add_grouped_conv2d_fwd_bf16_instances_6(std::vector<std::unique_ptr<DeviceOpFwd2DBF16>>& instances);

// FP16 instances
void add_grouped_conv2d_fwd_f16_instances(std::vector<std::unique_ptr<DeviceOpFwd2DF16>>& instances);

// INT8 instances
void add_grouped_conv2d_fwd_i8_instances(std::vector<std::unique_ptr<DeviceOpFwd2DINT8>>& instances);

template <ck_tile::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<GroupedConvolutionForwardBaseInvoker<
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
    using DeviceOp = GroupedConvolutionForwardBaseInvoker<NumDimSpatial,
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
                    add_grouped_conv2d_fwd_f16_instances(op_ptrs);
                }
                else if constexpr(std::is_same_v<InDataType, ck_tile::bfloat16_t> &&
                             std::is_same_v<WeiDataType, ck_tile::bfloat16_t> &&
                             std::is_same_v<OutDataType, ck_tile::bfloat16_t> &&
                             std::is_same_v<ComputeTypeA, ck_tile::bfloat16_t> &&
                             std::is_same_v<ComputeTypeB, ck_tile::bfloat16_t>)
                {
                    add_grouped_conv2d_fwd_bf16_instances(op_ptrs);
                    add_grouped_conv2d_fwd_bf16_instances_2(op_ptrs);
                    add_grouped_conv2d_fwd_bf16_instances_3(op_ptrs);
                    add_grouped_conv2d_fwd_bf16_instances_4(op_ptrs);
                    add_grouped_conv2d_fwd_bf16_instances_5(op_ptrs);
                    add_grouped_conv2d_fwd_bf16_instances_6(op_ptrs);
                }
                else if constexpr(std::is_same_v<InDataType, ck_tile::int8_t> && 
                             std::is_same_v<WeiDataType, ck_tile::int8_t> &&
                             std::is_same_v<OutDataType, ck_tile::int8_t> && 
                             std::is_same_v<ComputeTypeA, ck_tile::int8_t> &&
                             std::is_same_v<ComputeTypeB, ck_tile::int8_t>)
                {
                    add_grouped_conv2d_fwd_int8_instances(op_ptrs);
                }
                else
                {
                    std::cout << "Unsupported data type combination for GroupedConv2dFwd\n";
                }
            }
            else
            {
                std::cout << "Unsupported layout combination for GroupedConv2dFwd\n";
            }
        }

        return op_ptrs;
    }
};

} // namespace ops
} // namespace ck_tile
