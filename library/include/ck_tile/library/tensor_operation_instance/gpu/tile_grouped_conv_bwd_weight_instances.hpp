// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>

#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_bwd_weight_invoker.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_bwd_weight_fp16_instances.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_bwd_weight_bf16_instances.hpp"

namespace ck_tile {
namespace ops {

using BF16 = ck_tile::bfloat16_t;
using F16  = ck_tile::half_t;
using F32  = float;

using DeviceOp2DF16 = GroupedConvolutionBackwardWeightBaseInvoker<2,
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

using DeviceOp2DBF16 = GroupedConvolutionBackwardWeightBaseInvoker<2,
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

void add_grouped_conv2d_bwd_weight_f16_instances(std::vector<std::unique_ptr<DeviceOp2DF16>>& instances)
{
    add_device_operation_instances(instances,
                                   tile_grouped_conv_bwd_weight_f16_instances<
                                       2,
                                       NHWGC,
                                       GKYXC,
                                       NHWGK>{});
}

void add_grouped_conv2d_bwd_weight_bf16_instances(std::vector<std::unique_ptr<DeviceOp2DBF16>>& instances)
{
    add_device_operation_instances(instances,
                                   tile_grouped_conv_bwd_weight_bf16_instances<
                                       2,
                                       NHWGC,
                                       GKYXC,
                                       NHWGK>{});
}

} // namespace ops
} // namespace ck_tile
