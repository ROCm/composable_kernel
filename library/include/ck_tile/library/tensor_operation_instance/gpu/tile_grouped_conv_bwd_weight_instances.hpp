// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>

#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_bwd_weight_invoker.hpp"

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

template <ck_tile::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout>
using tile_grouped_conv_bwd_weight_f16_instances = std::tuple<
    // clang-format off
        //#####################################|       Num|  InLayout| WeiLayout| OutLayout| InData| WeiData| OutData|           In|         Wei|         Out|  K-block|  M-tile| N-tile | K-tile | M-warp| N-warp| K-warp| M-warp| N-warp| K-warp| Vector| Vector| Vector|  Split-K|
        //#####################################|       Dim|          |          |          |   Type|    Type|    Type|  Elementwise| Elementwise| Elementwise|      per|        |        |        |       |       |       |   tile|   tile|   tile|   size|   size|   size|       in|
        //#####################################|   Spatial|          |          |          |       |        |        |    Operation|   Operation|   Operation|       CU|        |        |        |       |       |       |   size|   size|   size|      A|      B|      C|      use|
        //#####################################|          |          |          |          |       |        |        |             |            |            |                    
        GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,     F16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      2,      2,      1,     32,     32,     16,      1,     1,       8,   false>,
        GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,     F16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      2,      2,      1,     32,     32,     16,      1,     1,       2,    true>,
        GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,     F16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      2,      2,      1,     32,     32,     16,      1,     1,       4,    true>,
        GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,     F16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      2,      2,      1,     32,     32,     16,      1,     1,       8,    true>
    // clang-format on
    >;

template <ck_tile::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout>
using tile_grouped_conv_bwd_weight_bf16_instances = std::tuple<
    // clang-format off
        //#####################################|       Num|  InLayout| WeiLayout| OutLayout| InData| WeiData| OutData|           In|         Wei|         Out|  K-block|  M-tile| N-tile | K-tile | M-warp| N-warp| K-warp| M-warp| N-warp| K-warp| Vector| Vector| Vector|  Split-K|
        //#####################################|       Dim|          |          |          |   Type|    Type|    Type|  Elementwise| Elementwise| Elementwise|      per|        |        |        |       |       |       |   tile|   tile|   tile|   size|   size|   size|       in|
        //#####################################|   Spatial|          |          |          |       |        |        |    Operation|   Operation|   Operation|       CU|        |        |        |       |       |       |   size|   size|   size|      A|      B|      C|      use|
        //#####################################|          |          |          |          |       |        |        |             |            |            |                    
        GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      2,      2,      1,     32,     32,     16,      1,     1,       8,   false>,
        GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      2,      2,      1,     32,     32,     16,      1,     1,       2,    true>,
        GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      2,      2,      1,     32,     32,     16,      1,     1,       4,    true>,
        GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      2,      2,      1,     32,     32,     16,      1,     1,       8,    true>
    // clang-format on
    >;

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
