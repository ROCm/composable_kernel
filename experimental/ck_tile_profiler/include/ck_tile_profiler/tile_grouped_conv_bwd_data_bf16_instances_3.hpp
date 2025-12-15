// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "tile_grouped_conv_instance_factory.hpp"
#include "tile_grouped_conv_bwd_data_invoker.hpp"

namespace ck_tile {
namespace ops {

using BF16 = ck_tile::bfloat16_t;

using DeviceOp2DBF16 = GroupedConvolutionBackwardDataBaseInvoker<2,
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

template <ck_tile::index_t NDimSpatial, typename ALayout, typename BLayout, typename ELayout>
using tile_grouped_conv_bwd_data_bf16_instances_3 = std::tuple<
    // clang-format off
    //###################################|    Num|  InLayout| WeiLayout|   OutLayout| InData|WeiData|OutData|           In|         Wei|         Out|                  Conv|K-block|  M-tile| N-tile | K-tile | M-warp| N-warp| K-warp| M-warp| N-warp| K-warp| Vector| Vector| Vector|    Double|                         GEMM|
    //###################################|    Dim|          |          |            |   Type|   Type|   Type|  Elementwise| Elementwise| Elementwise|                  Spec|    per|        |        |        |       |       |       |   tile|   tile|   tile|   size|   size|   size|      smem|                     pipeline|
    //###################################|Spatial|          |          |            |       |       |       |    Operation|   Operation|   Operation|                      |     CU|        |        |        |       |       |       |   size|   size|   size|      A|      B|      C|    buffer|                      version|                   
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     16,      64,     32,     1,    1,    1,16,     16,      32,      8,    8,4, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     16,      64,     32,     1,    1,    1,16,     16,      32,      8,    8,4, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     16,      64,     32,     1,    1,    1,16,     16,      32,      8,    8,4, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     64,      128,     32,     2,    2,    1,32,     32,      16,      1,    8,8, true, CK_TILE_PIPELINE_COMPUTE_V4>, // ta
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      32,     32,     2,    1,    1,32,     32,      16,      8,    1,1, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      32,     32,     2,    1,    1,32,     32,      16,      8,    1,1, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      256,     32,     2,    2,    1,32,     32,      16,      8,    4,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      256,     32,     2,    2,    1,32,     32,      16,      8,    4,8, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      256,     32,     2,    2,    1,32,     32,      16,      8,    4,8, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     256,      128,     32,     2,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     256,      128,     32,     2,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     256,      128,     32,     2,    2,    1,32,     32,      16,      8,    8,8, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      256,     32,     2,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      256,     32,     2,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      256,     32,     2,    2,    1,32,     32,      16,      8,    8,8, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      128,     32,     1,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      128,     32,     1,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      128,     32,     1,    2,    1,32,     32,      16,      8,    8,8, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      128,     32,     2,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      128,     32,     2,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      128,     32,     2,    2,    1,32,     32,      16,      8,    8,8, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      64,     32,     2,    1,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      64,     32,     2,    1,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      64,     32,     2,    1,    1,32,     32,      16,      8,    8,8, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     64,      128,     32,     1,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     64,      128,     32,     1,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     64,      128,     32,     1,    2,    1,32,     32,      16,      8,    8,8, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     64,      64,     32,     1,    1,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     64,      64,     32,     1,    1,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_COMPUTE_V3>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     64,      64,     32,     1,    1,    1,32,     32,      16,      8,    8,8, true, CK_TILE_PIPELINE_COMPUTE_V4>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      64,     32,     2,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_MEMORY>,
// GroupedConvolutionBackwardDataInvoker<NDimSpatial,   ALayout,   BLayout,     ELayout,   BF16,   BF16,   BF16,  PassThrough, PassThrough, PassThrough, ConvolutionSpecialization::Default,      1,     128,      64,     32,     2,    2,    1,32,     32,      16,      8,    8,8, false, CK_TILE_PIPELINE_COMPUTE_V3>
    // clang-format on
    >;

} // namespace ops
} // namespace ck_tile
