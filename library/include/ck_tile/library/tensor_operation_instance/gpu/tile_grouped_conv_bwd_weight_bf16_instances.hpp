// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_bwd_weight_invoker.hpp"

namespace ck_tile {
namespace ops {

using BF16 = ck_tile::bfloat16_t;
using F16  = ck_tile::half_t;
using F32  = float;

template <ck_tile::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout>
using tile_grouped_conv_bwd_weight_bf16_instances = std::tuple<
// clang-format off
    //#####################################|       Num|  InLayout| WeiLayout| OutLayout| InData| WeiData| OutData|           In|         Wei|         Out|  K-block|  M-tile| N-tile | K-tile | M-warp| N-warp| K-warp| M-warp| N-warp| K-warp| Vector| Vector| Vector|  Double|                         GEMM|
    //#####################################|       Dim|          |          |          |   Type|    Type|    Type|  Elementwise| Elementwise| Elementwise|      per|        |        |        |       |       |       |   tile|   tile|   tile|   size|   size|   size|    smem|                     pipeline|
    //#####################################|   Spatial|          |          |          |       |        |        |    Operation|   Operation|   Operation|       CU|        |        |        |       |       |       |   size|   size|   size|      A|      B|      C|  buffer|                      version|
    GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        2,      64,      64,      64,      2,      2,      1,     32,     32,     16,      2,     4,       2,   false,  CK_TILE_PIPELINE_COMPUTE_V3>,
    GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        2,      64,      64,      64,      2,      2,      1,     32,     32,      8,      2,     4,       2,   false,  CK_TILE_PIPELINE_COMPUTE_V3>
    // GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        2,      64,      64,      64,      2,      2,      1,     16,     16,     32,      2,     4,       2,   false,  CK_TILE_PIPELINE_COMPUTE_V3>,
    // GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        2,       8,     128,      64,      2,      2,      1,      4,     64,     16,      2,     4,       2,   false,  CK_TILE_PIPELINE_COMPUTE_V3>,
    // GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        2,     128,       8,      64,      2,      2,      1,     64,      4,     16,      2,     4,       2,   false,  CK_TILE_PIPELINE_COMPUTE_V3>,

    // GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      1,      1,      1,     32,     32,      8,      2,     2,       2,   false,  CK_TILE_PIPELINE_COMPUTE_V3>,
    // GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      1,      1,      1,     32,     32,     16,      2,     2,       2,   false,  CK_TILE_PIPELINE_COMPUTE_V3>,
    // GroupedConvolutionBackwardWeightInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,  PassThrough, PassThrough, PassThrough,        1,      64,      64,      64,      1,      1,      1,     16,     16,     32,      2,     2,       2,   false,  CK_TILE_PIPELINE_COMPUTE_V3>
// // clang-format on
>;

} // namespace ops
} // namespace ck_tile
