// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_fwd_invoker.hpp"

namespace ck_tile {
namespace ops {

using F16 = ck_tile::half_t;

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

template <ck_tile::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout>
using tile_grouped_conv_fwd_fp16_instances = std::tuple<
// clang-format off
//##############################|       Num|  InLayout| WeiLayout| OutLayout| InData| WeiData| OutData|           In|         Wei|         Out|                                             Conv|    K-block|  M-tile| N-tile | K-tile | M-warp| N-warp| K-warp| M-warp| N-warp| K-warp| Vector| Vector| Vector|  Double|                     GEMM|
//##############################|       Dim|          |          |          |   Type|    Type|    Type|  Elementwise| Elementwise| Elementwise|                                             Spec|        per|        |        |        |       |       |       |   tile|   tile|   tile|   size|   size|   size|    smem|                 pipeline|
//##############################|   Spatial|          |          |          |       |        |        |    Operation|   Operation|   Operation|                                                 |         CU|        |        |        |       |       |       |   size|   size|   size|      A|      B|      C|  buffer|                  version|

// These instances do not compile on Navi4x cards
#if defined(__gfx9__)
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,    true, CK_TILE_PIPELINE_COMPUTE_V4>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Pad0,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,    true, CK_TILE_PIPELINE_COMPUTE_V4>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter3x3,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,    true, CK_TILE_PIPELINE_COMPUTE_V4>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Default,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,    true, CK_TILE_PIPELINE_COMPUTE_V4>,
#endif

GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,     F16,  PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,      64,      64,      32,      1,      1,      1,     32,     32,     16,      1,      1,      1,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,     F16,  PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Default,                       1,      64,      64,      32,      1,      1,      1,     32,     32,     16,      1,      1,      1,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,      64,      64,      32,      1,      1,      1,     32,     32,     16,      1,      1,      1,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Default,                       1,      64,      64,      32,      1,      1,      1,     32,     32,     16,      1,      1,      1,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,   false, CK_TILE_PIPELINE_COMPUTE_V3>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Pad0,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Pad0,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,   false, CK_TILE_PIPELINE_COMPUTE_V3>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter3x3,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter3x3,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,   false, CK_TILE_PIPELINE_COMPUTE_V3>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Default,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Default,          1,     128,     128,     128,      2,      2,      1,     32,     32,     16,     16,     16,     16,   false, CK_TILE_PIPELINE_COMPUTE_V3>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_COMPUTE_V3>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,    true, CK_TILE_PIPELINE_COMPUTE_V4>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Pad0,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Pad0,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_COMPUTE_V3>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Pad0,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,    true, CK_TILE_PIPELINE_COMPUTE_V4>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter3x3,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter3x3,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_COMPUTE_V3>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter3x3,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,    true, CK_TILE_PIPELINE_COMPUTE_V4>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Default,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_MEMORY>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Default,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_COMPUTE_V3>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Default,          1,     128,     128,      32,      2,      2,      1,     32,     32,     16,      8,      8,      8,    true, CK_TILE_PIPELINE_COMPUTE_V4>,
GroupedConvolutionForwardInvoker<NDimSpatial,   ALayout,   BLayout,   ELayout,    F16,     F16,    F16,   PassThrough, PassThrough, PassThrough,  ConvolutionSpecialization::Filter1x1Stride1Pad0,          1,     256,     256,      32,      4,      4,      1,     32,     32,     16,      8,      8,      8,   false, CK_TILE_PIPELINE_MEMORY>

// clang-format on
>;

} // namespace ops
} // namespace ck_tile
