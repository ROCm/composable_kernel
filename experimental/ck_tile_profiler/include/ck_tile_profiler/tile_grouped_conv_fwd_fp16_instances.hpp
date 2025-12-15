// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "tile_grouped_conv_instance_factory.hpp"
#include "tile_grouped_conv_fwd_invoker.hpp"

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

template <ck_tile::index_t NDimSpatial, typename ALayout, typename BLayout, typename ELayout>
using tile_grouped_conv_fwd_fp16_instances = std::tuple<
    // clang-format off
    //##############################|       Num|  InLayout| WeiLayout| OutLayout| InData| WeiData| OutData|           In|         Wei|         Out|      Conv|    K-block|  M-tile| N-tile | K-tile | M-warp| N-warp| K-warp| M-warp| N-warp| K-warp| Vector| Vector| Vector|  Double|                         GEMM|
    //##############################|       Dim|          |          |          |   Type|    Type|    Type|  Elementwise| Elementwise| Elementwise|      Spec|        per|        |        |        |       |       |       |   tile|   tile|   tile|   size|   size|   size|    smem|                     pipeline|
    //##############################|   Spatial|          |          |          |       |        |        |    Operation|   Operation|   Operation|          |         CU|        |        |        |       |       |       |   size|   size|   size|      A|      B|      C|  buffer|                      version|

    // clang-format on
    >;

} // namespace ops
} // namespace ck_tile
