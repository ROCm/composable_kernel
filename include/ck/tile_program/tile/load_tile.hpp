// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool use_inline_asm = false>
__device__ auto load_tile(const TileWindowWithStaticDistribution<BottomTensorView_,
                                                                 WindowLengths_,
                                                                 TileDistribution_,
                                                                 NumCoord>& tile_window,
                          bool_constant<use_inline_asm> = {})
{
    return tile_window.Load(bool_constant<use_inline_asm>{});
}

template <typename DstDataType_,
          typename DstTileDistribution_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord>
__device__ void load_tile(StaticDistributedTensor<DstDataType_, DstTileDistribution_>& dst_tensor,
                          const TileWindowWithStaticDistribution<BottomTensorView_,
                                                                 WindowLengths_,
                                                                 TileDistribution_,
                                                                 NumCoord>& tile_window)
{
    tile_window.LoadRaw(dst_tensor);
}

template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord>
__device__ auto async_load_tile(LdsTileWindow_&& lds_tile,
                                const TileWindowWithStaticDistribution<BottomTensorView_,
                                                                       WindowLengths_,
                                                                       TileDistribution_,
                                                                       NumCoord>& tile_window)
{
    return tile_window.AsyncLoad(lds_tile);
}

__device__ auto async_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

} // namespace tile_program
} // namespace ck
