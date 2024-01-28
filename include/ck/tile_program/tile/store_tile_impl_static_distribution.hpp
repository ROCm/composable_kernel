// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          typename DataType_>
__device__ void store_tile(TileWindowWithStaticDistribution<BottomTensorView_,
                                                            WindowLengths_,
                                                            TileDistribution_,
                                                            NumCoord>& tile_window,
                           const StaticDistributedTensor<DataType_, TileDistribution_>& dstr_tensor)
{
    tile_window.Store(dstr_tensor);
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          typename DataType_>
__device__ void
store_tile_raw(TileWindowWithStaticDistribution<BottomTensorView_,
                                                WindowLengths_,
                                                TileDistribution_,
                                                NumCoord>& tile_window,
               const StaticDistributedTensor<DataType_, TileDistribution_>& dstr_tensor)
{
    tile_window.StoreRaw(dstr_tensor);
}

} // namespace tile_program
} // namespace ck
