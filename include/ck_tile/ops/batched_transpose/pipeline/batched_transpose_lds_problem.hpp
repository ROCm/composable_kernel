// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename Layout_, index_t kRow, index_t kCol>
struct TransposeTraits
{
    static constexpr index_t kLeadDim   = kCol;
    static constexpr index_t kSecondDim = kRow;
};

template <index_t kRow, index_t kCol>
struct TransposeTraits<tensor_layout::gemm::ColumnMajor, kRow, kCol>
{
    static constexpr index_t kLeadDim   = kRow;
    static constexpr index_t kSecondDim = kCol;
};

// supports 2D transpose which will store to lds, then use ds_read_b*_tr_b* instruction to get the
// transposed data; Layout in TransposePipelineProblem is the original layout of the data in the
// global memory
template <typename DataType_,
          typename Layout_,
          index_t kBlockSize_,
          index_t kRowWarps_,    // how many warps in row direction
          index_t kColWarps_,    // how many warps in col direction
          index_t kRowPerBlock_, // row number per block
          index_t kColPerBlock_, // col number per block
          index_t kRowPerXdl_,   // row number per xdl ops
          index_t kColPerXdl_>   // col number per xdl ops
struct BatchedTransposeLdsProblem
{
    static_assert(kRowWarps_ * kColWarps_ * get_warp_size() == kBlockSize_,
                  "the block size is not correct!");
    using DataType                      = remove_cvref_t<DataType_>;
    using Layout                        = remove_cvref_t<Layout_>;
    static constexpr index_t kBlockSize = kBlockSize_;
    // warps per block
    static constexpr index_t kLeadNumWarps =
        TransposeTraits<Layout, kRowWarps_, kColWarps_>::kLeadDim;
    static constexpr index_t kSecondNumWarps =
        TransposeTraits<Layout, kRowWarps_, kColWarps_>::kSecondDim;

    static constexpr index_t kLeadSizePerBlock =
        TransposeTraits<Layout, kRowPerBlock_, kColPerBlock_>::kLeadDim;
    static constexpr index_t kSecondSizePerBlock =
        TransposeTraits<Layout, kRowPerBlock_, kColPerBlock_>::kSecondDim;
        
    static constexpr index_t kLeadSizePerXdl =
        TransposeTraits<Layout, kRowPerXdl_, kColPerXdl_>::kLeadDim;
    static constexpr index_t kSecondSizePerXdl =
        TransposeTraits<Layout, kRowPerXdl_, kColPerXdl_>::kSecondDim;

    static constexpr index_t kQuadrantLeadDim   = LaneGroupTransposeTraits<DataType>::kleadDim;
    static constexpr index_t kQuadrantSecondDim = LaneGroupTransposeTraits<DataType>::ksecondDim;

    static_assert(kLeadSizePerBlock % kLeadNumWarps == 0,
                  "block dim should be divided by warp dim!");
    static_assert(kSecondSizePerBlock % kSecondNumWarps == 0,
                  "block dim should be divided by warp dim!");
    // how many rows/cols implemented in one warp
    static constexpr index_t kLeadSizePerWarp   = kLeadSizePerBlock / kLeadNumWarps;
    static constexpr index_t kSecondSizePerWarp = kSecondSizePerBlock / kSecondNumWarps;

    static_assert(kLeadSizePerWarp % kLeadSizePerXdl == 0,
                  "warp dim should be divided by xdl dim!");
    static_assert(kSecondSizePerWarp % kSecondSizePerXdl == 0,
                  "warp dim should be divided by xdl dim!");

    // warp rows/cols is divided into xdl.
    static constexpr index_t kLeadXdlNumPerWarp   = kLeadSizePerWarp / kLeadSizePerXdl;
    static constexpr index_t kSecondXdlNumPerWarp = kSecondSizePerWarp / kSecondSizePerXdl;

    static_assert(kLeadSizePerXdl % kQuadrantLeadDim == 0,
                  "xdl dim should be divided by quad dim!");
    static_assert(kSecondSizePerXdl % kQuadrantSecondDim == 0,
                  "xdl dim should be divided by quad dim!");
    // xdl rows/cols is divided into quadrants.
    static constexpr index_t kQuadNumPerLeadDim   = kLeadSizePerXdl / kQuadrantLeadDim;
    static constexpr index_t kQuadNumPerSecondDim = kSecondSizePerXdl / kQuadrantSecondDim;

    static constexpr index_t kIterationsInSecondDim =
        kQuadNumPerLeadDim * kQuadNumPerSecondDim * 16 / get_warp_size();
};

}