// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

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

// supports 2D transpose which will store to lds,
// then use ds_read_b*_tr_b* instruction to get the transposed data
template <typename DataType_,
          typename BlockTile> // sequence<block_x, block_y>
struct BatchedTransposeLdsProblem
{
    static constexpr index_t kRowWarps_    = 1;
    static constexpr index_t kColWarps_    = 1;
    static constexpr index_t kBlockSize_   = get_warp_size() * kRowWarps_ * kColWarps_;
    static constexpr index_t kRowPerBlock_ = BlockTile::at(number<1>{});
    static constexpr index_t kColPerBlock_ = BlockTile::at(number<0>{});
    // TODO: name mismatch
    static constexpr index_t kRowPerXdl_ = get_warp_size();
    static constexpr index_t kColPerXdl_ = get_warp_size();

    using DataType                      = remove_cvref_t<DataType_>;
    using Layout                        = tensor_layout::gemm::RowMajor;
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

    // definitions to adapt to BatchedTransposeKernel

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;

    static constexpr auto kMPerBlock = kLeadSizePerBlock;
    static constexpr auto kNPerBlock = kSecondSizePerBlock;

    // 128-bit is the max single-instruction bandwidth for load/store
    static constexpr index_t MaxLoadStoreSize = 16;
    static constexpr auto VectorSizeInput     = kPadM ? 1 : MaxLoadStoreSize / sizeof(DataType);
    static constexpr auto VectorSizeOutput    = kPadN ? 1 : MaxLoadStoreSize / sizeof(DataType);
};

} // namespace ck_tile
