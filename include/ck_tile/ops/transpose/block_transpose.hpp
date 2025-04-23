// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "transpose_policy.hpp"

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
          index_t kRowWarps_, // how many warps in row direction
          index_t kColWarps_, // how many warps in col direction
          index_t kRowPerBlock_,
          index_t kColPerBlock_,
          index_t kRowPerWarp_, // this is the row number per warp per iteration
          index_t kColPerWarp_> // this is the col number per warp per iteration
struct TransposePipelineProblem
{
    static_assert(kRowWarps_ * kColWarps_ * get_warp_size() == kBlockSize_,
                  "the block size is not correct!");
    using DataType                      = remove_cvref_t<DataType_>;
    using Layout                        = remove_cvref_t<Layout_>;
    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr index_t kLeadDimWarps =
        TransposeTraits<Layout, kRowWarps_, kColWarps_>::kLeadDim;
    static constexpr index_t kSecondDimWarps =
        TransposeTraits<Layout, kRowWarps_, kColWarps_>::kSecondDim;
    static constexpr index_t kLeadDimPerBlock =
        TransposeTraits<Layout, kRowPerBlock_, kColPerBlock_>::kLeadDim;
    static constexpr index_t kSecondDimPerBlock =
        TransposeTraits<Layout, kRowPerBlock_, kColPerBlock_>::kSecondDim;
    static constexpr index_t kLeadDimPerWarp =
        TransposeTraits<Layout, kRowPerWarp_, kColPerWarp_>::kLeadDim;
    static constexpr index_t kSecondDimPerWarp =
        TransposeTraits<Layout, kRowPerWarp_, kColPerWarp_>::kSecondDim;
};

template <typename Problem_, typename Policy_ = TransposePolicy>
struct BlockTranspose
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using DataType = remove_cvref_t<typename Problem::DataType>;
    using Layout   = remove_cvref_t<typename Problem::Layout>;

    static constexpr index_t kBlockSize         = Problem::kBlockSize;
    static constexpr index_t kLeadDimPerBlock   = Problem::kLeadDimPerBlock;
    static constexpr index_t kSecondDimPerBlock = Problem::kSecondDimPerBlock;
    static constexpr index_t kLeadDimPerWarp    = Problem::kLeadDimPerWarp;
    static constexpr index_t kSecondDimPerWarp  = Problem::kSecondDimPerWarp;

    static constexpr index_t kQuadrantLeadDim   = QuartTransposeTraits<DataType>::kleadDim;
    static constexpr index_t kQuadrantSecondDim = QuartTransposeTraits<DataType>::ksecondDim;

    static_assert(kLeadDimPerBlock % kLeadDimPerWarp == 0, "row per block is not correct!");
    static_assert(kSecondDimPerBlock % kSecondDimPerWarp == 0, "col per block is not correct!");

    static_assert(kLeadDimPerWarp % kQuadrantLeadDim == 0, "row per warp is not correct!");
    static_assert(kSecondDimPerWarp % kQuadrantSecondDim == 0, "col per warp is not correct!");

    static constexpr index_t kNumWarpInLeadDim   = kLeadDimPerBlock / kLeadDimPerWarp;
    static constexpr index_t kNumWarpInSecondDim = kSecondDimPerBlock / kSecondDimPerWarp;

    static constexpr index_t kLeadDimPerWarpInQuadrant   = kLeadDimPerWarp / kQuadrantLeadDim;
    static constexpr index_t kSecondDimPerWarpInQuadrant = kSecondDimPerWarp / kQuadrantSecondDim;

    // this pipeline is only designed for wave64 now
    static_assert(get_warp_size() == 64, "the warp size is not correct!");
    static_assert(kBlockSize == kNumWarpInLeadDim * kNumWarpInSecondDim * get_warp_size(),
                  "the block size is not correct!");
    //static_assert(kLeadDimPerWarpInQuadrant * kSecondDimPerWarpInQuadrant * 4 == get_warp_size(),
    //              "the warp size is not correct!");

    static constexpr index_t GetVectorSize() { return Policy::template GetVectorSize<Problem>(); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename InputTileWindow, typename OutputTileWindow>
    CK_TILE_DEVICE void operator()(const InputTileWindow& input_window,
                                   OutputTileWindow& out_window,
                                   void* __restrict__ p_smem)
    {
        auto input_tile_window =
            make_tile_window(input_window, Policy::template MakeInputDistribution<Problem>());
        auto output_tile_window =
            make_tile_window(out_window, Policy::template MakeOutputDistribution<Problem>());

        DataType* p_lds_ptr           = static_cast<DataType*>(p_smem);
        constexpr auto lds_block_desc = Policy::template MakeLdsStoreBlockDescriptor<Problem>();
        auto input_lds_block = make_tensor_view<address_space_enum::lds>(p_lds_ptr, lds_block_desc);

        constexpr auto out_lds_block_desc = Policy::template MakeLdsLoadBlockDescriptor<Problem>();
        auto output_lds_block =
            make_tensor_view<address_space_enum::lds>(p_lds_ptr, out_lds_block_desc);

        auto copy_to_lds_window =
            make_tile_window(input_lds_block,
                             make_tuple(number<kSecondDimPerBlock>{}, number<kLeadDimPerBlock>{}),
                             {0, 0});

        auto load_from_lds_window =
            make_tile_window(output_lds_block,
                             make_tuple(number<kSecondDimPerBlock>{}, number<kLeadDimPerBlock>{}),
                             {0, 0},
                             Policy::template MakeLdsLoadTileDistribution<Problem>());

        auto x = load_tile(input_tile_window);

        store_tile(copy_to_lds_window, x);
        block_sync_lds();

        auto y = load_tile_transpose(load_from_lds_window);
        store_tile(output_tile_window, y);
    }
};

} // namespace ck_tile
