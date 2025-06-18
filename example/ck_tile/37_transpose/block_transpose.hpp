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
          index_t kRowWarps_,    // how many warps in row direction
          index_t kColWarps_,    // how many warps in col direction
          index_t kRowPerBlock_, // row number per block
          index_t kColPerBlock_, // col number per block
          index_t kRowPerXdl_,   // row number per xdl ops
          index_t kColPerXdl_>   // col number per xdl ops
struct TransposePipelineProblem
{
    static_assert(kRowWarps_ * kColWarps_ * get_warp_size() == kBlockSize_,
                  "the block size is not correct!");
    using DataType                      = remove_cvref_t<DataType_>;
    using Layout                        = remove_cvref_t<Layout_>;
    static constexpr index_t kBlockSize = kBlockSize_;
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

template <typename Problem_, typename Policy_ = TransposePolicy>
struct BlockTranspose
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using DataType = remove_cvref_t<typename Problem::DataType>;
    using Layout   = remove_cvref_t<typename Problem::Layout>;

    static constexpr index_t kBlockSize          = Problem::kBlockSize;
    static constexpr index_t kLeadSizePerBlock   = Problem::kLeadSizePerBlock;
    static constexpr index_t kSecondSizePerBlock = Problem::kSecondSizePerBlock;

    static constexpr index_t GetVectorSize() { return Policy::template GetVectorSize<Problem>(); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename InputTileWindow, typename OutputTileWindow>
    CK_TILE_DEVICE void operator()(const InputTileWindow& input_window,
                                   OutputTileWindow& output_window,
                                   void* __restrict__ p_smem)
    {
        auto input_tile_window =
            make_tile_window(input_window, Policy::template MakeInputDistribution<Problem>());
        auto output_tile_window =
            make_tile_window(output_window, Policy::template MakeOutputDistribution<Problem>());

        DataType* p_lds_ptr              = static_cast<DataType*>(p_smem);
        constexpr auto in_lds_block_desc = Policy::template MakeLdsStoreBlockDescriptor<Problem>();
        auto input_lds_block =
            make_tensor_view<address_space_enum::lds>(p_lds_ptr, in_lds_block_desc);

        constexpr auto out_lds_block_desc = Policy::template MakeLdsLoadBlockDescriptor<Problem>();
        auto output_lds_block =
            make_tensor_view<address_space_enum::lds>(p_lds_ptr, out_lds_block_desc);

        auto copy_to_lds_window =
            make_tile_window(input_lds_block,
                             make_tuple(number<kSecondSizePerBlock>{}, number<kLeadSizePerBlock>{}),
                             {0, 0});
        auto load_from_lds_window =
            make_tile_window(output_lds_block,
                             make_tuple(number<kSecondSizePerBlock>{}, number<kLeadSizePerBlock>{}),
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
