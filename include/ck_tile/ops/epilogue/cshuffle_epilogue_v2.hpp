// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// this epilogue just store out a M*N matrix, row major

template <typename AccDataType_,
          typename ODataType_,
          index_t kBlockSize_,
          index_t kM_,
          index_t kN_,
          bool kPadM_,
          bool kPadN_>
struct CShuffleEpilogueV2Problem
{
    using AccDataType                 = remove_cvref_t<AccDataType_>;
    using ODataType                   = remove_cvref_t<ODataType_>;
    // static constexpr bool UseRawStore = UseRawStore_;
    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr index_t kMPerBlock = kM_;
    static constexpr index_t kNPerBlock = kN_;
    static constexpr bool kPadM       = kPadM_;
    static constexpr bool kPadN       = kPadN_;
};


template <typename Problem>
CK_TILE_HOST_DEVICE static constexpr auto MakeOLdsBlockDescriptor()
{
    constexpr index_t kMPerBlock = Problem::kMPerBlock;
    constexpr index_t kNPerBlock = Problem::kNPerBlock;

    return make_naive_tensor_descriptor(
        make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
        make_tuple(number<kNPerBlock>{}, number<1>{}));
}


CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 65536; }
template <typename Problem>
CK_TILE_HOST_DEVICE static constexpr auto MakeODramTileDistribution()
{
    constexpr index_t kMPerBlock = Problem::kMPerBlock;
    constexpr index_t kNPerBlock = Problem::kNPerBlock;
    constexpr index_t BlockSize  = Problem::kBlockSize;
    constexpr index_t WaveSize = get_warp_size();
    using ODataType = remove_cvref_t<typename Problem::ODataType>;
    
    // using OLayout   = remove_cvref_t<typename Problem::OLayout>;
    // if constexpr(std::is_same_v<OLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
    // {
    //     static_assert(0, "not impl");
    // }
    constexpr index_t N2 = 8;
    constexpr index_t N1 = min(kNPerBlock / N2, WaveSize);
    constexpr index_t N0 = integer_divide_ceil(kNPerBlock / N2,  WaveSize);

    constexpr index_t M2 = integer_divide_ceil(WaveSize, kNPerBlock / N2);
    constexpr index_t M1 = BlockSize / WaveSize;
    constexpr index_t M0 = integer_divide_ceil(kMPerBlock, M1 * M2);

    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<1>,
                                    tuple<sequence<M0, M1, M2>, sequence<N0, N1, N2>>,
                                    tuple<sequence<1>, sequence<1, 2>>,
                                    tuple<sequence<1>, sequence<2, 1>>,
                                    sequence<1, 2, 2>,
                                    sequence<0, 0, 2>>{});
}

    
template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogueV2
{
    using Problem                     = remove_cvref_t<Problem_>;
    using AccDataType                 = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                   = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM       = Problem::kPadM;
    static constexpr bool kPadN       = Problem::kPadN;
    static constexpr bool UseRawStore = Problem::UseRawStore;
    static constexpr index_t kMPerBlock = Problem::kMPerBlock;
    // static constexpr bool kMPerBlock      = 64;
    static constexpr index_t kNPerBlock      = Problem::kNPerBlock;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 65536;}//kMPerBlock * kNPerBlock * sizeof(ODataType); }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?
    template <typename ODramWindowTmp, typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile, void *p_smem)
    {
        block_sync_lds();
        auto o_lds_tile = cast_tile<ODataType>(o_acc_tile);
        constexpr auto o_lds_block_desc = MakeOLdsBlockDescriptor<Problem>();
        auto o_lds_block = make_tensor_view<address_space_enum::lds>(static_cast<ODataType*>(p_smem), o_lds_block_desc);
        auto o_lds_window0 = make_tile_window(o_lds_block, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {0, 0});

        store_tile(o_lds_window0, o_lds_tile);
        block_sync_lds();
        auto o_dram_distri = MakeODramTileDistribution<Problem>();
        auto o_dram_tile = load_tile(make_tile_window(o_lds_window0, o_dram_distri));
        store_tile(o_dram_window_tmp, o_dram_tile);
        block_sync_lds();
    }
};
} // namespace ck_tile
