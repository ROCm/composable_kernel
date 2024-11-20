// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

struct BlockFmhaFwdSplitKVCombinePipelineDefaultPolicy
{
    template <index_t BlockSize, index_t M, index_t N, typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeForTile()
    {
        constexpr index_t PixelsPerThread = (M * N) / BlockSize;
        static_assert(0 < PixelsPerThread);

        constexpr index_t MaxNPerThread = 16 / sizeof(DataType);
        constexpr index_t NPerThread    = min(MaxNPerThread, PixelsPerThread);

        return NPerThread;
    }

    // alignment for dram lse tile (shape=[kMaxSplits, kM0])
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentLSE()
    {
        return GetVectorSizeForTile<Problem::kBlockSize,
                                    Problem::kMaxSplits,
                                    Problem::kM0,
                                    typename Problem::LSEDataType>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kNPerBlock = Problem::kN1;

        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M2 = min(kMPerBlock / M1, get_warp_size());
        constexpr index_t N0 = get_warp_size() / M2;
        constexpr index_t N1 = kNPerBlock / N0;

        return min(N1, static_cast<index_t>(16 / sizeof(OaccDataType)));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        return GetAlignmentOacc<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return sizeof(typename Problem::LSEDataType) *
               MakeLSEaccLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    // shape=[kMaxSplits, kM0]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccDramTileDistribution()
    {
        using LSEDataType = remove_cvref_t<typename Problem::LSEDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNumWarps  = Problem::kNumWarps;

        constexpr index_t kNPerBlock = Problem::kM0;
        constexpr index_t kMPerBlock = Problem::kMaxSplits;

        constexpr index_t NPerThread =
            GetVectorSizeForTile<kBlockSize, kMPerBlock, kNPerBlock, LSEDataType>();
        constexpr index_t NThreads = kNPerBlock / NPerThread;

        constexpr index_t MThreadsPerWarp = get_warp_size() / NThreads;
        constexpr index_t MPerThread      = kMPerBlock / (kNumWarps * MThreadsPerWarp);

        static_assert(NThreads * NPerThread == kNPerBlock);
        static_assert(MPerThread * kNumWarps * MThreadsPerWarp == kMPerBlock);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MPerThread, kNumWarps, MThreadsPerWarp>,
                                             sequence<NThreads, NPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // 3d + padding, shape=[kMaxSplits, kM0]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccLdsStoreBlockDescriptor()
    {
        using LSEDataType = remove_cvref_t<typename Problem::LSEDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::kMaxSplits;
        constexpr index_t kNPerBlock = Problem::kM0;
        constexpr index_t NPack =
            GetVectorSizeForTile<kBlockSize, kMPerBlock, kNPerBlock, LSEDataType>();

        constexpr auto lse_acc_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / NPack>{}, number<kMPerBlock>{}, number<NPack>{}),
            make_tuple(number<(kMPerBlock + 1) * NPack>{}, number<NPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto lse_acc_lds_block_desc = transform_tensor_descriptor(
            lse_acc_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kNPerBlock / NPack, NPack))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lse_acc_lds_block_desc;
    }

    // 3d + padding, shape=[kM0, kMaxSplits]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccLdsBlockDescriptor()
    {
        using LSEDataType = remove_cvref_t<typename Problem::LSEDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::kMaxSplits;
        constexpr index_t kNPerBlock = Problem::kM0;
        constexpr index_t NPack =
            GetVectorSizeForTile<kBlockSize, kMPerBlock, kNPerBlock, LSEDataType>();

        constexpr auto lse_acc_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / NPack>{}, number<kMPerBlock>{}, number<NPack>{}),
            make_tuple(number<(kMPerBlock + 1) * NPack>{}, number<NPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto lse_acc_t_lds_block_desc = transform_tensor_descriptor(
            lse_acc_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kNPerBlock / NPack, NPack))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));

        return lse_acc_t_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccRegTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::kMaxSplits;
        constexpr index_t kMPerBlock = Problem::kM0;

        constexpr index_t NThreads   = 4;
        constexpr index_t NPerThread = kNPerBlock / NThreads;

        constexpr index_t MThreads       = kBlockSize / NThreads;
        constexpr index_t MPerThread     = kMPerBlock / MThreads;
        constexpr index_t MWarps         = kBlockSize / get_warp_size();
        constexpr index_t MThreadPerWarp = get_warp_size() / NThreads;

        static_assert(NThreads * NPerThread == kNPerBlock);
        static_assert(MWarps * MThreadPerWarp * MPerThread == kMPerBlock);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<MWarps, MThreadPerWarp, MPerThread>, sequence<NThreads, NPerThread>>,
                tuple<sequence<1>, sequence<2, 1>>,
                tuple<sequence<0>, sequence<0, 1>>,
                sequence<1, 2>,
                sequence<2, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOaccDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kNPerBlock = Problem::kN1;

        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M2 = min(kMPerBlock / M1, get_warp_size());
        constexpr index_t N0 = get_warp_size() / M2;
        constexpr index_t N1 = kNPerBlock / N0;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<N0, N1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
