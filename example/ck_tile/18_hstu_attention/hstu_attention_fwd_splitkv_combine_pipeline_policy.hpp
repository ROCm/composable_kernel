// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

struct HstuAttentionFwdSplitKVCombinePipelinePolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOaccDramTileDistribution()
    {
        constexpr index_t kMPerBlock = Problem::kM;
        constexpr index_t kKPerBlock = Problem::kOHeaddim;
        constexpr index_t NumWarps   = Problem::NumWarps;

        constexpr index_t KVector = GetAlignmentOacc<Problem>();
        constexpr index_t OtherK  = kKPerBlock / KVector;

        if constexpr(kKPerBlock == Problem::kSubOHeaddim)
        // for kKPerBlock=32,64,128,256
        {
            static_assert((OtherK & (OtherK - 1)) == 0, "Check failed!");

            constexpr index_t KPerThread = KVector;

            // try to assign more consecutive threads on dim-K
            constexpr index_t KThreads = OtherK;

            static_assert(KThreads <= get_warp_size(), "Check failed!");

            constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                                 sequence<KThreads, KPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else // for kKPerBlock=96,160
        {
            static_assert((OtherK & (OtherK - 1)) != 0, "Check failed!");

            // ensure KThreads be power-of-2 integer
            constexpr index_t KRepPerThread = (OtherK % 3 == 0) ? 3 : 5;
            constexpr index_t KThreads      = OtherK / KRepPerThread;

            constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                                 sequence<KRepPerThread, KThreads, KVector>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 1>>,
                                           sequence<1, 2, 2>,
                                           sequence<0, 0, 2>>{});
        };
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        return Problem::GetOaccDramTileAccessMaxVectorSize();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        // should be same as GetAlignmentOacc() since o_tile will use the same encoding as
        // o_acc_tile
        return GetAlignmentOacc<Problem>();
    }
};

} // namespace ck_tile
