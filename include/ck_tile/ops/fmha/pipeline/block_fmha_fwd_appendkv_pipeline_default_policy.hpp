// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaFwdAppendKVPipelineDefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        return 16 / sizeof(QDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        using VLayout   = remove_cvref_t<typename Problem::VLayout>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t kBlockSize   = Problem::kBlockSize;
            constexpr index_t kNPerBlock   = Problem::kTileSizeSk;
            constexpr index_t kKPerBlock   = Problem::kTileSizeDv;
            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

            // TODO: not correct!
            if constexpr(total_pixels > 4)
                return 4;
            else
                return 2;
        }
        else
        {
            return 16 / sizeof(VDataType);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return 1;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKnewDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::kTileSizeSk;
        constexpr index_t kKPerBlock = Problem::kTileSizeD;

        constexpr index_t K1 = 16 / sizeof(KDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVnewDramTileDistribution()
    {
        using VLayout   = remove_cvref_t<typename Problem::VLayout>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::kTileSizeDv;
        constexpr index_t kKPerBlock = Problem::kTileSizeSk;

        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {

            constexpr index_t N1 = 16 / sizeof(VDataType);
            constexpr index_t N0 = kNPerBlock / N1;
            constexpr index_t K2 = get_warp_size() / N0;
            constexpr index_t K1 = kBlockSize / get_warp_size();
            constexpr index_t K0 = kKPerBlock / (K2 * K1);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2>>,
                                           tuple<sequence<2>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 0>>{});
        }
        else
        {
            constexpr index_t K1 = 16 / sizeof(VDataType);
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N0 = kNPerBlock / (N2 * N1);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }
};

} // namespace ck_tile
