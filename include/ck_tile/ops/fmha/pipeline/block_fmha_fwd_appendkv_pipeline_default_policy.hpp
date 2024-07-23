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
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        return sizeof(KDataType) * Problem::kTileSizeSk * (Problem::kTileSizeD);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKnewDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::kTileSizeSk;
        constexpr index_t kKPerBlock = Problem::kTileSizeD;

        constexpr index_t KPerThread = [&]() {
            if constexpr(Problem::RotaryEnum == BlockRotaryEmbeddingEnum::HALF_ROTATED)
            {
                return 8 / sizeof(KDataType);
            }
            else
            {
                return 16 / sizeof(KDataType);
            }
        }();
        constexpr index_t KThreadPerBlock = kKPerBlock / KPerThread;
        constexpr index_t NThreadPerWarp  = get_warp_size() / KThreadPerBlock;
        constexpr index_t NumWarps        = kBlockSize / get_warp_size();
        constexpr index_t NPerThread      = kNPerBlock / (NumWarps * NThreadPerWarp);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                             sequence<KThreadPerBlock, KPerThread>>,
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

            constexpr index_t NPerThread      = 16 / sizeof(VDataType);
            constexpr index_t NThreadPerBlock = kNPerBlock / NPerThread;
            constexpr index_t KThreadPerWarp  = get_warp_size() / NThreadPerBlock;
            constexpr index_t NumWarps        = kBlockSize / get_warp_size();
            constexpr index_t KPerThread      = kKPerBlock / (NumWarps * KThreadPerWarp);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NThreadPerBlock, NPerThread>,
                                                 sequence<KPerThread, NumWarps, KThreadPerWarp>>,
                                           tuple<sequence<2>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 0>>{});
        }
        else
        {
            constexpr index_t KPerThread      = 16 / sizeof(VDataType);
            constexpr index_t KThreadPerBlock = kKPerBlock / KPerThread;
            constexpr index_t NThreadPerWarp  = get_warp_size() / KThreadPerBlock;
            constexpr index_t NumWarps        = kBlockSize / get_warp_size();
            constexpr index_t NPerThread      = kNPerBlock / (NumWarps * NThreadPerWarp);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                 sequence<KThreadPerBlock, KPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeRotaryCosSinTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::kTileSizeSk;
        constexpr index_t kKPerBlock = [&]() {
            if constexpr(Problem::RotaryEnum == BlockRotaryEmbeddingEnum::HALF_ROTATED)
            {
                return Problem::kTileSizeD;
            }
            else
            {
                return Problem::kTileSizeD / 2;
            }
        }();

        constexpr index_t KPerThread      = 8 / sizeof(KDataType);
        constexpr index_t KThreadPerBlock = kKPerBlock / KPerThread;
        constexpr index_t NThreadPerWarp  = get_warp_size() / KThreadPerBlock;
        constexpr index_t NumWarps        = kBlockSize / get_warp_size();
        constexpr index_t NPerThread      = kNPerBlock / (NumWarps * NThreadPerWarp);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                             sequence<KThreadPerBlock, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
