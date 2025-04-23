// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

struct UniversalFlatmmPipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck_tile;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / 8>{}, number<kMPerBlock>{}, number<8>{}),
            make_tuple(number<(kMPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a = sizeof(typename Problem::ADataType) *
                                        MakeALdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();

        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackA()
    {
        return Problem::VectorLoadSize;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t M1           = Problem::VectorLoadSize / sizeof(ADataType);
            constexpr index_t M0           = MPerBlock / M1;
            constexpr index_t total_pixels = MPerBlock * KPerBlock / BlockSize;
            static_assert(total_pixels % M1 == 0);
            constexpr index_t K3    = total_pixels / M1;
            constexpr index_t KPack = GetSmemPackA<Problem>();
            static_assert(KPack % K3 == 0);
            constexpr index_t K2 = KPack / K3;
            if constexpr(get_warp_size() % (K2 * M0))
            {
                constexpr index_t K1 = get_warp_size() / (K2 * M0);
                constexpr index_t K0 = BlockSize / get_warp_size();
                static_assert(KPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1>, sequence<K0, K1, K2, K3>>,
                                               tuple<sequence<2>, sequence<2, 1, 2>>,
                                               tuple<sequence<0>, sequence<1, 0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1   = (K2 * M0) / get_warp_size();
                constexpr index_t K2_m = K2 / K1;
                constexpr index_t K0   = BlockSize / get_warp_size() / K1;
                static_assert(KPerBlock == K0 * K1 * K2_m * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1>, sequence<K0, K1, K2_m, K3>>,
                                               tuple<sequence<2, 2>, sequence<1, 2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
        }
        else
        {
            constexpr index_t K1 = 16 / sizeof(ADataType);
            constexpr index_t K0 = KPerBlock / K1;
            constexpr index_t M2 = get_warp_size() / K0;
            // coalesce reading for each blocks
            if constexpr(get_warp_size() % (M2 * K0) == 0)
            {
                constexpr index_t M1 = BlockSize / get_warp_size();
                static_assert(M2 != 0, "M2 is zero, which will lead to a division by zero error.");
                static_assert(M1 != 0, "M1 is zero, which will lead to a division by zero error.");
                constexpr index_t M0 = MPerBlock / (M2 * M1);
                static_assert(M0 * M1 * M2 == MPerBlock,
                              "Incorrect M0, M2, M1 configuration! "
                              "M0, M1, M2 must cover whole MPerBlock!");

                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<1>, sequence<2, 0>>,
                                               sequence<1, 2>,
                                               sequence<0, 1>>{});
            }
            else
            {
                constexpr index_t M0 = BlockSize / get_warp_size();
                constexpr index_t M1 = MPerBlock / (M2 * M0);
                static_assert(M0 * M1 * M2 == MPerBlock,
                              "Incorrect M0, M1, M2 configuration! "
                              "M0, M1, M2 must cover whole MPerBlock!");
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<0>, sequence<2, 0>>,
                                               sequence<1, 2>,
                                               sequence<1, 1>>{});
            }
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBFlatDramTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        using TileShape = typename Problem::BlockGemmShape; // ck_tile::TileFlatmmShape

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t KBPerLoad =
            Problem::VectorLoadSize / sizeof(BDataType); // dwordx4 load B elem cnt
        constexpr index_t KThdPerWave = WaveSize;        // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t KRepeat     = 1;

        constexpr index_t NBPerLoad   = 1;
        constexpr index_t NThdPerWave = 1;
        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(TileShape::idxN); // N_Warp
        constexpr index_t NRepeat     = 1;

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WaveRepeat>,                                          // ?
                tuple<sequence<NRepeat, NWavePerBlk, NThdPerWave, NBPerLoad>,  // second direction
                      sequence<KRepeat, KWavePerBlk, KThdPerWave, KBPerLoad>>, // first  direction
                // wave in blk,     // thd in wave
                // <M, K>           // <M, K>
                tuple<sequence<0, 1, 2>, sequence<1, 2>>, // which direction
                tuple<sequence<0, 1, 1>, sequence<2, 2>>, // which index
                // <repeat, vec_load>
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledARegBlockDistribution()
    {
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t M1           = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t M0           = kMPerBlock / M1;
        constexpr index_t total_pixels = kMPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % M1 == 0);
        constexpr index_t K3     = total_pixels / M1;
        constexpr index_t kKPack = GetSmemPackA<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t warp_size = get_warp_size();
        if constexpr(warp_size % (K2 * M0) == 0)
        {
            constexpr index_t K1 = warp_size / (K2 * M0);
            constexpr index_t K0 = kBlockSize / warp_size;

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1>, sequence<K0, K1, K2, K3>>,
                                           tuple<sequence<2>, sequence<2, 1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
        else
        {
            constexpr index_t K1   = (K2 * M0) / get_warp_size();
            constexpr index_t K2_m = K2 / K1;
            constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1>, sequence<K0, K1, K2_m, K3>>,
                                           tuple<sequence<2, 2>, sequence<1, 2>>,
                                           tuple<sequence<0, 1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockFlatmm()
    {
        using AccDataType = float;
        using BlockWarps  = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile    = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm    = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                AccDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                Problem::TransposeC>;

        using BlockFlatmmPolicy =
            BlockFlatmmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                    typename Problem::BDataType,
                                                    typename Problem::CDataType,
                                                    BlockWarps,
                                                    WarpGemm>;
        return BlockFlatmmASmemBSmemCRegV1<Problem, BlockFlatmmPolicy>{};
    }
};

} // namespace ck_tile
