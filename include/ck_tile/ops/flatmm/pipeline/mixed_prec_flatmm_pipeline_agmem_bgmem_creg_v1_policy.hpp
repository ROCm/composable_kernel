// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

struct F16xMXF4FlatmmPipelineAgBgCrPolicy : UniversalFlatmmPipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t KBPerLoad = 32;
    static constexpr index_t N_Pack    = 2; // it's fixed for fp4
    static constexpr index_t K_Pack    = 2; // it's fixed for fp4

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeF16xF4_ALdsBlockDescriptor()
    {
        using namespace ck_tile;

        constexpr index_t MPerXdl = Problem::BlockGemmShape::WarpTile::at(I0);
        constexpr index_t NPerXdl = Problem::BlockGemmShape::WarpTile::at(I1);

        static_assert(MPerXdl == 16 && NPerXdl == 16);

        /*reduce transform layers,compare with old ck*/
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack     = GetSmemPackA<Problem>();

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / KPack>{}, number<MPerBlock>{}, number<KPack>{}),
            make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
            number<KPack>{},
            number<1>{});

        constexpr int ContiguousThreadsCntInDS_READ_16B = 4;

        constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<MPerBlock>{},
                                                     number<ContiguousThreadsCntInDS_READ_16B>{})),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_permuted,
            make_tuple(make_pass_through_transform(number<MPerBlock>{}),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeFp16xF4_ADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;

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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeF16xF4_ALDS_TileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        static_assert(TileShape::WarpTile::at(I1) == 16, "requires XDL_N == 16");
        static_assert(TileShape::BlockWarps::at(I0) == 1, "requires Wave_M == 1");

        constexpr int Repeat = TileShape::BlockWarps::at(number<1>{});
        constexpr int M0     = TileShape::WarpTile::at(I0);

        constexpr int K_Lane = 64 / TileShape::WarpTile::at(I1); // 4

        constexpr int K2             = TileShape::WarpTile::at(I2) / K_Lane; // 8
        constexpr int XDL_PerThreadK = KBPerLoad / K2;                       // 4
        constexpr int K0             = K_Lane;                               // 4

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<Repeat>,
                                       tuple<sequence<M0>, sequence<K0, XDL_PerThreadK, K2>>,
                                       tuple<sequence<0>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<0, 0>>,
                                       sequence<2>,
                                       sequence<2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeFp4BFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        static_assert(TileShape::WarpTile::at(I1) == 16, "only for XDL_N == 16");

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t KThdPerWave = WaveSize; // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;

        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WaveRepeat>,                                 // ?
                tuple<sequence<NWavePerBlk, N_Pack>,                  // second
                                                                      // direction
                      sequence<KWavePerBlk, KThdPerWave, KBPerLoad>>, // first  direction
                // wave in blk,     // thd in wave
                // <M, K>           // <M, K>
                tuple<sequence<0, 1, 2>, sequence<2>>, // which direction
                tuple<sequence<0, 0, 0>, sequence<1>>, // which index
                // <repeat, vec_load>
                sequence<2>,
                sequence<2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeFp4ScaleBFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape; // ck_tile::TileFlatmmShape

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t N_Warp = TileShape::BlockWarps::at(number<1>{});

        constexpr index_t XDLPerBlock = TileShape::kK / TileShape::WarpTile::at(I2);
        constexpr index_t K_Lane      = 64 / TileShape::WarpTile::at(I1);
        constexpr index_t N_Lane      = TileShape::WarpTile::at(I1);

        constexpr index_t NWavePerBlk = N_Warp;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,                                       // ?
                tuple<sequence<NWavePerBlk>,                      // second direction
                      sequence<K_Lane, N_Lane, N_Pack * K_Pack>>, // first
                                                                  // direction
                // wave in blk,     // thd in wave
                // <M, K>           // <M, K>
                tuple<sequence<1>, sequence<2, 2>>, // which direction
                tuple<sequence<0>, sequence<0, 1>>, // which index
                // <repeat, vec_load>
                sequence<2>,
                sequence<2>>{});
    }
};

} // namespace ck_tile