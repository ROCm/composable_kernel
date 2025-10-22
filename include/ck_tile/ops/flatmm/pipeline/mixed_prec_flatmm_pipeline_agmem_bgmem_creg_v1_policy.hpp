// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

#define CKTILE_FLATMM_USE_BUFFER_LOAD_LDS_AS_POSSIBLE 0

#if defined(__gfx950__)
#define CKTILE_FLATMM_ARCH_SUPPORT_BUFFER_LOAD_LDS_DWORDx4 1
#else
#define CKTILE_FLATMM_ARCH_SUPPORT_BUFFER_LOAD_LDS_DWORDx4 0
#endif

#define CKTILE_FLATMM_USE_BUFFER_LOAD_LDS             \
    (CKTILE_FLATMM_USE_BUFFER_LOAD_LDS_AS_POSSIBLE && \
     CKTILE_FLATMM_ARCH_SUPPORT_BUFFER_LOAD_LDS_DWORDx4)

struct F16xMXF4FlatmmPipelineAgBgCrPolicy : UniversalFlatmmPipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t KBPerLoad = 32;
    static constexpr index_t N_Pack    = 2; // it's fixed for fp4
    static constexpr index_t K_Pack    = 2; // it's fixed for fp4

    template <typename Problem, typename NativeADramTensorView>
    CK_TILE_HOST_DEVICE static constexpr auto
    TransformF16xF4_ATensorView(const NativeADramTensorView& a_dram_view)
    {
#if CKTILE_FLATMM_USE_BUFFER_LOAD_LDS
        constexpr int DynamicTileOffsetFlag = 0;

        constexpr index_t MPerXdl = Problem::BlockGemmShape::WarpTile::at(I0);
        constexpr index_t NPerXdl = Problem::BlockGemmShape::WarpTile::at(I1);

        static_assert(MPerXdl == 16 && NPerXdl == 16);

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack     = GetSmemPackA<Problem>();

        constexpr int ContiguousThreadsCntInDS_READ_16B = 4;

        // implement swizzle pattern on global side
        // because we can't adjust the ds_write pattern of BUFFER_LOAD_LDS.
        auto swizzle_a_dram_view_1 = transform_tensor_view(
            a_dram_view,
            make_tuple(
                // M-dim is not affected by swizzle pattern
                make_unmerge_transform(
                    make_tuple(number<DynamicTileOffsetFlag>{}, number<MPerBlock>{})),
                // K-dim is the swizzle dimension
                make_unmerge_transform(make_tuple(number<DynamicTileOffsetFlag>{},
                                                  number<KPerBlock / KPack>{},
                                                  number<KPack>{}))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}));

        auto swizzle_a_dram_view_2 = transform_tensor_view(
            swizzle_a_dram_view_1,
            make_tuple(make_pass_through_transform(number<DynamicTileOffsetFlag>{}),
                       make_xor_transform(make_tuple(number<MPerBlock>{},
                                                     number<ContiguousThreadsCntInDS_READ_16B>{})),
                       make_pass_through_transform(number<DynamicTileOffsetFlag>{}),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));

        return transform_tensor_view(
            swizzle_a_dram_view_2,
            make_tuple(
                make_merge_transform_v3_division_mod(
                    make_tuple(number<DynamicTileOffsetFlag>{}, number<MPerBlock>{})),
                make_merge_transform_v3_division_mod(make_tuple(number<DynamicTileOffsetFlag>{},
                                                                number<KPerBlock / KPack>{},
                                                                number<KPack>{}))),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
#else
        return a_dram_view;
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeF16xF4_ReadALdsBlockDescriptor()
    {
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
    CK_TILE_HOST_DEVICE static constexpr auto MakeF16xF4_WriteALdsBlockDescriptor()
    {
#if CKTILE_FLATMM_USE_BUFFER_LOAD_LDS
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack     = GetSmemPackA<Problem>();
        return make_naive_tensor_descriptor(make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                            make_tuple(number<KPerBlock>{}, number<1>{}),
                                            number<KPack>{},
                                            number<1>{});
#else
        return MakeF16xF4_ReadALdsBlockDescriptor<Problem>();
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeF16xF4_ALDS_TileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

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

        constexpr index_t BlockSize                = Problem::kBlockSize;
        constexpr index_t WaveSize                 = get_warp_size();
        [[maybe_unused]] constexpr index_t WaveNum = BlockSize / WaveSize;

        constexpr index_t N_Warp = TileShape::BlockWarps::at(number<1>{});

        [[maybe_unused]] constexpr index_t XDLPerBlock =
            TileShape::kK / TileShape::WarpTile::at(I2);
        constexpr index_t K_Lane = 64 / TileShape::WarpTile::at(I1);
        constexpr index_t N_Lane = TileShape::WarpTile::at(I1);

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
