// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/data_cache_prefetch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

template <DataCachePrefetchKind DataCachePrefetchA_ = DataCachePrefetchKind::None,
          DataCachePrefetchKind DataCachePrefetchB_ = DataCachePrefetchKind::None>
struct UniversalWeightPreshufflePipelineAgBgCrTDMPolicy
    : public GemmPipelineAgBgCrCompTDMDefaultPolicy<false, DataCachePrefetchA_, DataCachePrefetchB_>
{
    using Base =
        GemmPipelineAgBgCrCompTDMDefaultPolicy<false, DataCachePrefetchA_, DataCachePrefetchB_>;

    using Base::I0;
    using Base::I1;
    using Base::I2;

    static constexpr DataCachePrefetchKind DataCachePrefetchA = DataCachePrefetchA_;
    static constexpr DataCachePrefetchKind DataCachePrefetchB = DataCachePrefetchB_;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a =
            sizeof(typename Problem::ADataType) *
            Base::template MakeALdsBlockDescriptor<Problem>().get_element_space_size();
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
        return Problem::VectorLoadSize / sizeof(typename Problem::ADataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKBPerLoad()
    {
        using TileShape = typename Problem::BlockGemmShape;
        return TileShape::WarpTile::at(I2) / 2;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        constexpr index_t kNPerBlock = TileShape::kN;
        constexpr index_t kKPerBlock = TileShape::kK;
        constexpr index_t NIterPerWarp =
            kNPerBlock / TileShape::BlockWarps::at(I1) / TileShape::WarpTile::at(I1);
        constexpr index_t KIterPerWarp = kKPerBlock / TileShape::WarpTile::at(I2);

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        // need to match GetBlockWeightPreshuffle's implementation
        using BTypeToUse =
            std::conditional_t<std::is_same_v<typename Problem::BDataType, ck_tile::pk_int4_t>,
                               typename Problem::ADataType,
                               typename Problem::BDataType>;

        constexpr index_t PackedSize = numeric_traits<BTypeToUse>::PackedSize;

        constexpr index_t KBPerLoad = GetKBPerLoad<Problem>(); // WarpTileK / 2

        constexpr index_t NWarpBlock    = TileShape::WarpTile::at(I1) / 16;
        constexpr index_t MaxVecSize    = 16 / sizeof(BTypeToUse) * PackedSize;
        constexpr index_t KItemsPerLoad = min(KBPerLoad, MaxVecSize);
        // KFragment = how many loads iteration per thread per WarpTileK
        constexpr index_t KFragment = KBPerLoad / KItemsPerLoad;

        constexpr index_t KThdPerWave = WaveSize;
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t KRepeat     = KIterPerWarp;
        static_assert(TileShape::flatKPerWarp == KThdPerWave * KBPerLoad, "wrong");

        constexpr index_t NBPerLoad   = 1;
        constexpr index_t NThdPerWave = 1;
        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp
        constexpr index_t NRepeat     = NIterPerWarp;

        constexpr index_t WaveRepeat =
            WaveNum * NWarpBlock / TileShape::flatNPerWarp; // which is MWarps
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<WaveRepeat>,
                                       tuple<sequence<NRepeat,
                                                      NWavePerBlk,
                                                      NWarpBlock, // iterations per warp in dimN
                                                      NThdPerWave,
                                                      NBPerLoad>, // second
                                                                  // direction
                                             sequence<KRepeat,
                                                      KFragment,
                                                      KWavePerBlk,
                                                      KThdPerWave,
                                                      KItemsPerLoad>>, // first
                                                                       // direction
                                       // wave in blk,     // thd in wave
                                       // <M, K>           // <M, K>
                                       tuple<sequence<0, 1, 2>, sequence<1, 2>>, // which
                                                                                 // direction
                                       tuple<sequence<0, 1, 2>, sequence<3, 3>>, // which index
                                       // <repeat, vec_load>
                                       sequence<1, 2, 1, 2, 2>,
                                       sequence<0, 0, 2, 1, 4>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeScaleADramTileDistribution()
    {
        using TileShape  = typename Problem::BlockGemmShape;
        using BlockWarps = typename TileShape::BlockWarps;

        constexpr index_t MWarps     = BlockWarps::at(I0);
        constexpr index_t NWarps     = BlockWarps::at(I1);
        constexpr index_t kMPerBlock = TileShape::kM;
        constexpr index_t kKPerBlock = TileShape::kK;

        constexpr index_t ScaleSize = 32;

        // for gfx1250 mx gemm supports 32x32x128
        static_assert(TileShape::WarpTile::at(I0) == 32);

        constexpr index_t MIterPerWarp = kMPerBlock / MWarps / TileShape::WarpTile::at(I0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<NWarps>,
                tuple<sequence<MIterPerWarp, MWarps, get_warp_size()>,
                      sequence<kKPerBlock / ScaleSize / 4, 1>>, // 4 is because scale tensor is
                                                                // int32_t data type, each int32_t
                                                                // exists 4 fp8 scale values
                tuple<sequence<1, 0>, sequence<1>>,
                tuple<sequence<1, 0>, sequence<2>>,
                sequence<1, 2, 2>,
                sequence<0, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeScaleBDramTileDistribution()
    {
        using TileShape  = typename Problem::BlockGemmShape;
        using BlockWarps = typename TileShape::BlockWarps;

        constexpr index_t MWarps     = BlockWarps::at(I0);
        constexpr index_t NWarps     = BlockWarps::at(I1);
        constexpr index_t kNPerBlock = TileShape::kN;
        constexpr index_t kKPerBlock = TileShape::kK;

        constexpr index_t ScaleSize = 32;

        // for gfx1250 mx gemm supports 32x32x128
        static_assert(TileShape::WarpTile::at(I1) == 32);

        constexpr index_t NIterPerWarp = kNPerBlock / NWarps / TileShape::WarpTile::at(I1);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<MWarps>,
                tuple<sequence<NIterPerWarp, NWarps, get_warp_size()>,
                      sequence<kKPerBlock / ScaleSize / 4, 1>>, // 4 is because scale tensor is
                                                                // int32_t data type, each int32_t
                                                                // exists 4 fp8 scale values
                tuple<sequence<0, 1>, sequence<1>>,
                tuple<sequence<0, 1>, sequence<2>>,
                sequence<1, 2, 2>,
                sequence<0, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockWeightPreshuffle()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;
        using BTypeToUse =
            std::conditional_t<std::is_same_v<typename Problem::BDataType, ck_tile::pk_int4_t>,
                               typename Problem::ADataType,
                               typename Problem::BDataType>;
        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            BTypeToUse,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC>;

        using BlockWeightPreshufflePolicy =
            BlockWeightPreshuffleASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                              typename Problem::BDataType,
                                                              typename Problem::CDataType,
                                                              BlockWarps,
                                                              WarpGemm>;
        return BlockWeightPreshuffleASmemBRegCReg<Problem, BlockWeightPreshufflePolicy>{};
    }
};

} // namespace ck_tile
