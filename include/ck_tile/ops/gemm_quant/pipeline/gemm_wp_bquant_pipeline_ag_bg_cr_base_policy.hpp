// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/block/block_wp_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

struct GemmWPQuantPipelineAgBgCrPolicy : public UniversalWeightPreshufflePipelineAgBgCrPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ()
    {
        using BQDataType              = remove_cvref_t<typename Problem::BQDataType>;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t NPerBlockBQ = NPerBlock / Problem::BQuantGroupSize::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ = KPerBlock / Problem::BQuantGroupSize::kK;

        return GetABQGlobalVectorLoadSize<Problem, BQDataType, NPerBlockBQ, KPerBlockBQ>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::MakeBQDramTileDistribution<Problem>();
    }

    // as UniversalWeightPreshufflePipelineAgBgCrPolicy's MakeBFlatDramTileDistribution is changed;
    // move original UniversalWeightPreshufflePipelineAgBgCrPolicy's implementation to here
    // temporarily
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;
        using BTypeToUse =
            std::conditional_t<std::is_same_v<typename Problem::BDataType, ck_tile::pk_int4_t>,
                               typename Problem::ADataType,
                               typename Problem::BDataType>;

        constexpr index_t PackedSize = numeric_traits<BTypeToUse>::PackedSize;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

#if defined(__gfx11__)
        constexpr index_t KRepeatInWave = 2;
#else
        constexpr index_t KRepeatInWave = 1;
#endif
        constexpr index_t KBPerLoad =
            min(GetKBPerLoad<Problem>(),
                KRepeatInWave * 16 / static_cast<index_t>(sizeof(BTypeToUse)) * PackedSize);
        constexpr index_t KThdPerWave = WaveSize / KRepeatInWave; // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t KRepeat     = 1;
        constexpr index_t KAccess     = GetKBPerLoad<Problem>() / KBPerLoad;
        static_assert(TileShape::flatKPerWarp == KAccess * KThdPerWave * KBPerLoad, "wrong");

        constexpr index_t NBPerLoad   = 1;
        constexpr index_t NThdPerWave = 1;
        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp
        constexpr index_t NRepeat     = 1;

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WaveRepeat, KRepeatInWave>,                          // ?
                tuple<sequence<NRepeat, NWavePerBlk, NThdPerWave, NBPerLoad>, // second direction
                      sequence<KRepeat,
                               KAccess,
                               KWavePerBlk,
                               KThdPerWave,
                               KBPerLoad>>,                  // first direction
                tuple<sequence<0, 1, 2>, sequence<0, 1, 2>>, // which direction
                tuple<sequence<0, 1, 2>, sequence<1, 2, 3>>, // which index
                sequence<1, 2, 1, 2, 2>,
                sequence<0, 0, 3, 1, 4>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockWeightPreshuffleBQuant()
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

        // TODO : Use a custom block policy for AsBrCr
        using BlockGemmPolicy =
            BlockWeightPreshuffleASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                              typename Problem::BDataType,
                                                              typename Problem::CDataType,
                                                              BlockWarps,
                                                              WarpGemm>;
        return BlockGemmWeightPreshuffleBQuantARegBRegCReg<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
