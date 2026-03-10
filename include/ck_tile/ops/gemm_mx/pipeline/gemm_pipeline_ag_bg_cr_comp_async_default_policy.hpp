// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include <type_traits>

namespace ck_tile {
// Default policy for MXGemmPipelineAgBgCrCompAsync
// Customized methods: MakeALdsBlockDescriptor, MakeBLdsBlockDescriptor
// GetBlockGemm implementation is copied from GemmPipelineAgBgCrCompV4DefaultPolicy
// Adds MX scale tile distributions
struct MXGemmPipelineAgBgCrCompAsyncDefaultPolicy
    : public UniversalGemmBasePolicy<MXGemmPipelineAgBgCrCompAsyncDefaultPolicy>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;

    // MX scaling configuration: each e8m0 scale covers 32 elements in K
    static constexpr int BlockScaleSize = 32;

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = GetSmemPackA<Problem>();

            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack>{}, number<MPerBlock>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            return transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(
                    make_pass_through_transform(number<MPerBlock>{}),
                    make_merge_transform(make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_b_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                                             make_tuple(number<NPerBlock>{}, number<1>{}),
                                             number<NPerBlock>{},
                                             number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = GetSmemPackB<Problem>();

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack>{}, number<NPerBlock>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            return transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(
                    make_pass_through_transform(number<NPerBlock>{}),
                    make_merge_transform(make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        using ADataType = typename Problem::ADataType;
        using BDataType = typename Problem::BDataType;
        using CDataType = typename Problem::CDataType;

        // FP4 and FP8 require different layouts for the scaled mfma instructions
        constexpr auto wg_attr_num_access =
            (std::is_same_v<ADataType, fp8_t> || std::is_same_v<BDataType, fp8_t>)
                ? WGAttrNumAccessEnum::Double
                : WGAttrNumAccessEnum::Single;

        using WarpGemm = WarpGemmDispatcher<ADataType,
                                            BDataType,
                                            CDataType, // AccDataType
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }

    // MX Scale tile distributions for loading from global memory
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_DramTileDistribution()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t MWarp     = BlockWarps::at(number<0>{});
        constexpr index_t NWarp     = BlockWarps::at(number<1>{});
        constexpr index_t MPerXdl   = WarpTile::at(number<0>{});
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K_Lane = get_warp_size() / MPerXdl; // 64/16 = 4 threads in K dimension
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * MPerXdl);
        constexpr index_t KPerXdl      = WarpTile::at(number<2>{});
        constexpr index_t KIterPerWarp = KPerBlock / KPerXdl;
        constexpr index_t KPerLane     = KPerXdl / BlockScaleSize / K_Lane;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<NWarp>,                                 // repeat over MWarps
                tuple<sequence<MIterPerWarp, MWarp, MPerXdl>,    // M dimension (first)
                      sequence<KIterPerWarp, K_Lane, KPerLane>>, // K dimension (second)
                tuple<sequence<0, 1>, sequence<2, 1>>, // <MWarp, NWarp>, <K_Lane, MPerXdl>
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<2, 1, 2>, // <KIterPerWarp, MIterPerWarp, KPerLane>
                sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_DramTileDistribution()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t MWarp     = BlockWarps::at(number<0>{});
        constexpr index_t NWarp     = BlockWarps::at(number<1>{});
        constexpr index_t NPerXdl   = WarpTile::at(number<1>{});
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t K_Lane    = get_warp_size() / NPerXdl; // 64/16 = 4 threads in K dimension
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * NPerXdl);

        constexpr index_t KPerXdl      = WarpTile::at(number<2>{});
        constexpr index_t KIterPerWarp = KPerBlock / KPerXdl;
        constexpr index_t KPerLane     = KPerXdl / BlockScaleSize / K_Lane;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<MWarp>,                                 // repeat over MWarps
                tuple<sequence<NIterPerWarp, NWarp, NPerXdl>,    // N dimension (first)
                      sequence<KIterPerWarp, K_Lane, KPerLane>>, // K dimension (second)
                tuple<sequence<0, 1>, sequence<2, 1>>, // <MWarp, NWarp>, <K_Lane, MPerXdl>
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<2, 1, 2>, // <KIterPerWarp, NIterPerWarp, KPerLane>
                sequence<0, 0, 2>>{});
    }
};
} // namespace ck_tile
