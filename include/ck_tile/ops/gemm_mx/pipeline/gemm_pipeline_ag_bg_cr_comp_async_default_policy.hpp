// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

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

    // MX scaling configuration: pack 4 consecutive e8m0 scales in K dimension
    static constexpr int MXdlPack = 1;  // No M packing
    static constexpr int NXdlPack = 1;  // No N packing
    static constexpr int KXdlPack = 4;  // Pack 4 consecutive e8m0 scales in K = 4 bytes = 1 int32

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeALdsBlockDescriptor
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
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeBLdsBlockDescriptor
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

        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(is_a_load_tr<Problem> || is_b_load_tr<Problem>) ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements                  ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements              ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements              ? WGAttrNumAccessEnum::Quad
                                                              : WGAttrNumAccessEnum::Invalid;

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }

    // MX Scale tile distributions for loading from global memory  
    // Using the proven "Flat" patterns from v1 policy
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_DramTileDistribution()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps = typename BlockGemmShape::BlockWarps;
        using WarpTile = typename BlockGemmShape::WarpTile;
        
        constexpr index_t MWarp = BlockWarps::at(number<0>{});
        constexpr index_t NWarp = BlockWarps::at(number<1>{});
        constexpr index_t MPerXdl = WarpTile::at(number<0>{});
        constexpr index_t K_Lane = get_warp_size() / MPerXdl;  // 4 for 16x16 mfma
        
        // Scale A: [MWarp * MPerXdl, K/32/KXdlPack] for warp-level tile  
        // Distribution: simple 2D for loading int32 packed scales
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarp>,                      // repeat over NWarps
                                       tuple<sequence<MWarp, MPerXdl>,       // M dimension
                                             sequence<K_Lane, 1>>,            // K dimension (int32 vec load)
                                       tuple<sequence<1, 0>, sequence<2, 1>>, // which direction
                                       tuple<sequence<0, 0>, sequence<0, 1>>, // which index
                                       // <repeat, vec_load>
                                       sequence<2>,
                                       sequence<1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_DramTileDistribution()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps = typename BlockGemmShape::BlockWarps;
        using WarpTile = typename BlockGemmShape::WarpTile;
        
        constexpr index_t MWarp = BlockWarps::at(number<0>{});
        constexpr index_t NWarp = BlockWarps::at(number<1>{});
        constexpr index_t NPerXdl = WarpTile::at(number<1>{});
        constexpr index_t K_Lane = get_warp_size() / NPerXdl;  // 4 for 16x16 mfma

        static_assert(K_Lane == 4, "K_Lane must be 4 for 16x16 mfma");
        static_assert(NPerXdl == 16, "NPerXdl must be 16 for 16x16 mfma");
        static_assert(MWarp == 1, "MWarp must be 1 for 16x16 mfma");
        
        // Scale B: [K/32/KXdlPack, NWarp * NPerXdl] for warp-level tile
        // Layout is [K, N] where K is packed int32
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarp>,                      // repeat over MWarps
                                       tuple<sequence<K_Lane, 1>,             // K dimension (int32 vec load)
                                             sequence<NWarp, NPerXdl>>,      // N dimension
                                       tuple<sequence<2, 1>, sequence<0, 1>>, // which direction
                                       tuple<sequence<0, 1>, sequence<0, 0>>, // which index
                                       // <repeat, vec_load>
                                       sequence<1>,
                                       sequence<1>>{});
    }
};
} // namespace ck_tile
