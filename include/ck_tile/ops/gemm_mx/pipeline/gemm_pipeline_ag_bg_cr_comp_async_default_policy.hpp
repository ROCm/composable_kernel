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

    // MX scaling configuration: pack 4 consecutive e8m0 scales in K dimension
    static constexpr int MXdlPack = 1;  // No M packing
    static constexpr int NXdlPack = 1;  // No N packing
    static constexpr int KXdlPack = 4;  // Pack 4 consecutive e8m0 scales in K = 4 bytes = 1 int32
    static constexpr int BlockScaleSize = 32;  // Each e8m0 scale covers 32 elements in K

    // Override vector size methods to ensure compatibility with async buffer operations
    // Valid sizes for amd_async_buffer_load are 4, 12, or 16 bytes
    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeA()
    {
        using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
        using ADataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
        constexpr index_t APackedSize = numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
        
        // Call base policy's dynamic vector size calculation
        constexpr index_t vector_size = 
            UniversalGemmBasePolicy<MXGemmPipelineAgBgCrCompAsyncDefaultPolicy>::
                template GetVectorSizeA<Problem, IsWave32Host>();
        
        // Calculate actual byte load size (storage bytes = logical elements / PackedSize * sizeof)
        constexpr index_t byte_load_size = vector_size * sizeof(ADataType) / APackedSize;
        
        // Ensure async buffer load requirements: must be 4, 12, or 16 bytes
        static_assert(byte_load_size == 4 || byte_load_size == 12 || byte_load_size == 16,
                      "Vector load size must be 4, 12, or 16 bytes for async buffer operations");
        
        return vector_size;
    }

    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeB()
    {
        using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
        using BDataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
        constexpr index_t BPackedSize = numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
        
        // Call base policy's dynamic vector size calculation
        constexpr index_t vector_size = 
            UniversalGemmBasePolicy<MXGemmPipelineAgBgCrCompAsyncDefaultPolicy>::
                template GetVectorSizeB<Problem, IsWave32Host>();
        
        // Calculate actual byte load size (storage bytes = logical elements / PackedSize * sizeof)
        constexpr index_t byte_load_size = vector_size * sizeof(BDataType) / BPackedSize;
        
        // Ensure async buffer load requirements: must be 4, 12, or 16 bytes
        static_assert(byte_load_size == 4 || byte_load_size == 12 || byte_load_size == 16,
                      "Vector load size must be 4, 12, or 16 bytes for async buffer operations");
        
        return vector_size;
    }

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {        
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        /// NOTE: for flatmm style byte tensor, divide KPerBlock by APackedSize to get STORAGE dimensions
        // using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
        // using ADataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
        // constexpr index_t APackedSize = numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
        // constexpr index_t KPerBlock = Problem::BlockGemmShape::kK / APackedSize; // Use STORAGE dimensions
        /// NOTE: use original KPerBlock
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
            static_assert(KPack >= 16, "KPack must be at least 16");

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
        /// NOTE: for flatmm style byte tensor, divide KPerBlock by BPackedSize to get STORAGE dimensions
        // using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
        // using BDataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
        // constexpr index_t BPackedSize = numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
        // constexpr index_t KPerBlock = Problem::BlockGemmShape::kK / BPackedSize; // Use STORAGE dimensions
        /// NOTE: use original KPerBlock
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
            static_assert(KPack >= 16, "KPack must be at least 16");

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
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_DramTileDistribution()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps = typename BlockGemmShape::BlockWarps;
        using WarpTile = typename BlockGemmShape::WarpTile;
        
        constexpr index_t MWarp = BlockWarps::at(number<0>{});
        constexpr index_t NWarp = BlockWarps::at(number<1>{});
        constexpr index_t MPerXdl = WarpTile::at(number<0>{});
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t ScaleKDimPerBlock = KPerBlock / BlockScaleSize / KXdlPack;  // int32s per block
        constexpr index_t K_Lane = get_warp_size() / MPerXdl;  // 64/16 = 4 threads in K dimension
        // constexpr index_t KPackedElementsPerThread = ScaleKDimPerBlock / K_Lane;  // 4/4 = 1 for K=512
        
        // Scale A: [MWarp * MPerXdl, ScaleKDimPerBlock] warp-level tile  
        // For K=512: [16, 4], distribute 4 int32s across 4 K_Lane threads (1 each)
        // Strided packing: thread at K_lane=k gets one int32 with scales for all kIters at K position k
        // Distribution: Replicate in M dimension, distribute in K dimension (no vectorization - scalar loads)
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarp>,                              // repeat over NWarps
                                       tuple<sequence<MWarp, MPerXdl>,               // M dimension
                                             sequence<ScaleKDimPerBlock, K_Lane>>, // K dimension
                                       tuple<sequence<1, 0>, sequence<2, 1>>,        // <MWarp, NWarp>, <K_Lane, MPerXdl>
                                       tuple<sequence<0, 0>, sequence<1, 1>>,
                                       sequence<2>,                                   // ScaleKDimPerBlock, all int32 needed to cover KPerBlock
                                       sequence<0>>{});
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
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t ScaleKDimPerBlock = KPerBlock / BlockScaleSize / KXdlPack;  // int32s per block
        constexpr index_t K_Lane = get_warp_size() / NPerXdl;  // 64/16 = 4 threads in K dimension
        // constexpr index_t KPackedElementsPerThread = ScaleKDimPerBlock / K_Lane;  // 4/4 = 1 for K=512
        
        // Scale B: [ScaleKDimPerBlock, NWarp * NPerXdl] warp-level tile
        // For K=512: [4, 64], distribute 4 int32s across 4 K_Lane threads (1 each)
        // Strided packing: thread at K_lane=k gets one int32 with scales for all kIters at K position k
        // Distribution: Distribute in K dimension (no vectorization - scalar loads), replicate in N dimension
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarp>,                              // repeat over MWarps
                                       tuple<sequence<NWarp, NPerXdl>,               // N dimension
                                             sequence<ScaleKDimPerBlock, K_Lane>>, // K dimension
                                       tuple<sequence<0, 1>, sequence<2, 1>>,        // <MWarp, NWarp>, <K_Lane, NPerXdl>
                                       tuple<sequence<0, 0>, sequence<1, 1>>,
                                       sequence<2>,                                   // ScaleKDimPerBlock, all int32 needed to cover KPerBlock
                                       sequence<0>>{}); 
    }
};
} // namespace ck_tile
