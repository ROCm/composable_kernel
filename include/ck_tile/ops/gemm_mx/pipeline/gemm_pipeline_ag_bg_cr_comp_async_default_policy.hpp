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

    // Override vector size methods to force 16-byte loads for async buffer operations
    // Valid sizes for amd_async_buffer_load are 4, 12, or 16 bytes
    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeA()
    {
        // Get packed sizes for A/B
        using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
        using ADataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
        // Force 16-byte vector loads for optimal async buffer performance
        // For fp4 (1 byte): 16 elements = 16 bytes
        // For fp8 (1 byte): 16 elements = 16 bytes
        // For fp16 (2 bytes): 8 elements = 16 bytes
        // constexpr index_t vector_size_for_16_bytes = 16 / sizeof(ADataType);
        
        // return vector_size_for_16_bytes;
        static_assert(std::is_same_v<ADataType, pk_fp4_t>, "ADataType must be pk_fp4_t or pk_fp4_raw_t");
        if constexpr(std::is_same_v<ADataType, pk_fp4_t> || std::is_same_v<ADataType, pk_fp4_raw_t>)
        {
            return 32;
        }
        else
        {
            return 16;
        }
    }

    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeB()
    {
        // Get packed sizes for A/B
        using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
        using BDataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
        // Force 16-byte vector loads for optimal async buffer performance
        // For fp4 (1 byte): 16 elements = 16 bytes
        // For fp8 (1 byte): 16 elements = 16 bytes
        // For fp16 (2 bytes): 8 elements = 16 bytes
        // constexpr index_t vector_size_for_16_bytes = 16 / sizeof(BDataType);
        
        // return vector_size_for_16_bytes;
        static_assert(std::is_same_v<BDataType, pk_fp4_t>, "BDataType must be pk_fp4_t or pk_fp4_raw_t");
        if constexpr(std::is_same_v<BDataType, pk_fp4_t> || std::is_same_v<BDataType, pk_fp4_raw_t>)
        {
            return 32;
        }
        else
        {
            return 16;
        }
    }

    // Override DRAM tile distributions to use the constrained vector sizes
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize = GetVectorSizeA<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        using ALayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::AsLayoutTuple>>>;
        // Get packed sizes for A/B
        using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
        using ADataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
        constexpr index_t APackedSize = numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
        
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      MPerBlock,
                                                      KPerBlock / APackedSize,
                                                      VecLoadSize,
                                                      getATileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        else
        {
            static_assert(false, "Not implemented");
            // using TileEncodingPattern =
            //     tile_distribution_encoding_pattern_2d<BlockSize,
            //                                           KPerBlock,
            //                                           MPerBlock,
            //                                           VecLoadSize,
            //                                           getATileAccessPattern(),
            //                                           NumWaveGroups>;
            // return TileEncodingPattern::make_2d_static_tile_distribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize = GetVectorSizeB<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
        
        using BLayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::BsLayoutTuple>>>;

        // Get packed sizes for A/B
        using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
        using BDataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
        constexpr index_t BPackedSize = numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
        
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            static_assert(false, "Not implemented");
            // using TileEncodingPattern =
            //     tile_distribution_encoding_pattern_2d<BlockSize,
            //                                           KPerBlock,
            //                                           NPerBlock,
            //                                           VecLoadSize,
            //                                           getBTileAccessPattern(),
            //                                           NumWaveGroups>;
            // return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        else
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      NPerBlock,
                                                      KPerBlock / BPackedSize,
                                                      VecLoadSize,
                                                      getBTileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
    }

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        // Get packed sizes for A/B
        using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
        using ADataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
        constexpr index_t APackedSize = numeric_traits<remove_cvref_t<ADataType>>::PackedSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK / APackedSize;
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
            // constexpr index_t KPack = GetSmemPackA<Problem>();
            constexpr index_t KPack = 16;

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
        // Get packed sizes for A/B
        using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
        using BDataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
        constexpr index_t BPackedSize = numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK / BPackedSize;
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
            // constexpr index_t KPack = GetSmemPackB<Problem>();
            constexpr index_t KPack = 16;

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

    // Custom warp distribution encodings that account for packed types
    // For 16x16x128 MFMA with pk_fp4_t, the K dimension must use storage elements, not logical elements
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_AWarpDstrEncoding()
    {
        // For 16x16x128 MFMA with pk_fp4_t (PackedSize=2)
        // Physical layout in registers: [16 M-lanes, 4 K-lanes, 16 bytes per lane]
        // Each byte stores 2 fp4 values, so 16 bytes = 32 fp4 values
        // But we need to use STORAGE size (16) not LOGICAL size (32) in the distribution
        using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
        using ADataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        
        constexpr index_t kAMLane = 16;
        constexpr index_t kABKLane = 4;
        constexpr index_t kABKPerLane = 32 / APackedSize;  // Storage elements, not logical!
        
        return tile_distribution_encoding<
            sequence<>,
            tuple<sequence<kAMLane>, sequence<kABKLane, kABKPerLane>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_BWarpDstrEncoding()
    {
        using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
        using BDataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;
        
        constexpr index_t kBNLane = 16;
        constexpr index_t kABKLane = 4;
        constexpr index_t kABKPerLane = 32 / BPackedSize;  // Storage elements!
        
        return tile_distribution_encoding<
            sequence<>,
            tuple<sequence<kBNLane>, sequence<kABKLane, kABKPerLane>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};
    }

    // Custom LDS block distributions that account for packed types
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDistributionEncode()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps = typename BlockGemmShape::BlockWarps;
        using WarpTile = typename BlockGemmShape::WarpTile;
        
        constexpr index_t MWarp = BlockWarps::at(number<0>{});
        constexpr index_t NWarp = BlockWarps::at(number<1>{});
        constexpr index_t MPerXdl = WarpTile::at(number<0>{});
        constexpr index_t KPerXdl = WarpTile::at(number<2>{});
        
        using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
        using ADataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        
        // IMPORTANT: Use packed K for iteration count
        // LDS shape is [MPerBlock, KPerBlock / APackedSize]
        // WarpGemm expects [MPerXdl, KPerXdl / APackedSize] per warp per iteration
        constexpr index_t KIterPerWarp = (KPerBlock / APackedSize) / (KPerXdl / APackedSize);
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * MPerXdl);
        
        constexpr bool UseDefaultScheduler = (Problem::NumWaveGroups != 1);
        
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto a_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<NWarp>,
                                           tuple<sequence<MIterPerWarp>, sequence<KIterPerWarp>>,
                                           tuple<>,
                                           tuple<>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{};
            // Use custom warp encoding that accounts for packed types
            return detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encoding, MakeMX_AWarpDstrEncoding<Problem>());
        }
        else
        {
            constexpr auto a_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<NWarp>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            // Use custom warp encoding that accounts for packed types
            return detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encoding, MakeMX_AWarpDstrEncoding<Problem>());
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDistributionEncode()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps = typename BlockGemmShape::BlockWarps;
        using WarpTile = typename BlockGemmShape::WarpTile;
        
        constexpr index_t MWarp = BlockWarps::at(number<0>{});
        constexpr index_t NWarp = BlockWarps::at(number<1>{});
        constexpr index_t NPerXdl = WarpTile::at(number<1>{});
        constexpr index_t KPerXdl = WarpTile::at(number<2>{});
        
        using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
        using BDataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;
        
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        
        // IMPORTANT: Use packed K for iteration count
        // LDS shape is [NPerBlock, KPerBlock / BPackedSize]
        // WarpGemm expects [NPerXdl, KPerXdl / BPackedSize] per warp per iteration
        constexpr index_t KIterPerWarp = (KPerBlock / BPackedSize) / (KPerXdl / BPackedSize);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * NPerXdl);
        
        constexpr bool UseDefaultScheduler = (Problem::NumWaveGroups != 1);
        
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto b_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<MWarp>,
                                           tuple<sequence<NIterPerWarp>, sequence<KIterPerWarp>>,
                                           tuple<>,
                                           tuple<>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{};
            // Use custom warp encoding that accounts for packed types
            return detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encoding, MakeMX_BWarpDstrEncoding<Problem>());
        }
        else
        {
            constexpr auto b_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<0, 1>>,
                tuple<sequence<0, 1>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            // Use custom warp encoding that accounts for packed types
            return detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encoding, MakeMX_BWarpDstrEncoding<Problem>());
        }
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
