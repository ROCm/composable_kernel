// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

struct GemmMicroscalePipelineAgBgCrPolicy : public UniversalGemmPipelineAgBgCrPolicy
{
    using Base = UniversalGemmPipelineAgBgCrPolicy;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    template <typename Problem>
    using ALdsDataType_ = typename Problem::ADataType;

    template <typename Problem>
    using BLdsDataType_ = std::conditional_t<Problem::BCastPolicy == CastPolicy::BeforeLDSWrite,
                                             typename Problem::BComputeDataType,
                                             typename Problem::BDataType>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ()
    {
        using BQLayout                = remove_cvref_t<typename Problem::BQLayout>;
        using BQDataType              = remove_cvref_t<typename Problem::BQDataType>;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t NPerBlockBQ = NPerBlock / Problem::BQuantGroupSize::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ = KPerBlock / Problem::BQuantGroupSize::kK;

        // Support both RowMajor and ColumnMajor layouts for BQ
        if constexpr(std::is_same_v<BQLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetABQGlobalVectorLoadSize<Problem, BQDataType, KPerBlockBQ, NPerBlockBQ>();
        }
        else
        {
            return GetABQGlobalVectorLoadSize<Problem, BQDataType, NPerBlockBQ, KPerBlockBQ>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBRegTileDistribution()
    {
        using BLayout = remove_cvref_t<typename Problem::BLayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize =
            Problem::FixedVectorSize ? Problem::VectorSizeB : GetVectorSizeB<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
        // Tile: KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      KPerBlock,
                                                      NPerBlock,
                                                      VecLoadSize,
                                                      getBTileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        // Tile: NPerBlock X KPerBlock
        else
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      NPerBlock,
                                                      KPerBlock,
                                                      VecLoadSize,
                                                      getBTileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        using BQLayout = remove_cvref_t<typename Problem::BQLayout>;
        using BLayout  = remove_cvref_t<typename Problem::BLayout>;
        // If we apply scale before writing to LDS, we need a tile distribution for
        // BQuant consistent with global memory reading of matrix B, while
        // if we apply scale after reading from LDS, we need a tile distribution for
        // BQuant consistent with the MMA instructions layout
        if constexpr(Problem::BCastPolicy == CastPolicy::AfterLDSRead)
        {
            using BlockGemmShape = typename Problem::BlockGemmShape;

            constexpr index_t BlockSize   = Problem::kBlockSize;
            constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
            constexpr index_t NPerBlockBQ = NPerBlock / Problem::BQuantGroupSize::kN;
            constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
            constexpr index_t KPerBlockBQ = KPerBlock / Problem::BQuantGroupSize::kK;

            using WarpTile = typename Problem::BlockGemmShape::WarpTile;
            using WarpGemm = WarpGemmDispatcher<typename Problem::AComputeDataType,
                                                typename Problem::BComputeDataType,
                                                typename Problem::CDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                Problem::TransposeC>;

            using TileEncodingPattern =
                tile_distribution_encoding_pattern_bq<BlockGemmShape,
                                                      WarpGemm,
                                                      BlockSize,
                                                      KPerBlockBQ, // Logical K dimension
                                                      NPerBlockBQ, // Logical N dimension
                                                      Problem::BQuantGroupSize::kN,
                                                      Problem::BQuantGroupSize::kK,
                                                      BQLayout>;

            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        else
        {
            constexpr index_t BlockSize = Problem::kBlockSize;
            constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

            constexpr index_t VecLoadSize =
                Problem::FixedVectorSize ? Problem::VectorSizeB : GetVectorSizeB<Problem>();
            constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

            constexpr index_t warp_size  = get_warp_size();
            constexpr index_t num_warps  = BlockSize / get_warp_size();
            constexpr index_t LargestVec = (KPerBlock * NPerBlock) / (num_warps * warp_size);
            constexpr index_t b_vec      = VecLoadSize > LargestVec ? LargestVec : VecLoadSize;

            constexpr index_t KScale = KPerBlock / Problem::BQuantGroupSize::kK;

            // For each BQ layout we need different encodings whether B has the same layout or not
            // TODO: generalize encodings for different BQuantGroupSize granularity
            if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>)
            {
                if constexpr(std::is_same_v<BQLayout, BLayout>)
                {
                    constexpr index_t K0 = KPerBlock / b_vec;
                    constexpr index_t K1 = K0 / KScale;
                    constexpr index_t K3 = KScale;
                    constexpr index_t K2 = 1;

                    constexpr index_t N0 = num_warps / NumWaveGroups;
                    constexpr index_t N1 = warp_size / K0;
                    constexpr index_t N2 = NPerBlock / (N0 * N1);

                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<K1>,
                                                   tuple<sequence<N0, N1, N2>, sequence<K3, K2>>,
                                                   tuple<sequence<1>, sequence<1, 2, 0>>,
                                                   tuple<sequence<0>, sequence<1, 0, 0>>,
                                                   sequence<1, 2>,
                                                   sequence<2, 1>>{});
                }
                else
                {
                    constexpr index_t N1 = NPerBlock / b_vec;
                    constexpr index_t N2 = b_vec;

                    constexpr index_t KRepeatInWave     = warp_size / N1;
                    constexpr index_t KRepeatAcrossWave = num_warps / KScale;

                    constexpr index_t K2 = num_warps / KRepeatAcrossWave;

                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<KRepeatAcrossWave, KRepeatInWave>,
                                                   tuple<sequence<1, N1, N2>, sequence<K2, 1, 1>>,
                                                   tuple<sequence<1, 2, 0>, sequence<0, 1, 2>>,
                                                   tuple<sequence<0, 0, 0>, sequence<1, 1, 1>>,
                                                   sequence<1, 2>,
                                                   sequence<2, 2>>{});
                }
            }
            else
            {
                if constexpr(std::is_same_v<BQLayout, BLayout>)
                {
                    constexpr index_t NScale = NPerBlock / Problem::BQuantGroupSize::kN;
                    constexpr index_t N0     = NScale / b_vec;
                    constexpr index_t N1     = b_vec;

                    constexpr index_t KLanes  = warp_size / N0;
                    constexpr index_t KVec    = KPerBlock / KLanes / num_warps;
                    constexpr index_t KRepeat = KPerBlock / KScale / KVec;

                    constexpr index_t KRepeatInWave     = KRepeat > KLanes ? KLanes : 1;
                    constexpr index_t KRepeatAcrossWave = KRepeat > KLanes ? KRepeat / KLanes : 1;

                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<KRepeatAcrossWave, KRepeatInWave>,
                                                   tuple<sequence<1, 1, 1>, sequence<N0, N1>>,
                                                   tuple<sequence<1, 0>, sequence<1, 0, 2>>,
                                                   tuple<sequence<0, 0>, sequence<1, 1, 0>>,
                                                   sequence<1, 2>,
                                                   sequence<2, 1>>{});
                }
                else
                {
                    constexpr index_t KRepeatInWave = Problem::BQuantGroupSize::kK / b_vec;
                    constexpr index_t K1            = KScale;

                    constexpr index_t N0 = num_warps / NumWaveGroups;
                    constexpr index_t N1 = warp_size / (KRepeatInWave * K1);

                    // Number of contiguous elements in N dimension when reading B matrix
                    // becomes the vector size of BQ
                    constexpr index_t N2 = NPerBlock / (BlockSize / (KPerBlock / b_vec));

                    return make_static_tile_distribution(
                        tile_distribution_encoding<sequence<1, 1, KRepeatInWave>,
                                                   tuple<sequence<1, K1, 1>, sequence<N0, N1, N2>>,
                                                   tuple<sequence<1, 0, 2>, sequence<2, 0, 1, 0>>,
                                                   tuple<sequence<0, 0, 0>, sequence<1, 1, 1, 2>>,
                                                   sequence<1, 2>,
                                                   sequence<2, 2>>{});
                }
            }
        }
    }

    // Return AttrNumAccess for a given warp tile (defined by ThreadElements) and data type
    template <typename DataType, bool UseLoadTranspose, index_t ThreadElements>
    static constexpr auto GetAttrNumAccess(bool_constant<UseLoadTranspose>, number<ThreadElements>)
    {
        constexpr index_t PackedSize  = numeric_traits<remove_cvref_t<DataType>>::PackedSize;
        constexpr index_t vector_size = DS_READ_TR_SIZE() / sizeof(DataType) * PackedSize;

        return !UseLoadTranspose                   ? WGAttrNumAccessEnum::Single
               : vector_size == ThreadElements     ? WGAttrNumAccessEnum::Single
               : vector_size * 2 == ThreadElements ? WGAttrNumAccessEnum::Double
               : vector_size * 4 == ThreadElements ? WGAttrNumAccessEnum::Quad
                                                   : WGAttrNumAccessEnum::Invalid;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps       = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile         = typename Problem::BlockGemmShape::WarpTile;
        using AComputeDataType = typename Problem::AComputeDataType;
        using BComputeDataType = typename Problem::BComputeDataType;
#if defined(__gfx125__)
        constexpr auto wg_attr_num_accessA = WGAttrNumAccessEnum::Default;
        constexpr auto wg_attr_num_accessB = WGAttrNumAccessEnum::Default;
#else

        using LDSADataType = ALdsDataType_<Problem>;
        using LDSBDataType = BLdsDataType_<Problem>;

        static_assert(Problem::BQuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize!");
        constexpr auto thread_elements =
            number<WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size()>{};

        constexpr auto is_a_load_tr_v = bool_constant<Base::template is_a_load_tr<Problem>>{};
        constexpr auto is_b_load_tr_v = bool_constant<Base::template is_b_load_tr<Problem>>{};
        constexpr auto is_any_load_tr = is_a_load_tr_v || is_b_load_tr_v;

        constexpr auto wg_attr_num_access_compute =
            GetAttrNumAccess<AComputeDataType>(is_any_load_tr, thread_elements);
        constexpr auto wg_attr_num_accessA =
            std::is_same_v<LDSADataType, LDSBDataType>
                ? wg_attr_num_access_compute
                : GetAttrNumAccess<LDSADataType>(is_a_load_tr_v, thread_elements);
        constexpr auto wg_attr_num_accessB =
            std::is_same_v<LDSADataType, LDSBDataType>
                ? wg_attr_num_access_compute
                : GetAttrNumAccess<LDSBDataType>(is_b_load_tr_v, thread_elements);
#endif
        using WarpGemm = WarpGemmDispatcher<AComputeDataType,
                                            BComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_accessA,
                                            wg_attr_num_accessB>;
        static_assert(is_any_of<AComputeDataType, fp8_t, bf8_t, bf16_t, fp16_t>::value &&
                      is_any_of<BComputeDataType, fp8_t, bf8_t, bf16_t, fp16_t>::value);
        static_assert(std::is_same_v<typename Problem::CDataType, float>);

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<
            typename Problem::ADataType,
            std::conditional_t<std::is_same_v<typename Problem::BDataType, pk_fp4_t>,
                               typename Problem::ADataType,
                               typename Problem::BDataType>,
            typename Problem::CDataType,
            BlockWarps,
            WarpGemm>;

        return BQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
