// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/data_cache_prefetch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

namespace detail {
// Resolves a problem/trait type's microscaling block size: yields T::ScaleBlockSize
// when the member exists, otherwise falls back to the legacy default of 32 (so
// problems that predate the ScaleBlockSize member keep compiling unchanged).
template <typename T, typename = void>
struct scale_block_size_or_default
{
    static constexpr index_t value = 32;
};
template <typename T>
struct scale_block_size_or_default<T, std::void_t<decltype(T::ScaleBlockSize)>>
{
    static constexpr index_t value = T::ScaleBlockSize;
};
} // namespace detail

enum class MultiCastDirection
{
    kM,
    kN,
    kMN
};

// Default policy for GemmPipelineAgBgCrCompTDM
template <bool WaveSpecialized                               = false,
          ck_tile::DataCachePrefetchKind DataCachePrefetchA_ = ck_tile::DataCachePrefetchKind::None,
          ck_tile::DataCachePrefetchKind DataCachePrefetchB_ = ck_tile::DataCachePrefetchKind::None>
struct GemmPipelineAgBgCrCompTDMDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompTDMDefaultPolicy<WaveSpecialized,
                                                                            DataCachePrefetchA_,
                                                                            DataCachePrefetchB_>>
{
    using Base =
        UniversalGemmBasePolicy<GemmPipelineAgBgCrCompTDMDefaultPolicy<WaveSpecialized,
                                                                       DataCachePrefetchA_,
                                                                       DataCachePrefetchB_>>;

    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchA = DataCachePrefetchA_;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchB = DataCachePrefetchB_;

    template <typename Problem>
    using LdsADataType = typename Problem::ADataType;

    template <typename Problem>
    using LdsBDataType = typename Problem::BDataType;

    static constexpr index_t VecByteSize = 16;
    // currently implement basic situation: the tile is divided into same parts
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        // for wave specialized policy, only one wave per workgroup will load A / B matrix from DRAM
        // to LDS
        constexpr index_t warpNum = WaveSpecialized ? 1 : (BlockSize / get_warp_size());

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ALayout =
            remove_cvref_t<std::tuple_element_t<number<0>{}, problem_as_layout_t<Problem>>>;

        // Tile : MPerBlock X KPerBlock
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(MPerBlock % warpNum == 0, "MPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, MPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : KPerBlock * MPerBlock
        else
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<MPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        // for wave specialized policy, only one wave per workgroup will load A / B matrix from DRAM
        // to LDS
        constexpr index_t warpNum = WaveSpecialized ? 1 : (BlockSize / get_warp_size());

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using BLayout =
            remove_cvref_t<std::tuple_element_t<number<0>{}, problem_bs_layout_t<Problem>>>;

        // Tile : KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<NPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : NPerBlock * KPerBlock
        else
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(NPerBlock % warpNum == 0, "NPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, NPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        if constexpr(Base::template is_a_load_tr<Problem>)
        {
            return Base::template MakeALdsBlockDescriptorForTrLoad<Problem>();
        }
        else
        {
            constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

            constexpr auto LdsPaddingConfigA = Base::template GetLdsPaddingConfig<Problem, true>();
            constexpr auto IsNeedPadding     = LdsPaddingConfigA[Base::I0];
            // set to -1 to make sure PaddingDataAmount = 0 when IsNeedPadding = false
            constexpr auto PaddingAmount = IsNeedPadding ? LdsPaddingConfigA[Base::I1] : -1;
            using ADataType              = LdsADataType<Problem>;
            constexpr index_t PackedSize = numeric_traits<ADataType>::PackedSize;
            constexpr auto DataTypeSize  = sizeof(ADataType);
            constexpr index_t AVectorLen = VecByteSize / DataTypeSize * PackedSize;
            constexpr index_t MLdsLayerRequired =
                get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize * PackedSize;
            constexpr auto MLdsLayer = max(1, MLdsLayerRequired);
            // calculate how many elements to pad to avoid bank conflict
            constexpr index_t BytesPerDword = sizeof(int32_t);
            constexpr auto PaddingDataAmount =
                (PaddingAmount + 1) * BytesPerDword / DataTypeSize * PackedSize;

            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<MPerBlock / MLdsLayer>{},
                           number<KPerBlock / AVectorLen * MLdsLayer>{},
                           number<AVectorLen>{}),
                make_tuple(number<KPerBlock * MLdsLayer + PaddingDataAmount>{},
                           number<AVectorLen>{},
                           number<1>{}),
                number<AVectorLen>{},
                number<1>{});

            constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                           make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<KPerBlock / AVectorLen>{})),
                           make_pass_through_transform(number<AVectorLen>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            constexpr auto a_lds_block_desc = transform_tensor_descriptor(
                a_lds_block_desc_1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<MPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KPerBlock / AVectorLen>{}, number<AVectorLen>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return a_lds_block_desc;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        if constexpr(Base::template is_b_load_tr<Problem>)
        {
            return Base::template MakeBLdsBlockDescriptorForTrLoad<Problem>();
        }
        else
        {
            constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

            constexpr auto LdsPaddingConfigB = Base::template GetLdsPaddingConfig<Problem, false>();
            constexpr auto IsNeedPadding     = LdsPaddingConfigB[Base::I0];
            // set to -1 to make sure PaddingDataAmount = 0 when IsNeedPadding = false
            constexpr auto PaddingAmount = IsNeedPadding ? LdsPaddingConfigB[Base::I1] : -1;
            using BDataType              = LdsBDataType<Problem>;
            constexpr index_t PackedSize = numeric_traits<BDataType>::PackedSize;
            constexpr auto DataTypeSize  = sizeof(BDataType);

            constexpr index_t BVectorLen = VecByteSize / DataTypeSize * PackedSize;
            constexpr index_t NLdsLayerRequired =
                get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize * PackedSize;
            constexpr auto NLdsLayer = max(1, NLdsLayerRequired);
            // calculate how many elements to pad to avoid bank conflict
            constexpr index_t BytesPerDword = sizeof(int32_t);
            constexpr auto PaddingDataAmount =
                (PaddingAmount + 1) * BytesPerDword / DataTypeSize * PackedSize;

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NPerBlock / NLdsLayer>{},
                           number<KPerBlock / BVectorLen * NLdsLayer>{},
                           number<BVectorLen>{}),
                make_tuple(number<KPerBlock * NLdsLayer + PaddingDataAmount>{},
                           number<BVectorLen>{},
                           number<1>{}),
                number<BVectorLen>{},
                number<1>{});

            constexpr auto b_lds_block_desc_1 = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                           make_unmerge_transform(
                               make_tuple(number<NLdsLayer>{}, number<KPerBlock / BVectorLen>{})),
                           make_pass_through_transform(number<BVectorLen>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                b_lds_block_desc_1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KPerBlock / BVectorLen>{}, number<BVectorLen>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return b_lds_block_desc;
        }
    }

    template <MultiCastDirection Direction, typename Problem>
    CK_TILE_DEVICE static uint16_t GetTDMWorkgroupMask(dim3 block_id_in_cluster)
    {
        constexpr index_t MCluster = Problem::BlockGemmShape::kclusterM;
        constexpr index_t NCluster = Problem::BlockGemmShape::kclusterN;

        auto is_participant = [&](auto i_m, auto i_n) {
            if constexpr(Direction == MultiCastDirection::kM)
            {
                return i_m == block_id_in_cluster.x;
            }
            else if constexpr(Direction == MultiCastDirection::kN)
            {
                return (i_n == block_id_in_cluster.y);
            }
            else // Direction == MultiCastDirection::kMN
            {
                return (i_m == block_id_in_cluster.x) || (i_n == block_id_in_cluster.y);
            }
        };

        // Iterate over all possible (m, n) block coordinates in the cluster. If the current (m,
        // n) block is a participant according to the multicast direction, set the corresponding
        // bit in the mask. for matmul AxB, A broadcasts from M direction, B broadcasts from N
        // direction.
        uint16_t block_id_mask = 0;
        static_for<0, NCluster, 1>{}([&](auto n) {
            static_for<0, MCluster, 1>{}([&](auto m) {
                if(is_participant(m, n))
                {
                    block_id_mask |= (1 << (n * MCluster + m));
                }
            });
        });
        return block_id_mask;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetEstimatedVgprCount()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using CDataType = remove_cvref_t<typename Problem::CDataType>;

        constexpr index_t MWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I0);
        constexpr index_t NWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I1);
        constexpr index_t warpSize     = get_warp_size();
        constexpr index_t BlockSize    = Problem::kBlockSize;
        constexpr index_t BytesPerVGPR = 4;
        constexpr index_t AccVGPRNum =
            sizeof(CDataType) * MPerBlock * NPerBlock / BlockSize / BytesPerVGPR;

        // this is used to calculate DoubleBufferFactor which is 2.5; this is to make sure float
        // calculation in constexpr is avoided
        constexpr index_t DoubleBufferNumerator   = 5;
        constexpr index_t DoubleBufferDenominator = 2;

        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

        constexpr index_t ALoadVGPRNum = sizeof(ADataType) / APackedSize * MPerBlock * KPerBlock /
                                         MWarps / warpSize / BytesPerVGPR * DoubleBufferNumerator /
                                         DoubleBufferDenominator;

        constexpr index_t BLoadVGPRNum = sizeof(BDataType) / BPackedSize * NPerBlock * KPerBlock /
                                         NWarps / warpSize / BytesPerVGPR * DoubleBufferNumerator /
                                         DoubleBufferDenominator;

        constexpr index_t TotalInputVGPRNum = ALoadVGPRNum + BLoadVGPRNum;

        return make_tuple(number<AccVGPRNum>{}, number<TotalInputVGPRNum>{});
    }

    // this function is used to get SubTile Number
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPipelineSubTileNum()
    {
        constexpr index_t KPerBlock        = Problem::BlockGemmShape::kK;
        constexpr index_t KPerTile         = Problem::BlockGemmShape::WarpTile::at(Base::I2);
        constexpr index_t max_sub_tile_num = KPerBlock / KPerTile;

        constexpr auto estimated_vgpr = GetEstimatedVgprCount<Problem>();

        constexpr auto acc_vgpr_num   = estimated_vgpr.at(number<0>{});
        constexpr auto input_vgpr_num = estimated_vgpr.at(number<1>{});

        constexpr index_t vgpr_capacity = get_max_vgpr_count();
        // sub tile number; have 1, 2, 4 choices
        constexpr index_t sub_tile_num = ((input_vgpr_num + acc_vgpr_num) <= vgpr_capacity) ? 1
                                         : ((input_vgpr_num / 2 + acc_vgpr_num) <= vgpr_capacity)
                                             ? 2
                                             : 4;

        return number<min(sub_tile_num, max_sub_tile_num)>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeScaleADramTileDistribution()
    {
        using TileShape  = typename Problem::BlockGemmShape;
        using BlockWarps = typename TileShape::BlockWarps;

        constexpr index_t MWarps     = BlockWarps::at(Base::I0);
        constexpr index_t NWarps     = BlockWarps::at(Base::I1);
        constexpr index_t kMPerBlock = TileShape::kM;
        constexpr index_t kKPerBlock = TileShape::kK;

        constexpr index_t ScaleSize = detail::scale_block_size_or_default<Problem>::value;
        // PackSize = number of e8m0 bytes per int32_t element (always 4).
        // scale32: 1 int32_t per thread (4 bytes); scale16: 2 int32_t per thread (8 bytes ->
        // int64_t).
        constexpr index_t PackSize = 4;

        // WarpM: unique M-lanes per warp (equals WarpTile::M for WMMA).
        // For scale32, WarpM == WaveSize (all lanes are unique in M).
        // For scale16, WarpM < WaveSize: the remaining WaveSize/WarpM lanes
        // repeat the same M-rows (they cover different N columns), so they
        // must be replicated in the RS rather than treated as distinct M-positions.
        constexpr index_t WarpM        = TileShape::WarpTile::at(Base::I0);
        constexpr index_t NWithinWarp  = get_warp_size() / WarpM;
        constexpr index_t MIterPerWarp = kMPerBlock / MWarps / WarpM;
        static_assert(WarpM == 16 || WarpM == 32,
                      "scale tile distribution supports only WarpTile::M of 16 or 32");
        static_assert(get_warp_size() % WarpM == 0, "WarpTile::M must divide the wavefront size");

        // H0 = <MIterPerWarp, MWarps, WarpM> matching A block distribution order.
        // M-index = mIter * (MWarps * WarpM) + m_warp * WarpM + lane_m
        // P1 (lane_id) decomposes as: lane_id = n_within_warp + NWithinWarp * lane_m
        // so P1[0]=NWithinWarp (least sig, replication) and P1[1]=WarpM (most sig, M-position).
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps, NWithinWarp>,
                                       tuple<sequence<MIterPerWarp, MWarps, WarpM>,
                                             sequence<kKPerBlock / ScaleSize / PackSize, 1>>,
                                       tuple<sequence<1, 0>, sequence<0, 1>>,
                                       tuple<sequence<1, 0>, sequence<1, 2>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeScaleBDramTileDistribution()
    {
        using TileShape  = typename Problem::BlockGemmShape;
        using BlockWarps = typename TileShape::BlockWarps;

        constexpr index_t MWarps     = BlockWarps::at(Base::I0);
        constexpr index_t NWarps     = BlockWarps::at(Base::I1);
        constexpr index_t kNPerBlock = TileShape::kN;
        constexpr index_t kKPerBlock = TileShape::kK;

        constexpr index_t ScaleSize = detail::scale_block_size_or_default<Problem>::value;
        constexpr index_t PackSize  = 4;

        // WarpN: unique N-lanes per warp (equals WarpTile::N for WMMA).
        // For scale16, WaveSize/WarpN lanes repeat the same N-rows (covering
        // different M columns) and must be replicated in RS.
        constexpr index_t WarpN        = TileShape::WarpTile::at(Base::I1);
        constexpr index_t MWithinWarp  = get_warp_size() / WarpN;
        constexpr index_t NIterPerWarp = kNPerBlock / NWarps / WarpN;
        static_assert(WarpN == 16 || WarpN == 32,
                      "scale tile distribution supports only WarpTile::N of 16 or 32");
        static_assert(get_warp_size() % WarpN == 0, "WarpTile::N must divide the wavefront size");

        // H0 = <NIterPerWarp, NWarps, WarpN> matching B block distribution order.
        // N-index = nIter * (NWarps * WarpN) + n_warp * WarpN + lane_n
        // P1 (lane_id) decomposes as: lane_id = m_within_warp + MWithinWarp * lane_n
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps, MWithinWarp>,
                                       tuple<sequence<NIterPerWarp, NWarps, WarpN>,
                                             sequence<kKPerBlock / ScaleSize / PackSize, 1>>,
                                       tuple<sequence<0, 1>, sequence<0, 1>>,
                                       tuple<sequence<0, 1>, sequence<1, 2>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr auto pipeline_tune_params = GetPipelineSubTileNum<Problem>();
        constexpr index_t sub_tile_num      = pipeline_tune_params.value;
        constexpr index_t kScaleBlockSize   = detail::scale_block_size_or_default<Problem>::value;

#if defined(__gfx125__)
        // Compute WGAttrNumAccess for a single operand (A or B).
        //
        // For normal types:
        //   vec_size       = instruction K-pack (kAKPack / kBKPack)
        //   total_elements = K_warp_tile / 2  (2 lanes share K)
        //
        // For packed types (f4/f6) with transpose load:
        //   vec_size       = PackedSize * elements_per_vgpr  (one VGPR worth of packed data)
        //   total_elements = instruction K-pack              (kAKPack / kBKPack)
        constexpr auto compute_num_access = []<typename DataType>(bool is_load_tr,
                                                                  index_t instr_kpack,
                                                                  index_t k_warp_tile) constexpr {
            constexpr index_t packed_size   = numeric_traits<DataType>::PackedSize;
            constexpr index_t bits_per_elem = sizeof(DataType) * 8 / packed_size;
            // in gfx1250 always use double vgpr for fp4 and fp8 in tr load
            constexpr index_t elems_per_dvgpr = 64 / bits_per_elem;

            const bool is_packed =
                is_load_tr && ((packed_size > 1) || ((bits_per_elem == 8) && (instr_kpack > 8)));
            const auto vec_size       = is_packed ? elems_per_dvgpr : instr_kpack;
            const auto total_elements = is_packed ? instr_kpack : (k_warp_tile / 2);
            const auto ratio          = total_elements / vec_size;

            // Map the ratio to WGAttrNumAccessEnum;
            // is_packed selects between Packed* and non-Packed variants.
            switch(ratio)
            {
            case 1:
                return is_packed ? WGAttrNumAccessEnum::PackedSingle : WGAttrNumAccessEnum::Single;
            case 2:
                return is_packed ? WGAttrNumAccessEnum::PackedDouble : WGAttrNumAccessEnum::Double;
            case 4: return is_packed ? WGAttrNumAccessEnum::PackedQuad : WGAttrNumAccessEnum::Quad;
            case 8: return is_packed ? WGAttrNumAccessEnum::PackedOcta : WGAttrNumAccessEnum::Octa;
            default: return WGAttrNumAccessEnum::Invalid;
            }
        };

        // Probe the default warp gemm to get instruction K-pack sizes.
        // For scale16, probe the scale16 type directly; otherwise use the standard dispatcher.
        using WarpGemmProbe      = WarpGemmDispatcher<typename Problem::ADataType,
                                                      typename Problem::BDataType,
                                                      typename Problem::CDataType,
                                                      WarpTile::at(Base::I0),
                                                      WarpTile::at(Base::I1),
                                                      WarpTile::at(Base::I2),
                                                      Problem::TransposeC,
                                                      false,
                                                      false,
                                                      WGAttrNumAccessEnum::Default,
                                                      WGAttrNumAccessEnum::Default,
                                                      (kScaleBlockSize == 16)>;
        constexpr index_t k_warp = WarpTile::at(Base::I2);

        constexpr auto a_wg_attr_num_access =
            compute_num_access.template operator()<typename Problem::ADataType>(
                Base::template is_a_load_tr<Problem>, WarpGemmProbe::kAKPack, k_warp);

        constexpr auto b_wg_attr_num_access =
            compute_num_access.template operator()<typename Problem::BDataType>(
                Base::template is_b_load_tr<Problem>, WarpGemmProbe::kBKPack, k_warp);
#else
        constexpr auto a_wg_attr_num_access = WGAttrNumAccessEnum::Default;
        constexpr auto b_wg_attr_num_access = WGAttrNumAccessEnum::Default;
#endif
        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(Base::I0),
                                            WarpTile::at(Base::I1),
                                            WarpTile::at(Base::I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            a_wg_attr_num_access,
                                            b_wg_attr_num_access,
                                            (kScaleBlockSize == 16)>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm,
                                                                    sub_tile_num>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

// Type aliases for backward compatibility
using GemmPipelineAgBgCrCompTDMWaveSpecializedPolicy = GemmPipelineAgBgCrCompTDMDefaultPolicy<true>;

} // namespace ck_tile
