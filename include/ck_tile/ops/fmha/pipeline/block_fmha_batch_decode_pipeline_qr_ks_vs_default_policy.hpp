// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaBatchDecodeWithPagedKVCachePipelineQRKSVSDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ false,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    using BasePolicy = BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                                           /* AsyncCopy = */ false,
                                                           /* NumPrefetchK = */ 1,
                                                           /* NumPrefetchV = */ 1>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        // this should align with MakeQDramTileDistribution()
        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;

        return static_cast<index_t>(16 / sizeof(OaccDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

        constexpr index_t KPerThread     = kMaxVecLoad;
        constexpr index_t KThreads       = kKPerBlock / KPerThread;
        constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                             sequence<KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        return BasePolicy::template MakeQRegTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        // TODO: this is for 3d layout
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return static_cast<index_t>(16 / sizeof(QDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kKPack = min(ElemPerThread, GetSmemKPackQ<Problem>());

        constexpr auto q_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kMPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto q_lds_block_desc = transform_tensor_descriptor(
            q_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return q_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemNPackS()
    {
        using SDataType = remove_cvref_t<typename Problem::SaccDataType>;
        return static_cast<index_t>(16 / sizeof(SDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kNPack     = GetSmemNPackS<Problem>();

        constexpr auto s_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / kNPack>{}, number<kMPerBlock>{}, number<kNPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kNPack>{}, number<kNPack>{}, number<1>{}),
            number<kNPack>{},
            number<1>{});

        constexpr auto s_lds_block_desc = transform_tensor_descriptor(
            s_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kNPerBlock / kNPack>{}, number<kNPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return s_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;

        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        static_assert(MWarp == 1, "Check failed!");

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kTileK     = Problem::BlockFmhaShape::kN0;

        // K2 is equal to Impl::kABKPerLane * kKIterPerWarpGemm
        constexpr index_t K3 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K2 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = kKPerBlock / (K2 * K3);
        constexpr index_t K0 = kTileK / kKPerBlock;
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        constexpr auto s2_block_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<2, 2>>,
                                       sequence<1, 2, 2, 2>,
                                       sequence<0, 0, 1, 3>>{};

        constexpr auto s2_block_dstr = make_static_tile_distribution(s2_block_dstr_encoding);

        return s2_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        return MakeKLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeS()
    {
        return MakeSLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::SaccDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return max(GetSmemSizeQ<Problem>(), GetSmemSizeK<Problem>()) +
               max(GetSmemSizeV<Problem>(), GetSmemSizeS<Problem>());
    }
    
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistributionPreshuffled()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck_tile::get_warp_size();

        constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t KLanes  = 4; // 16x16x16 b16 300x
        constexpr index_t NFlat  = 16; // within a wave
        static_assert(kKPerBlock % (KLanes * KVector) == 0);
        constexpr index_t KRepeat  = kKPerBlock / KLanes / KVector; // within a wave


        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                        tuple<sequence<N0, N1, N2>, sequence<KRepeat, KLanes, NFlat, KVector>>,
                                        tuple<sequence<1>, sequence<1, 2>>,
                                        tuple<sequence<2>, sequence<1, 0>>,
                                        sequence<1, 2>,
                                        sequence<0, 1>>{});
    }
    
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kNumGemm0Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr auto warp_gemm = []() {
            constexpr index_t WarpGemmM = Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
            static_assert(WarpGemmM == 4 || WarpGemmM == 16 || WarpGemmM == 32);

            if constexpr(std::is_same_v<typename Problem::QDataType, half_t> &&
                         std::is_same_v<typename Problem::KDataType, half_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                    return WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
                else // WarpGemmM == 4
                    return WarpGemmMfmaF16F16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                              std::is_same_v<typename Problem::KDataType, bf16_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                    return WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
                else // WarpGemmM == 4
                    return WarpGemmMfmaBf16Bf16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, fp8_t> &&
                              std::is_same_v<typename Problem::KDataType, fp8_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                static_assert(WarpGemmM == 32);

                // TODO: hard coded here. Otherwise, it may incorrect result
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            } // TODO - bf8_t
        }();

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 decltype(warp_gemm)>;
        static_assert(1 < Problem::kNumGemm0Warps);
        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
