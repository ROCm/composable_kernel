// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ true,
                                          /* NumPrefetchK = */ 3,
                                          /* NumPrefetchV = */ 3>
{
    using Base = BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                                     /* AsyncCopy = */ true,
                                                     /* NumPrefetchK = */ 3,
                                                     /* NumPrefetchV = */ 3>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        if constexpr(Problem::kKVMemoryLayout ==
                     BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
        {
            using VDataType                 = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t kDwordx4Bytes = 16;
            return kDwordx4Bytes / sizeof(VDataType);
        }
        else
        {
            return Base::template GetAlignmentV<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        if constexpr(Problem::kKVMemoryLayout ==
                     BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
        {
            // For VECTORIZED_LAYOUT, kKPack should match GEMM's kKPerThread
            // to ensure correct LDS access pattern
            constexpr auto gemm_k_decomp  = GetGemmKDecomposition<Problem>();
            constexpr index_t kKPerThread = gemm_k_decomp.template at<1>();
            return kKPerThread;
        }
        else
        {
            return Base::template GetSmemKPackV<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        if constexpr(Problem::kKVMemoryLayout ==
                     BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
        {
            // For VECTORIZED_LAYOUT, we need to use our GetSmemKPackV for V size calculation
            constexpr index_t SingleKSize = [&]() {
                constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
                constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
                constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
                constexpr index_t WarpSize   = ck_tile::get_warp_size();

                constexpr index_t KPack   = Base::template GetSmemKPackK<Problem>();
                constexpr index_t KVector = Base::template GetAlignmentK<Problem>();
                constexpr index_t kPad    = KPack;

                static_assert(WarpSize * KVector >= kKPerBlock &&
                              WarpSize * KVector % kKPerBlock == 0);
                constexpr index_t LanesPerK  = kKPerBlock / KVector;
                constexpr index_t LaneGroups = WarpSize / LanesPerK;
                constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

                return NumIssues * NumWarps * (WarpSize * KVector + kPad);
            }();

            constexpr index_t SingleVSize = [&]() {
                using VDataType                = remove_cvref_t<typename Problem::VDataType>;
                constexpr index_t Banks        = get_n_lds_banks();
                constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
                constexpr index_t kKPack       = GetSmemKPackV<Problem>(); // Use our override!
                static_assert(PixelsPerRow % kKPack == 0);
                constexpr index_t NPerRow    = PixelsPerRow / kKPack;
                constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
                constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
                static_assert(kNPerBlock % NPerRow == 0);
                static_assert(kKPerBlock % kKPack == 0);

                return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
            }();

            return max(SingleKSize, SingleVSize);
        }
        else
        {
            return Base::template GetSingleSmemElementSpaceSize<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        if constexpr(Problem::kKVMemoryLayout ==
                     BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
        {
            using VDataType                = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t Banks        = get_n_lds_banks();
            constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
            constexpr index_t kKPack       = GetSmemKPackV<Problem>();
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr index_t NPerRow    = PixelsPerRow / kKPack;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<Base::NumKVLdsBuffers>{},
                           number<kKPerBlock / kKPack>{},
                           number<kNPerBlock / NPerRow>{},
                           number<NPerRow>{},
                           number<kKPack>{}),
                make_tuple(number<GetSingleSmemElementSpaceSize<Problem>()>{},
                           number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                           number<PixelsPerRow + kKPack>{},
                           number<kKPack>{},
                           number<1>{}),
                number<kKPack>{},
                number<1>{});

            constexpr auto v_lds_block_desc = transform_tensor_descriptor(
                v_lds_block_desc_0,
                make_tuple(make_merge_transform(make_tuple(number<Base::NumKVLdsBuffers>{},
                                                           number<kNPerBlock / NPerRow>{},
                                                           number<NPerRow>{})),
                           make_merge_transform(
                               make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return v_lds_block_desc;
        }
        else
        {
            return Base::template MakeVLdsBlockDescriptor<Problem>();
        }
    }

    // Helper to get GEMM's K decomposition parameters (kABKLane, kKPerThread)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetGemmKDecomposition()
    {
        // Get the KV block GEMM and extract warp gemm's K decomposition
        constexpr auto gemm = Base::template GetKVBlockGemm<Problem>();
        using BlockGemm     = remove_cvref_t<decltype(gemm)>;
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        // Return kABKLane and kKPerThread from warp gemm
        return make_tuple(number<WG::WarpGemmAttribute::Impl::kABKLane>{},
                          number<WG::kKPerThread>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        if constexpr(Problem::kKVMemoryLayout ==
                     BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
        {
            // For VECTORIZED_LAYOUT, use column-major distribution (K direction vector load)
            // The K decomposition must match GEMM's BWarpDstrEncoding to ensure correct LDS access
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

            // Get GEMM's K decomposition (kABKLane, kKPerThread)
            constexpr auto gemm_k_decomp  = GetGemmKDecomposition<Problem>();
            constexpr index_t kABKLane    = gemm_k_decomp.template at<0>();
            constexpr index_t kKPerThread = gemm_k_decomp.template at<1>();

            // K1 = kKPerThread (inner K dimension, matches GEMM's expectation)
            // K0 = kKPerBlock / K1 (outer K dimension)
            // But we need K0 to match kABKLane for the per-warp iteration
            constexpr index_t K1 = kKPerThread;
            constexpr index_t K0 = kABKLane;

            // Verify K decomposition matches GEMM's BWarpDstrEncoding requirements
            static_assert(K0 == kABKLane, "K0 must match GEMM's kABKLane for correct LDS access");
            static_assert(K1 == kKPerThread,
                          "K1 must match GEMM's kKPerThread for correct LDS access");

            // K0 * K1 may be less than kKPerBlock, so we need outer iteration
            constexpr index_t KPerIter   = K0 * K1;
            constexpr index_t KOuterIter = kKPerBlock / KPerIter;

            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            static_assert(N2 != 0, "N2 is zero, which will lead to a division by zero error.");
            static_assert(N1 != 0, "N1 is zero, which will lead to a division by zero error.");
            constexpr index_t N0 = kNPerBlock / (N2 * N1);
            static_assert(N0 != 0, "N0 is zero");

            if constexpr(KOuterIter == 1)
            {
                // Simple case: K decomposition matches exactly
                constexpr auto dstr = make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<1>, sequence<2, 0>>,
                                               sequence<2, 1>,
                                               sequence<1, 0>>{});
                static_assert(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                              kNPerBlock * kKPerBlock);
                return dstr;
            }
            else
            {
                // Need outer K iteration
                constexpr index_t K2 = KOuterIter;
                constexpr auto dstr  = make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                                tuple<sequence<N0, N1, N2>, sequence<K2, K0, K1>>,
                                                tuple<sequence<1>, sequence<1, 2>>,
                                                tuple<sequence<1>, sequence<2, 1>>,
                                                sequence<2, 1, 2>,
                                                sequence<2, 0, 0>>{});
                static_assert(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                              kNPerBlock * kKPerBlock);
                return dstr;
            }
        }
        else
        {
            // For non-VECTORIZED_LAYOUT, use base class implementation
            return Base::template MakeVDramTileDistribution<Problem>();
        }
    }
};

} // namespace ck_tile
