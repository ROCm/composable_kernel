// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"

#ifndef CK_TILE_FMHA_BATCH_PREFILL_GFX11_DISABLE_KVEC4
#define CK_TILE_FMHA_BATCH_PREFILL_GFX11_DISABLE_KVEC4 0
#endif

// gfx11 WMMA C vs A layouts do not match. Fallback BlockGemmARegBSmemCRegV2
// rejects the P tile (thread_buffer size). Independent-V uses ARegBReg and a
// dedicated V LDS layout for the 256-thread N0=32 / K1=32 / N1=128 linear case.
#ifndef CK_TILE_FMHA_BATCH_PREFILL_GFX11_INDEPENDENT_V
#define CK_TILE_FMHA_BATCH_PREFILL_GFX11_INDEPENDENT_V 1
#endif

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
    // gfx11 has no parent async dwordx2 K-copy; the gfx11 policy overrides this.
    static constexpr bool kUseSyncKLoad = false;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseIndependentVBuffer()
    {
        return false;
    }

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

// gfx1100 batch-prefill: synchronous K loads, vec4 K packing on 32x32 linear
// tiles, fallback V DRAM loads, and Independent-V gemm1 on the N0=32 / K1=32 /
// N1=128 linear 256-thread predicate. Vectorized KV is out of scope.
struct BlockFmhaBatchPrefillPipelineQRKSVSAsyncGfx11Policy
    : BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicy
{
    using Parent                        = BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicy;
    using Fallback                      = Parent::Base;
    static constexpr bool kUseSyncKLoad = true;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseIndependentVBuffer()
    {
#if CK_TILE_FMHA_BATCH_PREFILL_GFX11_INDEPENDENT_V
        return Problem::kBlockSize == 256 && Problem::BlockFmhaShape::kN0 == 32 &&
               Problem::BlockFmhaShape::kK1 == 32 && Problem::BlockFmhaShape::kN1 == 128 &&
               !Problem::kHasDropout &&
               Problem::kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT;
#else
        return false;
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetIndependentVElementSpaceSize()
    {
        if constexpr(UseIndependentVBuffer<Problem>())
        {
            constexpr index_t kN = Problem::BlockFmhaShape::kN1;
            constexpr index_t kK = Problem::BlockFmhaShape::kK1;
            return kK * (kN + kN / 8);
        }
        else
        {
            return 0;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetIndependentVByteOffset()
    {
        constexpr auto lds_seq = Parent::template GetLdsBufferSequence<Problem>();
        constexpr index_t k0_loops =
            Problem::BlockFmhaShape::kQKHeaddim / Problem::BlockFmhaShape::kK0;
        constexpr index_t k1_loops = Problem::BlockFmhaShape::kN0 / Problem::BlockFmhaShape::kK1;
        constexpr index_t v_buffer = lds_seq.at(number<k0_loops + k1_loops - 1>{});
        constexpr index_t slot_elements = Parent::template GetSingleSmemElementSpaceSize<Problem>();
        static_assert(GetIndependentVElementSpaceSize<Problem>() <= slot_elements);
        constexpr index_t offset = v_buffer * slot_elements * sizeof(typename Problem::VDataType);
        // K buffers are packed from offset 0 with their own stride, not by slot, so the
        // slot-based placement alone does not keep V clear of the K prefetch.
        constexpr index_t k_bytes =
            MakeKLdsLoadBlockDescriptor<Problem>().get_element_space_size() *
            sizeof(typename Problem::KDataType);
        static_assert(offset >= k_bytes, "independent-V LDS tile overlaps the K buffers");
        return offset;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
#if CK_TILE_FMHA_BATCH_PREFILL_GFX11_DISABLE_KVEC4
        return Parent::template GetAlignmentK<Problem>();
#else
        if constexpr(Problem::kKVMemoryLayout ==
                         BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT &&
                     Problem::BlockFmhaShape::kN0 == 32 && Problem::BlockFmhaShape::kK0 == 32)
        {
            return 4;
        }
        else
        {
            return Parent::template GetAlignmentK<Problem>();
        }
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
#if CK_TILE_FMHA_BATCH_PREFILL_GFX11_DISABLE_KVEC4
        return Parent::template GetSmemKPackK<Problem>();
#else
        if constexpr(Problem::kKVMemoryLayout ==
                         BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT &&
                     Problem::BlockFmhaShape::kN0 == 32 && Problem::BlockFmhaShape::kK0 == 32)
        {
            return 4;
        }
        else
        {
            return Parent::template GetSmemKPackK<Problem>();
        }
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsLoadBlockDescriptor()
    {
        // gfx11 stages K with ordinary loads + LDS stores, not async copy.
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t kPad       = GetSmemKPackK<Problem>();

        if constexpr(Problem::kKVMemoryLayout ==
                         BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT &&
                     kKPerBlock == 32 && kNPerBlock == 32)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<Parent::Base::NumKVLdsBuffers * kNPerBlock>{},
                           number<kKPerBlock>{}),
                make_tuple(number<kKPerBlock + kPad>{}, number<1>{}),
                number<kPad>{},
                number<1>{});
        }
        else
        {
            return Parent::template MakeKLdsLoadBlockDescriptor<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = GetAlignmentK<Problem>();
        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

        constexpr index_t K1 = min(MaxVectorSize, ElemPerThread);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);
        // GetAlignmentK depends on CK_TILE_FMHA_BATCH_PREFILL_GFX11_DISABLE_KVEC4, so check
        // that the chosen vector still tiles the K block exactly.
        static_assert((kNPerBlock * kKPerBlock) % kBlockSize == 0 && kKPerBlock % K1 == 0 &&
                          get_warp_size() % K0 == 0 && kNPerBlock % (N2 * N1) == 0,
                      "K DRAM tile does not divide evenly across the block");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        return Fallback::template MakeVDramTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        return Fallback::template GetSmemKPackV<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        return Fallback::template GetSingleSmemElementSpaceSize<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        return Fallback::template MakeVLdsBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeIndependentVLdsStoreBlockDescriptor()
    {
        constexpr index_t kN            = Problem::BlockFmhaShape::kN1;
        constexpr index_t kK            = Problem::BlockFmhaShape::kK1;
        constexpr index_t kTile         = 8;
        constexpr index_t kNOuter       = kN / kTile;
        constexpr index_t kKOuter       = kK / kTile;
        constexpr index_t kNOuterStride = kTile * (kTile + 1);
        constexpr index_t kKOuterStride = kNOuter * kNOuterStride;
        constexpr auto tiled_desc       = make_naive_tensor_descriptor(
            make_tuple(number<kKOuter>{}, number<kNOuter>{}, number<kTile>{}, number<kTile>{}),
            make_tuple(
                number<kKOuterStride>{}, number<kNOuterStride>{}, number<kTile>{}, number<1>{}),
            number<8>{},
            number<1>{});

        return transform_tensor_descriptor(
            tiled_desc,
            make_tuple(make_merge_transform(make_tuple(number<kNOuter>{}, number<kTile>{})),
                       make_merge_transform(make_tuple(number<kKOuter>{}, number<kTile>{}))),
            make_tuple(sequence<1, 2>{}, sequence<0, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeIndependentVLdsLoadBlockDescriptor()
    {
        return MakeIndependentVLdsStoreBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetIndependentVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kNumGemm1Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN1,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::PDataType,
                                            typename Problem::VDataType,
                                            typename Problem::OaccDataType,
                                            Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                                            true,
                                            false,
                                            false,
                                            WGAttrNumAccessEnum::Double>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                typename Problem::VDataType,
                                                typename Problem::OaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
