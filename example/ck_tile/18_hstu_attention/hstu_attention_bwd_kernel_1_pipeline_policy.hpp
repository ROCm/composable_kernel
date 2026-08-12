// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_problem.hpp>
#include <ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp>

#include "block_gemm_areg_bsmem_creg_v2_hack_0.hpp"
#include "block_gemm_areg_bsmem_creg_v2_hack_1.hpp"
#include "block_gemm_areg_bsmem_trload_creg_v2_hack_1.hpp"

#include "hstu_attention_kernel_util.hpp"

namespace ck_tile {

// Pipeline policy for Kernel 1 backward (dQ computation).
//
// Tile shapes in Kernel 1:
//   Q, dO, O : [kM0, kQKHeaddim]   -- A operands for Gemm0/Gemm2, register-resident
//   K, V     : [kN0Sub, kQKHeaddim] -- B operands for Gemm0/Gemm2, staged through LDS
//   dQ       : [kM0, kQKHeaddim]   -- C output of Gemm4
//
// Gemm0: S    [kM0, kN0Sub] = alpha * Q  [kM0, kQKHeaddim] @ K [kN0Sub, kQKHeaddim]
// Gemm2: dP   [kM0, kN0Sub] = dO [kM0, kQKHeaddim] @ V [kN0Sub, kQKHeaddim]
// Gemm4: dQ   [kM0, kQKHeaddim] += alpha * dS [kM0, kN0Sub] @ K^T [kQKHeaddim, kN0Sub]
//
// K and V occupy separate LDS regions (Gemm0 and Gemm2 run in separate static_for loops),
// laid out consecutively:
//   [k_lds region | v_lds region | kt_lds region]
// GetSmemSize returns GetSmemSizeK + GetSmemSizeV + GetSmemSizeKT.
struct HstuAttentionBwdKernel1PipelinePolicy
{
    // Gemm0, Gemm2 and Gemm4 all use n0_loop, which unrolls the Gemm along kN0
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumN0Loops()
    {
        constexpr index_t n0_loops =
            Problem::HstuAttentionTileSetting::kN0 / Problem::HstuAttentionTileSetting::kN0Sub;

        return n0_loops;
    }

    // -------------------------------------------------------------------------
    // K, V each need two buffers to support pre-write of the prefecthed data to LDS
    // -------------------------------------------------------------------------
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumKVLdsBuffers()
    {
        return 2;
    }

    // -------------------------------------------------------------------------
    // Alignment helpers (vector load widths)
    // -------------------------------------------------------------------------

    // Q alignment  -- based on [kM0, kQKHeaddim] tile
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QKVDataType);
        using BlockGemm                 = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    // K alignment  -- based on [kN0Sub, kQKHeaddim] tile
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        return Problem::GetKDramTileAccessMaxVectorSize();
    }

    // V alignment  -- same tile shape as K, so same alignment
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        return GetAlignmentK<Problem>();
    }

    // dO alignment -- same tile shape as Q, so same alignment
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOGrad()
    {
        return GetAlignmentQ<Problem>();
    }

    // dQ alignment -- same tile shape as Q (output [kM0, kQKHeaddim])
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQGrad()
    {
        return GetAlignmentQ<Problem>();
    }

    // -------------------------------------------------------------------------
    // DRAM tile distributions (govern how data is partitioned across threads)
    // -------------------------------------------------------------------------

    // Q (and dO, O) : [kM0, kQKHeaddim], register-resident A operand
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        return BlockGemm::template MakeABlockTileDistribution<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kQKHeaddim>();
    }

    // dO -- identical to Q
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradRegTileDistribution()
    {
        return MakeQRegTileDistribution<Problem>();
    }

    // O (softmax path) -- identical to Q
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeORegTileDistribution()
    {
        return MakeQRegTileDistribution<Problem>();
    }

    // K : [kN0Sub, kQKHeaddim], B operand loaded from DRAM into registers then stored to LDS
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKVector   = GetAlignmentK<Problem>();
        constexpr index_t OtherK     = kKPerBlock / kKVector;

        if constexpr(detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            static_assert((OtherK & (OtherK - 1)) == 0, "Check failed!");
            constexpr index_t KPerThread     = kKVector;
            constexpr index_t KThreads       = OtherK;
            constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                 sequence<KThreads, KPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else
        {
            static_assert((OtherK & (OtherK - 1)) != 0, "Check failed!");
            constexpr index_t KRepPerThread  = (OtherK % 3 == 0) ? 3 : 5;
            constexpr index_t KThreads       = OtherK / KRepPerThread;
            constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                 sequence<KRepPerThread, KThreads, kKVector>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 1>>,
                                           sequence<1, 2, 2>,
                                           sequence<0, 0, 2>>{});
        }
    }

    // V : [kN0Sub, kVHeaddim], B operand loaded from DRAM into registers then stored to LDS
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
        constexpr index_t kKVector   = GetAlignmentV<Problem>();
        constexpr index_t OtherK     = kKPerBlock / kKVector;

        if constexpr(detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            static_assert((OtherK & (OtherK - 1)) == 0, "Check failed!");
            constexpr index_t KPerThread     = kKVector;
            constexpr index_t KThreads       = OtherK;
            constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                 sequence<KThreads, KPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else
        {
            static_assert((OtherK & (OtherK - 1)) != 0, "Check failed!");
            constexpr index_t KRepPerThread  = (OtherK % 3 == 0) ? 3 : 5;
            constexpr index_t KThreads       = OtherK / KRepPerThread;
            constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                 sequence<KRepPerThread, KThreads, kKVector>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 1>>,
                                           sequence<1, 2, 2>,
                                           sequence<0, 0, 2>>{});
        }
    }

    // Bias : [kM0, kN0] -- use Gemm0 C-tile distribution
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasDramTileDistribution()
    {
        using BlockGemm                       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto bias_block_dstr_encode = BlockGemm::template MakeCBlockDistributionEncode<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kN0>();
        return make_static_tile_distribution(bias_block_dstr_encode);
    }

    // LSE : [kM0] -- row-wise scalar derived from Gemm0/Gemm2 output row distribution
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSETileDistribution()
    {
        // The LSE/delta tile covers the same rows as the Gemm0 C tile.
        // We derive a 1-D distribution by reducing Gemm0's C [kM0, kN0] along N.
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        using CombinedTile =
            decltype(BlockGemm::template MakeCBlockTile<Problem::HstuAttentionTileSetting::kM0,
                                                        Problem::HstuAttentionTileSetting::kN0>());
        const auto f_sum        = [](auto a, auto b) { return a + b; };
        using reduced_tile_type = decltype(block_tile_reduce<typename Problem::CompDataType>(
            CombinedTile{}, sequence<1>{}, f_sum, typename Problem::CompDataType{0}));
        return reduced_tile_type::get_tile_distribution();
    }

    // ML (delta) tile -- same 1-D distribution as LSE
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMLTileDistribution()
    {
        return MakeLSETileDistribution<Problem>();
    }

    // Delta LDS descriptor: flat 1-D buffer of length kM0 for LDS shuffle of delta tile
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDeltaLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0;

        constexpr auto desc = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}), make_tuple(number<1>{}), number<1>{}, number<1>{});
        return desc;
    }

    // -------------------------------------------------------------------------
    // LDS block descriptors
    // -------------------------------------------------------------------------

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        return WG::WarpGemmAttribute::kKPerThread;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetOGradVWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        return WG::WarpGemmAttribute::kKPerThread;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
            return 8;
        else
            return 4;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        if constexpr(GetOGradVWarpGemmKPerThreadSize<Problem>() >= 8)
            return 8;
        else
            return 4;
    }

    // K LDS descriptor: NumKVLdsBuffers * [kN0Sub, kQKHeaddim]
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t NumBuffers =
            kUseTrLoad ? GetNumN0Loops<Problem>() : GetNumKVLdsBuffers<Problem>();
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKPack     = GetSmemKPackK<Problem>();
        constexpr index_t kKVector   = GetAlignmentK<Problem>();

        if constexpr(!detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            constexpr index_t SingleBufferSize = kNPerBlock * kKPerBlock;

            constexpr auto desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumBuffers>{}, number<kNPerBlock>{}, number<kKPerBlock>{}),
                make_tuple(number<SingleBufferSize>{}, number<kKPerBlock>{}, number<1>{}),
                number<kKVector>{},
                number<1>{});
            return transform_tensor_descriptor(
                desc_0,
                make_tuple(
                    make_merge_transform(make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
                    make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
        {
            static_assert(kKVector == kKPack);

            // XOR-swizzled physical layout [NumBuffers, kNPerBlock, kKPerBlock] -- shared
            // with the transposed staging buffers (see MakeSwizzledNativeDesc).
            constexpr auto desc_native =
                MakeSwizzledNativeDesc<Problem, NumBuffers, kNPerBlock, kKPerBlock, kKPack>();

            // Logical view: [NumBuffers * kNPerBlock, kKPerBlock] -- buffers stacked along
            // dim0, matching the other branches and the per-buffer caller slicing.
            return transform_tensor_descriptor(
                desc_native,
                make_tuple(
                    make_merge_transform(make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
                    make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            static_assert(kKVector % kKPack == 0);

            constexpr index_t SingleBufferSize =
                kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector;

            constexpr auto desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<NumBuffers>{},
                                                        number<kKPerBlock / kKVector>{},
                                                        number<kKVector / kKPack>{},
                                                        number<kNPerBlock>{},
                                                        number<kKPack>{}),
                                             make_tuple(number<SingleBufferSize>{},
                                                        number<kNPerBlock * kKVector + kKPack>{},
                                                        number<kNPerBlock * kKPack>{},
                                                        number<kKPack>{},
                                                        number<1>{}),
                                             number<kKPack>{},
                                             number<1>{});

            return transform_tensor_descriptor(
                desc_0,
                make_tuple(
                    make_merge_transform(make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
                    make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                    number<kKVector / kKPack>{},
                                                    number<kKPack>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // V LDS descriptor: NumKVLdsBuffers * [kN0Sub, kVHeaddim]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t NumBuffers = GetNumKVLdsBuffers<Problem>();
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
        constexpr index_t kKPack     = GetSmemKPackV<Problem>();
        constexpr index_t kKVector   = GetAlignmentV<Problem>();

        if constexpr(!detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            constexpr index_t SingleBufferSize = kNPerBlock * kKPerBlock;

            constexpr auto desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumBuffers>{}, number<kNPerBlock>{}, number<kKPerBlock>{}),
                make_tuple(number<SingleBufferSize>{}, number<kKPerBlock>{}, number<1>{}),
                number<kKVector>{},
                number<1>{});
            return transform_tensor_descriptor(
                desc_0,
                make_tuple(
                    make_merge_transform(make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
                    make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else if constexpr(GetOGradVWarpGemmKPerThreadSize<Problem>() >= 8)
        {
            static_assert(kKVector == kKPack);

            // XOR-swizzled physical layout [NumBuffers, kNPerBlock, kKPerBlock] -- shared
            // with the transposed staging buffers (see MakeSwizzledNativeDesc).
            constexpr auto desc_native =
                MakeSwizzledNativeDesc<Problem, NumBuffers, kNPerBlock, kKPerBlock, kKPack>();

            // Logical view: [NumBuffers * kNPerBlock, kKPerBlock] -- buffers stacked along
            // dim0, matching the other branches and the per-buffer caller slicing.
            return transform_tensor_descriptor(
                desc_native,
                make_tuple(
                    make_merge_transform(make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
                    make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            static_assert(kKVector % kKPack == 0);

            constexpr index_t SingleBufferSize =
                kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector;

            constexpr auto desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<NumBuffers>{},
                                                        number<kKPerBlock / kKVector>{},
                                                        number<kKVector / kKPack>{},
                                                        number<kNPerBlock>{},
                                                        number<kKPack>{}),
                                             make_tuple(number<SingleBufferSize>{},
                                                        number<kNPerBlock * kKVector + kKPack>{},
                                                        number<kNPerBlock * kKPack>{},
                                                        number<kKPack>{},
                                                        number<1>{}),
                                             number<kKPack>{},
                                             number<1>{});

            return transform_tensor_descriptor(
                desc_0,
                make_tuple(
                    make_merge_transform(make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
                    make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                    number<kKVector / kKPack>{},
                                                    number<kKPack>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // K^T LDS read descriptor: NumN0Loops * [kQKHeaddim, kN0Sub]
    // Used by Gemm4 (dQ += dS @ K^T) to read K transposed from LDS.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradKTWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        return WG::WarpGemmAttribute::kKPerThread;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackKT()
    {
        if constexpr(GetSGradKTWarpGemmKPerThreadSize<Problem>() >= 8)
            return 8;
        else
            return 4;
    }

    // -------------------------------------------------------------------------
    // Conflict-free physical layout for the transposed staging buffer (kt_lds).
    //
    // The plain physical layout is [NumBuffers, kN, kK] with the
    // kK1 leading dim contiguous. A warp's ds_read (load_tile in Gemm4 dQ) gathering a
    // column across the kN rows re-hits the same bank groups every few rows, causing an
    // LDS bank conflict on the read.
    //
    // Padding the row stride removes the conflict but grows LDS. Instead we apply an XOR
    // swizzle (exactly like MakeKLdsBlockDescriptor's NLdsLayer swizzle for k_lds/v_lds):
    // element_space_size is UNCHANGED (NumBuffers*kN*kK1) so GetSmemSizeKT and the
    // pipeline byte offsets are byte-identical to baseline -- ZERO extra LDS -- while
    // successive rows are scattered across bank groups.
    //
    // The swizzle is baked into the shared 3D physical descriptor BELOW, before the
    // write/read transform chains diverge, so the transposed scalar write view
    // ([kN0Sub, kQKHeaddim]) and the read view ([kQKHeaddim, kN0Sub]) are automatically
    // consistent (both compose over the same swizzled physical descriptor).
    template <typename Problem, index_t NumBuffers, index_t kN, index_t kK, index_t kKPack>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSwizzledNativeDesc()
    {
        using DataType             = remove_cvref_t<typename Problem::QKVDataType>;
        constexpr index_t DataSize = sizeof(DataType);
        // Number of kKPack groups the kN row is scattered into (bank-group span).
#ifdef __gfx950__
        constexpr index_t NLdsLayer = (64 * 4 / kK / DataSize) < 1 ? 1 : (64 * 4 / kK / DataSize);
#else
        constexpr index_t NLdsLayer = (32 * 4 / kK / DataSize) < 1 ? 1 : (32 * 4 / kK / DataSize);
#endif

        // 4D packed physical layout [NumBuffers, kN/NLdsLayer, (kK/kKPack)*NLdsLayer, kKPack].
        constexpr index_t SingleBufferSize = kN * kK;
        constexpr auto desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumBuffers>{},
                                                    number<kN / NLdsLayer>{},
                                                    number<kK / kKPack * NLdsLayer>{},
                                                    number<kKPack>{}),
                                         make_tuple(number<SingleBufferSize>{},
                                                    number<kK * NLdsLayer>{},
                                                    number<kKPack>{},
                                                    number<1>{}),
                                         number<kKPack>{},
                                         number<1>{});

        // XOR-swizzle the (kN/NLdsLayer, kK-group*NLdsLayer) dims -> scatter banks.
        constexpr auto desc_permuted = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                       make_xor_transform(
                           make_tuple(number<kN / NLdsLayer>{}, number<kK / kKPack * NLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

        // Split the kK-group dim back into [kK/kKPack, NLdsLayer].
        constexpr auto desc_split = transform_tensor_descriptor(
            desc_permuted,
            make_tuple(
                make_pass_through_transform(number<NumBuffers>{}),
                make_pass_through_transform(number<kN / NLdsLayer>{}),
                make_unmerge_transform(make_tuple(number<kK / kKPack>{}, number<NLdsLayer>{})),
                make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}));

        // Re-merge to the logical 3D physical view [NumBuffers, kN, kK]:
        //   kN = (kN/NLdsLayer) * NLdsLayer
        //   kK = (kK/kKPack) * kKPack
        return transform_tensor_descriptor(
            desc_split,
            make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<kN / NLdsLayer>{}, number<NLdsLayer>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTLdsReadBlockDescriptor()
    {
        constexpr index_t NumBuffers = GetNumN0Loops<Problem>();
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPack     = GetSmemKPackKT<Problem>();

        // XOR-swizzled physical [NumBuffers, kNPerBlock, kKPerBlock] -- scatters the
        // kNPerBlock (=kQKHeaddim) rows across bank groups so the vectorized load_tile()
        // read (Gemm4 dQ) avoids LDS bank conflicts. The write descriptor composes over
        // the SAME swizzled physical layout, so the transposed scalar store and this read
        // agree on element mapping.
        constexpr auto desc_native =
            MakeSwizzledNativeDesc<Problem, NumBuffers, kNPerBlock, kKPerBlock, kKPack>();

        constexpr auto desc = transform_tensor_descriptor(
            desc_native,
            make_tuple(
                make_pass_through_transform(number<kNPerBlock>{}),
                make_merge_transform(make_tuple(number<NumBuffers>{}, number<kKPerBlock>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTLdsWriteBlockDescriptor()
    {
        constexpr index_t NumBuffers = GetNumN0Loops<Problem>();
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPack     = GetSmemKPackKT<Problem>();

        // Same XOR-swizzled physical layout as MakeKTLdsReadBlockDescriptor -- read and
        // write share kt_lds_ptr, so they MUST agree on the physical element mapping.
        // The store is a transposed, element-by-element (scalar) write, so the swizzle
        // adds no store-side cost while removing the read-side bank conflict.
        constexpr auto desc_native =
            MakeSwizzledNativeDesc<Problem, NumBuffers, kNPerBlock, kKPerBlock, kKPack>();

        return transform_tensor_descriptor(
            desc_native,
            make_tuple(make_merge_transform(make_tuple(number<NumBuffers>{}, number<kKPerBlock>{})),
                       make_pass_through_transform(number<kNPerBlock>{})),
            make_tuple(sequence<0, 2>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // -------------------------------------------------------------------------
    // Shared memory sizing
    // K and V use separate LDS regions (Gemm0 and Gemm2 run in separate loops).
    // -------------------------------------------------------------------------
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        return MakeKLdsBlockDescriptor<Problem, kUseTrLoad>().get_element_space_size() *
               sizeof(typename Problem::QKVDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QKVDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeKT()
    {
        return MakeKTLdsReadBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QKVDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout()
    {
        if constexpr(Problem::kHasDropout)
        {
            using BlockGemm          = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
            constexpr auto config    = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG                 = remove_cvref_t<decltype(config.template at<0>())>;
            constexpr bool IsWG32    = (WG::kM == 32);
            constexpr index_t MWarps = config.template at<1>();
            using BlockGemmShape     = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
            constexpr index_t kMPerBlock   = BlockGemmShape::kM;
            constexpr index_t MIterPerWarp = (!IsWG32 && kMPerBlock > MWarps * WG::kM) ? 2 : 1;
            constexpr index_t kMPerStep    = MIterPerWarp * MWarps * WG::kM;
            // assume the all warps are assigned on dim-M
            constexpr index_t kNPerStep = WG::kN;

            return (kMPerStep + 1) * kNPerStep * sizeof(uint8_t);
        }
        else
        {
            return 0;
        }
    };

    // Total smem: k_lds + v_lds + kt_lds
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        if constexpr(!kUseTrLoad)
        {
            // K region + V region, laid out consecutively.
            // The delta LDS shuffle reuses the K region (delta is consumed before the main loop).
            return GetSmemSizeK<Problem>() + GetSmemSizeV<Problem>() + GetSmemSizeKT<Problem>() +
                   GetSmemSizeDropout<Problem>();
        }
        else
        {
            return GetSmemSizeK<Problem, kUseTrLoad>() + GetSmemSizeV<Problem>() +
                   GetSmemSizeDropout<Problem>();
        }
    }

    // -------------------------------------------------------------------------
    // Block GEMM objects
    // -------------------------------------------------------------------------

    // Gemm0: S = alpha * Q @ K   [kM0, kN0Sub] = [kM0, kQKHeaddim] x [kN0Sub, kQKHeaddim]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm0Gemm2Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN0Sub,
                                   Problem::HstuAttentionTileSetting::kQKHeaddim>,
                          typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<2>{});

#ifdef __gfx950__
                static_assert((WarpGemmM == 16 && WarpGemmK == 32) ||
                                  (WarpGemmM == 32 && WarpGemmK == 16),
                              "Not supported WarpGemm sizes!");
#else
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");
#endif
                return WarpGemmDispatcher<
                    typename Problem::QKVDataType,
                    typename Problem::QKVDataType,
                    typename Problem::GemmAccDataType,
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<0>{}),
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<1>{}),
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<2>{}),
                    true,
                    false,
                    false,
                    WGAttrNumAccessEnum::Single>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            decltype(warp_gemm)>;

        return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
    }

    // Same as GetQKBlockGemm but with kN0 (instead of kN0Sub) as the N tile dimension.
    // This is used as the BlockGemm template argument to BlockDropout::Run() so that
    // kNPerBlock = kN0, ensuring dropout is applied to the full pcomp_tile [kM0, kN0]
    // rather than only the first kN0Sub columns.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKCombinedBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm0Gemm2Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN0,
                                   Problem::HstuAttentionTileSetting::kQKHeaddim>,
                          typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<2>{});

#ifdef __gfx950__
                static_assert((WarpGemmM == 16 && WarpGemmK == 32) ||
                                  (WarpGemmM == 32 && WarpGemmK == 16),
                              "Not supported WarpGemm sizes!");
#else
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");
#endif
                return WarpGemmDispatcher<
                    typename Problem::QKVDataType,
                    typename Problem::QKVDataType,
                    typename Problem::GemmAccDataType,
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<0>{}),
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<1>{}),
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<2>{}),
                    true,
                    false,
                    false,
                    WGAttrNumAccessEnum::Single>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            decltype(warp_gemm)>;

        return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
    }

    // Gemm2: dP = dO @ V   [kM0, kN0Sub] = [kM0, kVHeaddim] x [kN0Sub, kVHeaddim]
    // Uses kVHeaddim as the reduction dimension (V head dim, not QK head dim).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetOGradVBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm0Gemm2Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN0Sub,
                                   Problem::HstuAttentionTileSetting::kVHeaddim>,
                          typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<2>{});

#ifdef __gfx950__
                static_assert((WarpGemmM == 16 && WarpGemmK == 32) ||
                                  (WarpGemmM == 32 && WarpGemmK == 16),
                              "Not supported WarpGemm sizes!");
#else
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");
#endif
                return WarpGemmDispatcher<
                    typename Problem::QKVDataType,
                    typename Problem::QKVDataType,
                    typename Problem::GemmAccDataType,
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<0>{}),
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<1>{}),
                    Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(number<2>{}),
                    true,
                    false,
                    false,
                    WGAttrNumAccessEnum::Single>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            decltype(warp_gemm)>;

        return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
    }

    // Gemm4 single-rep N (used by the epilogue to stride over dQ output)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSGradKTBlockGemmSingleRepN()
    {
        return Problem::HstuAttentionTileSetting::Gemm4BlockWarps::at(number<1>{}) *
               Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<1>{});
    }

    // Gemm4: dQ += alpha * dS @ K^T   [kM0, kQKHeaddim] = [kM0, kN0Sub] x [kQKHeaddim, kN0Sub]
    // A = dS cast to QKVDataType, B = K^T (QKVDataType LDS), C = dQ accumulator
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradKTBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm4Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kQKHeaddim,
                                   Problem::HstuAttentionTileSetting::kN0Sub>,
                          typename Problem::HstuAttentionTileSetting::Gemm4BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm4WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<2>{});

#ifdef __gfx950__
                static_assert((WarpGemmM == 16 && WarpGemmK == 32) ||
                                  (WarpGemmM == 32 && WarpGemmK == 16),
                              "Not supported WarpGemm sizes!");
#else
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");
#endif

                if constexpr((WarpGemmM == 16 && WarpGemmK == 32) ||
                             (WarpGemmM == 32 && WarpGemmK == 16))
                    return WarpGemmDispatcher<
                        typename Problem::QKVDataType,
                        typename Problem::QKVDataType,
                        typename Problem::GemmAccDataType,
                        Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<2>{}),
                        true,
                        false,
                        false,
                        WGAttrNumAccessEnum::Double>{};
                else
                    return WarpGemmDispatcher<
                        typename Problem::QKVDataType,
                        typename Problem::QKVDataType,
                        typename Problem::GemmAccDataType,
                        Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<2>{}),
                        true,
                        false,
                        false,
                        WGAttrNumAccessEnum::Single>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm4BlockWarps,
            decltype(warp_gemm)>;

        if constexpr(!kUseTrLoad)
        {
            return BlockGemmARegBSmemCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
        }
        else
        {
            return BlockGemmARegBSmemTrLoadCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
        }
    }
};

} // namespace ck_tile
