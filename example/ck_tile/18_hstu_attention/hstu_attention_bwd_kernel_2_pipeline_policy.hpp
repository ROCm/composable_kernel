// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_problem.hpp>
#include <ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp>

#include "block_gemm_areg_bsmem_creg_v2_hack_1.hpp"
#include "block_gemm_areg_bsmem_trload_creg_v2_hack_1.hpp"
#include "block_gemm_asmem_breg_creg_v1_hack.hpp"

#include "hstu_attention_kernel_util.hpp"

namespace ck_tile {

struct HstuAttentionBwdKernel2PipelinePolicy
{
    // Gemm0, Gemm2 use m0_loop, which unrolls the Gemm along kM0
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumM0Loops()
    {
        constexpr index_t m0_loops =
            Problem::HstuAttentionTileSetting::kM0 / Problem::HstuAttentionTileSetting::kM0Sub;

        return m0_loops;
    }

    // Gemm1/Gemm3 all use k1_loop, which unrolls the Gemm along kM0, kK1 reuse kM0Sub at present
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumK1Loops()
    {
        constexpr index_t k1_loops =
            Problem::HstuAttentionTileSetting::kM0 / Problem::HstuAttentionTileSetting::kK1;

        return k1_loops;
    }

    // Number of Lds slots for q_lds, do_lds
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumQOGradLdsBuffers()
    {
        return 2;
    }

    // -------------------------------------------------------------------------
    // Alignment helpers (vector load widths)
    // -------------------------------------------------------------------------

    // K alignment -- based on [kN0, kQKHeaddim] tile
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QKVDataType);
        using BlockGemm                 = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    // Q alignment -- based on [kM0Sub, kQKHeaddim] tile
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        return Problem::GetQDramTileAccessMaxVectorSize();
    }

    // V alignment -- same tile shape as K
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        return GetAlignmentK<Problem>();
    }

    // dO alignment -- based ib [kM0Sub, kVHeaddim] tile
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOGrad()
    {
        return Problem::GetOGradDramTileAccessMaxVectorSize();
    }

    // dK alignment -- same tile shape as K (output [kN0, kQKHeaddim])
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentKGrad()
    {
        return GetAlignmentK<Problem>();
    }

    // dV alignment -- same tile shape as V (output [kN0, kVHeaddim])
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentVGrad()
    {
        return GetAlignmentV<Problem>();
    }

    // -------------------------------------------------------------------------
    // DRAM tile distributions
    // -------------------------------------------------------------------------

    // Q DRAM distribution -- [kM0Sub, kQKHeaddim], loaded sub-tile by sub-tile
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;

        constexpr index_t kKVector = GetAlignmentQ<Problem>();
        constexpr index_t OtherK   = kKPerBlock / kKVector;

        if constexpr(detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            constexpr index_t KPerThread = kKVector;
            constexpr index_t KThreads   = OtherK;

            constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

            static_assert(MPerThread > 0, "Check failed!");

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
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
            constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

            static_assert(MPerThread > 0, "Check failed!");

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                                 sequence<KRepPerThread, KThreads, kKVector>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 1>>,
                                           sequence<1, 2, 2>,
                                           sequence<0, 0, 2>>{});
        }
    }

    // dO DRAM distribution -- [kM0Sub, kQKHeaddim], loaded sub-tile by sub-tile
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;

        constexpr index_t kKVector = GetAlignmentOGrad<Problem>();
        constexpr index_t OtherK   = kKPerBlock / kKVector;

        if constexpr(detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            constexpr index_t KPerThread = kKVector;
            constexpr index_t KThreads   = OtherK;

            constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

            static_assert(MPerThread > 0, "Check failed!");

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
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
            constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

            static_assert(MPerThread > 0, "Check failed!");

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                                 sequence<KRepPerThread, KThreads, kKVector>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 1>>,
                                           sequence<1, 2, 2>,
                                           sequence<0, 0, 2>>{});
        }
    }

    // K : [kN0, kQKHeaddim], register-resident B operand
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        return BlockGemm::template MakeBBlockTileDistribution<
            Problem::HstuAttentionTileSetting::kN0,
            Problem::HstuAttentionTileSetting::kQKHeaddim>();
    }

    // V : [kN0, kVHeaddim], register-resident B operand
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
        return BlockGemm::template MakeBBlockTileDistribution<
            Problem::HstuAttentionTileSetting::kN0,
            Problem::HstuAttentionTileSetting::kVHeaddim>();
    }

    // PT register tile distribution
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePTRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        return BlockGemm::template MakeABlockTileDistribution<
            Problem::HstuAttentionTileSetting::kN0,
            Problem::HstuAttentionTileSetting::kM0>();
    }

    // SGradT register tile distribution
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradTRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        return BlockGemm::template MakeABlockTileDistribution<
            Problem::HstuAttentionTileSetting::kN0,
            Problem::HstuAttentionTileSetting::kM0>();
    }

    // Bias -- [kM0Sub, kN0], use the C-tile distribution of Gemm0
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasDramTileDistribution()
    {
        using BlockGemm                       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto bias_block_dstr_encode = BlockGemm::template MakeCBlockDistributionEncode<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kN0>();
        return make_static_tile_distribution(bias_block_dstr_encode);
    }

    // LSE -- [kM0], 1-D row scalar derived by reducing Gemm0 C-tile along N
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSETileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        auto sacc_tile =
            BlockGemm::template MakeCBlockTile<Problem::HstuAttentionTileSetting::kM0,
                                               Problem::HstuAttentionTileSetting::kN0>();
        const auto f_sum        = [](auto a, auto b) { return a + b; };
        using reduced_tile_type = decltype(block_tile_reduce<typename Problem::CompDataType>(
            sacc_tile, sequence<1>{}, f_sum, typename Problem::CompDataType{0}));
        return reduced_tile_type::get_tile_distribution();
    }

    // Delta (D[sq]) -- same 1-D distribution as LSE
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDeltaTileDistribution()
    {
        return MakeLSETileDistribution<Problem>();
    }

    // -------------------------------------------------------------------------
    // LDS smem sizing helpers
    // -------------------------------------------------------------------------

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem, kUseTrLoad>().get_element_space_size() *
               sizeof(typename Problem::QKVDataType);
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGrad()
    {
        return MakeOGradLdsBlockDescriptor<Problem, kUseTrLoad>().get_element_space_size() *
               sizeof(typename Problem::QKVDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQT()
    {
        return MakeQTLdsReadBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QKVDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGradT()
    {
        return MakeOGradTLdsReadBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QKVDataType);
    }

    // Total smem: q_lds + do_lds, pt_lds, dst_lds) + qt_lds + dot_lds
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        if constexpr(!kUseTrLoad)
        {
            return GetSmemSizeQ<Problem>() + GetSmemSizeOGrad<Problem>() +
                   GetSmemSizeQT<Problem>() + GetSmemSizeOGradT<Problem>();
        }
        else
        {
            return GetSmemSizeQ<Problem, kUseTrLoad>() + GetSmemSizeOGrad<Problem, kUseTrLoad>();
        }
    }

    // -------------------------------------------------------------------------
    // WarpGemm K-per-thread helpers (used for LDS bank-conflict-free layouts)
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
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        return max(GetQKWarpGemmKPerThreadSize<Problem>(), GetAlignmentQ<Problem>());
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
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackOGrad()
    {
        return max(GetOGradVWarpGemmKPerThreadSize<Problem>(), GetAlignmentOGrad<Problem>());
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPTOGradTWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        return WG::WarpGemmAttribute::kKPerThread;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackOGradT()
    {
        return max(GetPTOGradTWarpGemmKPerThreadSize<Problem>(), GetAlignmentOGrad<Problem>());
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradTQTWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        return WG::WarpGemmAttribute::kKPerThread;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQT()
    {
        return max(GetSGradTQTWarpGemmKPerThreadSize<Problem>(), GetAlignmentQ<Problem>());
    }

    // -------------------------------------------------------------------------
    // Conflict-free physical layout for the transposed staging buffers
    // (qt_lds / dot_lds).
    //
    // The plain physical layout is [NumBuffers, kN(=headdim), kK1] with the kK1
    // leading dim contiguous. kK1 = 16 bf16 = 32 B = 8 LDS banks and the kN row
    // stride is also 16 elems = 8 banks, so a warp's ds_read gathering a column
    // across the kN rows re-hits the same 8 banks every 4 rows (~4-way conflict).
    //
    // Padding the row stride removes the conflict but grows LDS, and kernel-2 is
    // pinned to 2 blocks/CU (__launch_bounds__(kBlockSize,2)); any effective pad
    // pushes 2*SmemSize over the 64 KB/CU budget and drops occupancy to 1 block/CU,
    // which costs more than the conflict saves. Instead we apply an XOR swizzle
    // (exactly like MakeQLdsBlockDescriptor's NLdsLayer swizzle for q_lds/do_lds):
    // element_space_size is UNCHANGED (NumBuffers*kN*kK1) so GetSmemSize and the
    // pipeline byte offsets are byte-identical to baseline -- ZERO extra LDS, no
    // occupancy change -- while successive rows are scattered across bank groups.
    //
    // The swizzle is baked into the shared 3D physical descriptor BELOW, before the
    // write/read transform chains diverge, so the write ([kM0,kK0]) and read
    // ([kN,kK]) views are automatically consistent (both compose over the same
    // swizzled physical descriptor).
    template <typename Problem, index_t NumBuffers, index_t kN, index_t kK, index_t kKPack>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSwizzledNativeDesc()
    {
        constexpr index_t ElementBytes = sizeof(typename Problem::QKVDataType);

        // Number of kKPack groups the kN row is scattered into (bank-group span).
#ifdef __gfx950__
        constexpr index_t BankSpanBytes = 64 * 4;
#else
        constexpr index_t BankSpanBytes = 32 * 4;
#endif

        constexpr index_t SingleBufferSize = kN * kK;

        if constexpr(kK * ElementBytes < BankSpanBytes)
        {
            constexpr index_t NLdsLayer = BankSpanBytes / (kK * ElementBytes);

            // 4D packed physical layout [NumBuffers, kN/NLdsLayer, (kK/kKPack)*NLdsLayer, kKPack].
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
                           make_xor_transform(make_tuple(number<kN / NLdsLayer>{},
                                                         number<kK / kKPack * NLdsLayer>{})),
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
        else
        {
            // 4D packed physical layout [NumBuffers, kN, kK/kKPack, kKPack].
            constexpr auto desc_0 = make_naive_tensor_descriptor(
                make_tuple(
                    number<NumBuffers>{}, number<kN>{}, number<kK / kKPack>{}, number<kKPack>{}),
                make_tuple(number<SingleBufferSize>{}, number<kK>{}, number<kKPack>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            // XOR-swizzle the (kN, kK-group) dims -> scatter banks.
            constexpr auto desc_permuted = transform_tensor_descriptor(
                desc_0,
                make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                           make_xor_transform(make_tuple(number<kN>{}, number<kK / kKPack>{})),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            // Re-merge to the logical 3D physical view [NumBuffers, kN, kK]:
            //   kK = (kK/kKPack) * kKPack
            return transform_tensor_descriptor(
                desc_permuted,
                make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                           make_pass_through_transform(number<kN>{}),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
        }
    }

    // -------------------------------------------------------------------------
    // LDS block descriptors
    // -------------------------------------------------------------------------

    // q_lds write/read descriptor: NumBuffers * [kM0Sub, kQKHeaddim]
    template <typename Problem,
              index_t NumBuffers,
              index_t kNPerBlock,
              index_t kKPerBlock,
              index_t kKPack,
              index_t kKVector,
              index_t WarpGemmKPerThread,
              bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQOGradLdsBlockDescriptor()
    {
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
        else if constexpr(WarpGemmKPerThread == kKPack)
        {
            // In the trload pipeline this q_lds/do_lds buffer is read BOTH normally
            // (Gemm0/Gemm2 A operand) and transposed (Gemm3/Gemm1 via ds_read_b64_tr).
            // Profiling shows the transpose read is the dominant LDS-conflict source (it is 2x
            // the normal-read count in kernel 2) and it prefers a plain (contiguous) layout.
            // Use a plain physical layout for the shared trload buffer (same element space ->
            // GetSmemSize/byte offsets unchanged), and keep the XOR swizzle for the non-trload
            // buffer whose read is normal-only.
            constexpr auto desc_native = [] {
                if constexpr(kUseTrLoad)
                {
                    if constexpr(kKPerBlock <= 16)
                    {
                        return make_naive_tensor_descriptor(
                            make_tuple(
                                number<NumBuffers>{}, number<kNPerBlock>{}, number<kKPerBlock>{}),
                            make_tuple(number<kNPerBlock * kKPerBlock>{},
                                       number<kKPerBlock>{},
                                       number<1>{}),
                            number<kKPack>{},
                            number<1>{});
                    }
                    else
                    {
                        // With trload read,  16 threads per cycle access the [4Tl, 4Tm*4E]
                        // block and cross-bar transpose it to [4E, 4Tm*4Tl] layout suitable for
                        // mfma, we want to ensure 16T * 2 dwords to exactly hit 32 banks (for
                        // KPerBlock = 16, 4 * 16 * sizeof(bf16) = 32 banks mapped by 4 rows;
                        // and actual hit 16 Threads * 2 banks/per-inst is also 32 banks )
                        static_assert(kKPerBlock % 16 == 0,
                                      "kKPerBlock should be a multiplier of 16!");

                        constexpr auto desc_native_0 = make_naive_tensor_descriptor(
                            make_tuple(number<NumBuffers>{},
                                       number<kKPerBlock / 16>{},
                                       number<kNPerBlock>{},
                                       number<16>{}),
                            make_tuple(number<kNPerBlock * kKPerBlock>{},
                                       number<kNPerBlock * 16>{},
                                       number<16>{},
                                       number<1>{}),
                            number<kKPack>{},
                            number<1>{});

                        return transform_tensor_descriptor(
                            desc_native_0,
                            make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                                       make_pass_through_transform(number<kNPerBlock>{}),
                                       make_merge_transform(
                                           make_tuple(number<kKPerBlock / 16>{}, number<16>{}))),
                            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
                    }
                }
                else
                    return MakeSwizzledNativeDesc<Problem,
                                                  NumBuffers,
                                                  kNPerBlock,
                                                  kKPerBlock,
                                                  kKPack>();
            }();

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
    }

    // q_lds write/read descriptor
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock         = Problem::HstuAttentionTileSetting::kM0Sub;
        constexpr index_t kKPerBlock         = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKPack             = GetSmemKPackQ<Problem>();
        constexpr index_t kKVector           = GetAlignmentQ<Problem>();
        constexpr index_t WarpGemmKPerThread = GetQKWarpGemmKPerThreadSize<Problem>();

        if constexpr(kUseTrLoad)
        {
            constexpr index_t NumBuffers = GetNumM0Loops<Problem>();
            return MakeQOGradLdsBlockDescriptor<Problem,
                                                NumBuffers,
                                                kNPerBlock,
                                                kKPerBlock,
                                                kKPack,
                                                kKVector,
                                                WarpGemmKPerThread,
                                                true /*kUseTrLoad*/>();
        }
        else
        {
            constexpr index_t NumBuffers = GetNumQOGradLdsBuffers<Problem>();
            return MakeQOGradLdsBlockDescriptor<Problem,
                                                NumBuffers,
                                                kNPerBlock,
                                                kKPerBlock,
                                                kKPack,
                                                kKVector,
                                                WarpGemmKPerThread>();
        }
    }

    // do_lds write/read descriptor
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock         = Problem::HstuAttentionTileSetting::kM0Sub;
        constexpr index_t kKPerBlock         = Problem::HstuAttentionTileSetting::kVHeaddim;
        constexpr index_t kKPack             = GetSmemKPackOGrad<Problem>();
        constexpr index_t kKVector           = GetAlignmentOGrad<Problem>();
        constexpr index_t WarpGemmKPerThread = GetOGradVWarpGemmKPerThreadSize<Problem>();

        if constexpr(kUseTrLoad)
        {
            constexpr index_t NumBuffers = GetNumM0Loops<Problem>();
            return MakeQOGradLdsBlockDescriptor<Problem,
                                                NumBuffers,
                                                kNPerBlock,
                                                kKPerBlock,
                                                kKPack,
                                                kKVector,
                                                WarpGemmKPerThread,
                                                true /*kUseTrLoad*/>();
        }
        else
        {
            constexpr index_t NumBuffers = GetNumQOGradLdsBuffers<Problem>();
            return MakeQOGradLdsBlockDescriptor<Problem,
                                                NumBuffers,
                                                kNPerBlock,
                                                kKPerBlock,
                                                kKPack,
                                                kKVector,
                                                WarpGemmKPerThread>();
        }
    }

#if !HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE
    template <typename Problem, typename WarpGemm, index_t NumBuffers, index_t kN, index_t kK>
    CK_TILE_HOST_DEVICE static constexpr auto MakeWarpGemmAwareBLdsReadBlockNativeDesc()
    {
        constexpr index_t kKPack = WarpGemm::WarpGemmAttribute::Impl::kABKPerLane;

        constexpr index_t ElementBytes = sizeof(typename Problem::QKVDataType);

        // Number of kKPack groups the kN row is scattered into (bank-group span).
#ifdef __gfx950__
        constexpr index_t MaxNLdsLayer =
            (64 * 4 / kK / ElementBytes) < 1 ? 1 : (64 * 4 / kK / ElementBytes);
#else
        constexpr index_t MaxNLdsLayer =
            (32 * 4 / kK / ElementBytes) < 1 ? 1 : (32 * 4 / kK / ElementBytes);
#endif

        constexpr index_t NThreads = WarpGemm::WarpGemmAttribute::Impl::kBNLane;

        constexpr index_t NLdsLayer = min(kN / NThreads, MaxNLdsLayer);

        // 4D packed physical layout [NumBuffers, NThreads, (kK/kKPack)*NLdsLayer, kKPack].
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

        // XOR-swizzle the (NThreads, kK-group*NLdsLayer) dims -> scatter banks.
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
        //   kN = NLdsLayer * NThreads
        //   kK = (kK/kKPack) * kKPack
        return transform_tensor_descriptor(
            desc_split,
            make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<NLdsLayer>{}, number<kN / NLdsLayer>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0>{}, sequence<3, 1>{}, sequence<2, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQTLdsWriteBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t NumBuffers = GetNumK1Loops<Problem>();
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        constexpr auto desc_native = MakeWarpGemmAwareBLdsReadBlockNativeDesc<Problem,
                                                                              WG,
                                                                              NumBuffers,
                                                                              kNPerBlock,
                                                                              kKPerBlock>();
        // the same native tensor desc as the ReadBlockDescriptor, but transposed tensor view
        return transform_tensor_descriptor(
            desc_native,
            make_tuple(make_merge_transform(make_tuple(number<NumBuffers>{}, number<kKPerBlock>{})),
                       make_pass_through_transform(number<kNPerBlock>{})),
            make_tuple(sequence<0, 2>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradTLdsWriteBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t NumBuffers = GetNumK1Loops<Problem>();
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        constexpr auto desc_native = MakeWarpGemmAwareBLdsReadBlockNativeDesc<Problem,
                                                                              WG,
                                                                              NumBuffers,
                                                                              kNPerBlock,
                                                                              kKPerBlock>();

        // the same native tensor desc as the ReadBlockDescriptor, but transposed tensor view
        return transform_tensor_descriptor(
            desc_native,
            make_tuple(make_merge_transform(make_tuple(number<NumBuffers>{}, number<kKPerBlock>{})),
                       make_pass_through_transform(number<kNPerBlock>{})),
            make_tuple(sequence<0, 2>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQTLdsReadBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t NumBuffers = GetNumK1Loops<Problem>();
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        constexpr auto desc_native = MakeWarpGemmAwareBLdsReadBlockNativeDesc<Problem,
                                                                              WG,
                                                                              NumBuffers,
                                                                              kNPerBlock,
                                                                              kKPerBlock>();

        // merge: NumK1Loops * [kQKHeaddim, kK1] -> [kQKHeaddim, kM0]
        return transform_tensor_descriptor(
            desc_native,
            make_tuple(
                make_pass_through_transform(number<kNPerBlock>{}),
                make_merge_transform(make_tuple(number<NumBuffers>{}, number<kKPerBlock>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradTLdsReadBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t NumBuffers = GetNumK1Loops<Problem>();
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        constexpr auto desc_native = MakeWarpGemmAwareBLdsReadBlockNativeDesc<Problem,
                                                                              WG,
                                                                              NumBuffers,
                                                                              kNPerBlock,
                                                                              kKPerBlock>();

        // merge: NumK1Loops * [kQKHeaddim, kK1] -> [kQKHeaddim, kM0]
        return transform_tensor_descriptor(
            desc_native,
            make_tuple(
                make_pass_through_transform(number<kNPerBlock>{}),
                make_merge_transform(make_tuple(number<NumBuffers>{}, number<kKPerBlock>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }
#endif

    // -------------------------------------------------------------------------
    // Block GEMM objects
    // -------------------------------------------------------------------------

    // Gemm0: S = Q_lds @ K_reg    [kM0Sub, kN0] = [kM0Sub, kQKHeaddim] x [kN0, kQKHeaddim]
    // Gemm2: dP = dO_lds @ V_reg  [kM0Sub, kN0] = [kM0Sub, kVHeaddim] x [kN0, kVHeaddim]
    // A = Q/dO from LDS (A-smem), B = K/V register-resident (B-reg)
    // -> BlockGemmASmemBRegCRegV1
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm0Gemm2Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0Sub,
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
                    false, // not CTransposed
                    false,
                    false,
                    WGAttrNumAccessEnum::Single>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using BlockGemmPolicy = BlockGemmASmemBRegCRegV1CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            decltype(warp_gemm)>;

        return BlockGemmASmemBRegCRegV1Hack<GemmProblem, BlockGemmPolicy>{};
    }

    // Same as GetQKBlockGemm but with kM0 (instead of kM0Sub) as the M tile dimension.
    // This is used as the BlockGemm template argument to BlockDropout::Run() so that
    // kMPerBlock = kM0, ensuring dropout is applied to the full pcomp_tile [kM0, kN0]
    // rather than only the first kM0Sub rows.
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
                    false, // not CTransposed
                    false,
                    false,
                    WGAttrNumAccessEnum::Single>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using BlockGemmPolicy = BlockGemmASmemBRegCRegV1CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            decltype(warp_gemm)>;

        return BlockGemmASmemBRegCRegV1Hack<GemmProblem, BlockGemmPolicy>{};
    }

    // Gemm2: dP = dO @ V   [kM0Sub, kN0] = [kM0Sub, kVHeaddim] x [kN0, kVHeaddim]
    // Uses kVHeaddim as the reduction dimension (V head dim, not QK head dim).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetOGradVBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm0Gemm2Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0Sub,
                                   Problem::HstuAttentionTileSetting::kN0,
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
                    false, // not CTransposed
                    false,
                    false,
                    WGAttrNumAccessEnum::Single>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using BlockGemmPolicy = BlockGemmASmemBRegCRegV1CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            decltype(warp_gemm)>;

        return BlockGemmASmemBRegCRegV1Hack<GemmProblem, BlockGemmPolicy>{};
    }

    // is_target_warptile_16_32 == true selects the 16x16x32 (native mfma, WGAttrNumAccessEnum::
    // Double) A operand for Gemm1. Its per-lane A register count is twice that of the incoming
    // 16x16 C fragment, so each K=32 A tile is assembled from two consecutive 16x16 C
    // sub-tiles. is_target_warptile_16_32 == false keeps the original 16x16x16 (1:1)
    // transpose-free reuse.
    template <typename Problem,
              bool is_target_warptile_16_32,
              typename PTOutTensor,
              typename PInTensor>
    CK_TILE_DEVICE static constexpr void PTFromGemm0CToGemm1A(PTOutTensor& pt_out,
                                                              const PInTensor& p_in)
    {
#if defined(__gfx125__)
        pt_out.get_thread_buffer() = p_in.get_thread_buffer();
#else
        if constexpr(Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}) == 16)
        {
            using BlockGemm       = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr index_t MWarp =
                Problem::HstuAttentionTileSetting::Gemm1BlockWarps::at(number<0>{});

            constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kN0;
            constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kM0;

            constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
            constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

            using AWarpDstr = typename WarpGemm::AWarpDstr;
            using CWarpDstr = typename WarpGemm::CWarpDstr;
            auto p_warp_tensor =
                make_static_distributed_tensor<typename Problem::QKVDataType>(CWarpDstr{});
            auto pt_warp_tensor =
                make_static_distributed_tensor<typename Problem::QKVDataType>(AWarpDstr{});

            constexpr auto a_warp_y_lengths =
                to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

            constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            if constexpr(is_target_warptile_16_32)
            {
                // Number of 16x16 C sub-tiles packed along K into one 16x16x32 A tile (== 2),
                // and the per-lane register count of a single 16x16 fragment (== C fragment
                // size).
                constexpr index_t NumKSub = WarpGemm::kK / 16;
                constexpr index_t kSubPerThread =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_element_space_size();

                static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
                    constexpr auto kIter = number<km[number<0>{}]>{};
                    constexpr auto mIter = number<km[number<1>{}]>{};

                    static_for<0, NumKSub, 1>{}([&](auto kSub) {
                        p_warp_tensor.get_thread_buffer() = p_in.get_y_sliced_thread_data(
                            merge_sequences(sequence<kIter * NumKSub + kSub, mIter>{},
                                            c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // C->A transpose is the register identity for one 16x16 fragment; place
                        // it into sub-access kSub of the K=32 A tile (Double packs the accesses
                        // as [sub0(kSubPerThread), sub1(kSubPerThread)] in the thread buffer).
                        static_for<0, kSubPerThread, 1>{}([&](auto i) {
                            pt_warp_tensor.get_thread_buffer()(number<kSub * kSubPerThread + i>{}) =
                                p_warp_tensor.get_thread_buffer()(number<i>{});
                        });
                    });

                    pt_out.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
                        pt_warp_tensor.get_thread_buffer());
                });
            }
            else
            {
                static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
                    constexpr auto kIter              = number<km[number<0>{}]>{};
                    constexpr auto mIter              = number<km[number<1>{}]>{};
                    p_warp_tensor.get_thread_buffer() = p_in.get_y_sliced_thread_data(
                        merge_sequences(sequence<kIter, mIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    pt_warp_tensor.get_thread_buffer() = p_warp_tensor.get_thread_buffer();

                    pt_out.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
                        pt_warp_tensor.get_thread_buffer());
                });
            }
        }
        else
        {
            pt_out.get_thread_buffer() = p_in.get_thread_buffer();
        }
#endif // defined(__gfx125__)
    }

    // is_target_warptile_16_32 has the same meaning as in PTFromGemm0CToGemm1A, but for Gemm3.
    template <typename Problem,
              bool is_target_warptile_16_32,
              typename SGradTOutTensor,
              typename SGradInTensor>
    CK_TILE_DEVICE static constexpr void SGradTFromGemm2CToGemm3A(SGradTOutTensor& dst_out,
                                                                  const SGradInTensor& ds_in)
    {
#if defined(__gfx125__)
        dst_out.get_thread_buffer() = ds_in.get_thread_buffer();
#else
        if constexpr(Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<0>{}) == 16)
        {
            using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr index_t MWarp =
                Problem::HstuAttentionTileSetting::Gemm3BlockWarps::at(number<0>{});

            constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kN0;
            constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kM0;

            constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
            constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

            using AWarpDstr = typename WarpGemm::AWarpDstr;
            using CWarpDstr = typename WarpGemm::CWarpDstr;
            auto ds_warp_tensor =
                make_static_distributed_tensor<typename Problem::QKVDataType>(CWarpDstr{});
            auto dst_warp_tensor =
                make_static_distributed_tensor<typename Problem::QKVDataType>(AWarpDstr{});

            constexpr auto a_warp_y_lengths =
                to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

            constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            if constexpr(is_target_warptile_16_32)
            {
                constexpr index_t NumKSub = WarpGemm::kK / 16;
                constexpr index_t kSubPerThread =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_element_space_size();

                static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
                    constexpr auto kIter = number<km[number<0>{}]>{};
                    constexpr auto mIter = number<km[number<1>{}]>{};

                    static_for<0, NumKSub, 1>{}([&](auto kSub) {
                        ds_warp_tensor.get_thread_buffer() = ds_in.get_y_sliced_thread_data(
                            merge_sequences(sequence<kIter * NumKSub + kSub, mIter>{},
                                            c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // C->A transpose is the register identity for one 16x16 fragment; place
                        // it into sub-access kSub of the K=32 A tile (Double packs the accesses
                        // as [sub0(kSubPerThread), sub1(kSubPerThread)] in the thread buffer).
                        static_for<0, kSubPerThread, 1>{}([&](auto i) {
                            dst_warp_tensor.get_thread_buffer()(
                                number<kSub * kSubPerThread + i>{}) =
                                ds_warp_tensor.get_thread_buffer()(number<i>{});
                        });
                    });

                    dst_out.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
                        dst_warp_tensor.get_thread_buffer());
                });
            }
            else
            {
                static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
                    constexpr auto kIter               = number<km[number<0>{}]>{};
                    constexpr auto mIter               = number<km[number<1>{}]>{};
                    ds_warp_tensor.get_thread_buffer() = ds_in.get_y_sliced_thread_data(
                        merge_sequences(sequence<kIter, mIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    dst_warp_tensor.get_thread_buffer() = ds_warp_tensor.get_thread_buffer();
                    dst_out.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
                        dst_warp_tensor.get_thread_buffer());
                });
            }
        }
        else
        {
            dst_out.get_thread_buffer() = ds_in.get_thread_buffer();
        }
#endif // defined(__gfx125__)
    }

    // -------------------------------------------------------------------------
    // Gemm1 single-rep N (used by the epilogue to stride over dV output)
    // -------------------------------------------------------------------------
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetPTOGradTBlockGemmSingleRepN()
    {
        return Problem::HstuAttentionTileSetting::Gemm1BlockWarps::at(number<1>{}) *
               Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{});
    }

    // Gemm1: dV += P^T @ dO^T   [kN0, kVHeaddim] = [kN0, kK1] x [kVHeaddim, kK1]
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetPTOGradTBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm1Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kN0,
                                   Problem::HstuAttentionTileSetting::kVHeaddim,
                                   Problem::HstuAttentionTileSetting::kK1>,
                          typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm1WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{});

#ifdef __gfx950__
                // Gemm1 (dV = P^T @ dO^T) reuses Gemm0's C output as its A input via the
                // transpose-free register copy in PTFromGemm0CToGemm1A, which requires the mfma
                // A- and C-operand per-lane sizes to coincide (i.e. WarpGemmK == 16). gfx950
                // still provides the 16x16x16 fp16 mfma, so allow it here in addition to
                // 16x16x32.
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
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
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{}),
                        true,
                        false,
                        false,
                        WGAttrNumAccessEnum::Double>{};
                else
                    return WarpGemmDispatcher<
                        typename Problem::QKVDataType,
                        typename Problem::QKVDataType,
                        typename Problem::GemmAccDataType,
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{}),
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
            typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
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

    // -------------------------------------------------------------------------
    // Gemm3 single-rep N (used by the epilogue to stride over dK output)
    // -------------------------------------------------------------------------
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSGradTQTBlockGemmSingleRepN()
    {
        return Problem::HstuAttentionTileSetting::Gemm3BlockWarps::at(number<1>{}) *
               Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<1>{});
    }

    // Gemm3: dK += dS^T @ Q^T   [kN0, kQKHeaddim] = [kN0, kK1] x [kQKHeaddim, kK1]
    // Uses Gemm3BlockWarps/Gemm3WarpTile which may differ from Gemm1's configuration.
    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradTQTBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm3Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kN0,
                                   Problem::HstuAttentionTileSetting::kQKHeaddim,
                                   Problem::HstuAttentionTileSetting::kK1>,
                          typename Problem::HstuAttentionTileSetting::Gemm3BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm3WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<2>{});

#ifdef __gfx950__
                // Gemm3 (dK = dS^T @ Q^T) reuses Gemm2's C output as its A input via the
                // transpose-free register copy in SGradTFromGemm2CToGemm3A, which requires the
                // mfma A- and C-operand per-lane sizes to coincide (i.e. WarpGemmK == 16).
                // gfx950 still provides the 16x16x16 fp16 mfma, so allow it here in addition to
                // 16x16x32.
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
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
                        Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<2>{}),
                        true,
                        false,
                        false,
                        WGAttrNumAccessEnum::Double>{};
                else
                    return WarpGemmDispatcher<
                        typename Problem::QKVDataType,
                        typename Problem::QKVDataType,
                        typename Problem::GemmAccDataType,
                        Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<2>{}),
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
            typename Problem::HstuAttentionTileSetting::Gemm3BlockWarps,
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
