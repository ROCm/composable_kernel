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

#include "hstu_attention_config.hpp"
#include "hstu_attention_kernel_util.hpp"
#include "hstu_attention_pipeline_policy_helper.hpp"

namespace ck_tile {

struct HstuAttentionFwdPipelineQRKSVSPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumKLdsBuffers()
    {
        return 2;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumVLdsBuffers()
    {
        return 2;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeABlockTileDistribution<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetQKWarpGemmBScalarPerVector()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        using BEncoding = typename WG::BWarpDstrEncoding;
        return BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];
    };

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetPVTWarpGemmBScalarPerVector()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, kUseTrLoad>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        using BEncoding = typename WG::BWarpDstrEncoding;

        if constexpr(kUseTrLoad)
        {
            using BEncodingForTrLoad =
                typename InputTileDistributionTraits<BEncoding, typename BlockGemm::BDataType>::
                    TransposedDstrEncode;

            return BEncodingForTrLoad::detail::ys_lengths_[BEncoding::NDimY - 1];
        }
        else
        {
            return BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];
        }
    };

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBiasDramTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        constexpr auto bias_block_dstr_encode = BlockGemm::template MakeCBlockDistributionEncode<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kN0>();
        constexpr auto bias_block_dstr = make_static_tile_distribution(bias_block_dstr_encode);

        return bias_block_dstr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetAlignmentBias()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QKVDataType);

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        return max(GetQKWarpGemmBScalarPerVector<Problem>(), GetAlignmentK<Problem>());
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        return Problem::GetKDramTileAccessMaxVectorSize();
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        return max(GetPVTWarpGemmBScalarPerVector<Problem, kUseTrLoad>(),
                   GetAlignmentV<Problem, kUseTrLoad>());
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        // special consideration when shuffling is required before storing V to LDS
        if constexpr(!kUseTrLoad)
        {
            using VDataType = remove_cvref_t<typename Problem::QKVDataType>;

            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
            constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            constexpr index_t kMaxVecLoad = Problem::GetVDramTileAccessMaxVectorSize();
            constexpr index_t kMinVecLoad = 4 / sizeof(VDataType);

            // try to avoid writing sub-dword to LDS due to poor performance
            constexpr index_t kVecLoad = ((ElemPerThread / kMaxVecLoad) >= kMinVecLoad)
                                             ? kMaxVecLoad
                                             : (ElemPerThread / kMinVecLoad);

            if constexpr(kVecLoad == 4 * (ElemPerThread / kVecLoad))
                return kVecLoad / 2;

            return kVecLoad;
        }
        else
        {
            return Problem::GetVDramTileAccessMaxVectorSize();
        };
    }

    // ---- Tdm staging: plain-Lds descriptors and their pad ------------------------------
    //
    // The Tdm engine writes K and V into Lds itself, in plain row-major order, so the
    // xor-swizzled layouts MakeK/VLdsBlockDescriptor() build are not available to a Tdm
    // staged buffer; bank conflicts are avoided with a row pad instead. These four entry
    // points are additive -- the existing descriptors are untouched and still serve the
    // Default and TrLoad pipelines.
    //
    // They live here rather than in the pipelines because detail::GetTdm*() only exist inside
    // #if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE. A call spelled `detail::GetTdm...` in a
    // pipeline is a non-dependent qualified-id, so it is looked up when the template is
    // *defined* rather than instantiated -- merely including a Tdm pipeline header on
    // gfx94/gfx95 would then be a parse error, however carefully the call is guarded by
    // if constexpr. Going through the policy makes the name dependent on Problem and keeps
    // the #if on this side, which is the same shape the rest of this file already uses.

    // The (pad_interval, pad_amount) pair for the global -> Lds copy that stages K, already
    // encoded the way the Tdm descriptor carries it. MakeKLdsPlainBlockDescriptor() derives
    // the reader side of the same buffer from the same GetTdmLdsPaddingConfigForNormalRead()
    // call; the two are separate derivations and have been checked to agree, so an edit to
    // one must be mirrored in the other.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTdmPadConfigK()
    {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto lds_pad =
            detail::GetTdmLdsPaddingConfigForNormalRead<WG, false /*inputB*/, kKPerBlock>();

        return detail::GetTdmRawPaddingConfig<lds_pad.at(number<0>{}), lds_pad.at(number<1>{})>();
#else
        static_assert(sizeof(Problem) == 0,
                      "GetTdmPadConfigK() is only meaningful where Tdm staging is available");
        return make_tuple(number<0>{}, number<0>{});
#endif
    }

    // As GetTdmPadConfigK(), for the V copy. V is read back with ds_read_tr, so its pad comes
    // from GetTdmLdsPaddingConfigForTrLoadRead() and is wider than K's.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTdmPadConfigV()
    {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;

        using BlockGemm = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, true /*kUseTrLoad*/>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto lds_pad =
            detail::GetTdmLdsPaddingConfigForTrLoadRead<WG, false /*inputB*/, kNPerBlock>();

        return detail::GetTdmRawPaddingConfig<lds_pad.at(number<0>{}), lds_pad.at(number<1>{})>();
#else
        static_assert(sizeof(Problem) == 0,
                      "GetTdmPadConfigV() is only meaningful where Tdm staging is available");
        return make_tuple(number<0>{}, number<0>{});
#endif
    }

    // Reader-side descriptor for a Tdm staged K buffer.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsPlainBlockDescriptor()
    {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
        constexpr index_t NumKLdsBuffers = GetNumKLdsBuffers<Problem>();
        constexpr index_t kNPerBlock     = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock     = Problem::HstuAttentionTileSetting::kQKHeaddim;

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto pad_config =
            detail::GetTdmLdsPaddingConfigForNormalRead<WG, false /*inputB*/, kKPerBlock>();

        return detail::MakeRowMajorLdsPaddedBlockDescriptor<typename Problem::QKVDataType,
                                                            NumKLdsBuffers,
                                                            kNPerBlock,
                                                            kKPerBlock,
                                                            pad_config.at(number<0>{}),
                                                            pad_config.at(number<1>{})>();
#else
        static_assert(sizeof(Problem) == 0,
                      "MakeKLdsPlainBlockDescriptor() is only meaningful where Tdm staging is "
                      "available");
        return 0;
#endif
    }

    // Reader-side descriptor for a Tdm staged V buffer. V is always read back with
    // ds_read_tr, so the pad is the wider tr-load one. kNPerBlock (= kN1) is the row length
    // of the V buffer, i.e. the descriptor's Cols, which is what the pad interval has to be
    // derived from.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsPlainBlockDescriptor()
    {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
        constexpr index_t NumVLdsBuffers = GetNumVLdsBuffers<Problem>();
        constexpr index_t kNPerBlock     = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock     = Problem::HstuAttentionTileSetting::kK1;

        using BlockGemm = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, true /*kUseTrLoad*/>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto pad_config =
            detail::GetTdmLdsPaddingConfigForTrLoadRead<WG, false /*inputB*/, kNPerBlock>();

        return detail::MakeRowMajorLdsPaddedBlockDescriptor<typename Problem::QKVDataType,
                                                            NumVLdsBuffers,
                                                            kKPerBlock,
                                                            kNPerBlock,
                                                            pad_config.at(number<0>{}),
                                                            pad_config.at(number<1>{})>();
#else
        static_assert(sizeof(Problem) == 0,
                      "MakeVLdsPlainBlockDescriptor() is only meaningful where Tdm staging is "
                      "available");
        return 0;
#endif
    }

    template <typename Problem, bool kPipelineUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t NumKLdsBuffers = GetNumKLdsBuffers<Problem>();
        constexpr index_t kNPerBlock     = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock     = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKVector       = GetAlignmentK<Problem>();

        // for hdim96 and hdim160, use simplest layout
        if constexpr(!detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            constexpr index_t SingleBufferSize = kNPerBlock * kKPerBlock;

            constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{}, number<kKPerBlock>{}),
                make_tuple(number<SingleBufferSize>{}, number<kKPerBlock>{}, number<1>{}),
                number<kKVector>{},
                number<1>{});

            constexpr auto k_lds_block_desc = transform_tensor_descriptor(
                k_lds_block_desc_0,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                           make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return k_lds_block_desc;
        }
        else if constexpr(GetQKWarpGemmBScalarPerVector<Problem>() >= GetAlignmentK<Problem>())
        { // This path can only be reached if WarpGemm is 16x16x32 or 32x32x16
            using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG              = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr auto kSwizzleUnit =
                detail::GetSwizzleUnitForNormalRead<WG, false /*inputB*/>();

            constexpr auto desc_native = detail::MakeSwizzledNativeDesc<Problem,
                                                                        NumKLdsBuffers,
                                                                        kNPerBlock,
                                                                        kKPerBlock,
                                                                        kSwizzleUnit>();

            return transform_tensor_descriptor(
                desc_native,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                           make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            constexpr index_t kDsReadVector = GetQKWarpGemmBScalarPerVector<Problem>();

            constexpr index_t SingleBufferSize =
                kKPerBlock * kNPerBlock + kKPerBlock * kDsReadVector / kKVector;

            constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumKLdsBuffers>{},
                           number<kKPerBlock / kKVector>{},
                           number<kKVector / kDsReadVector>{},
                           number<kNPerBlock>{},
                           number<kDsReadVector>{}),
                make_tuple(number<SingleBufferSize>{},
                           number<kNPerBlock * kKVector + kDsReadVector>{},
                           number<kNPerBlock * kDsReadVector>{},
                           number<kDsReadVector>{},
                           number<1>{}),
                number<kDsReadVector>{},
                number<1>{});

            constexpr auto k_lds_block_desc = transform_tensor_descriptor(
                k_lds_block_desc_0,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                           make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                           number<kKVector / kDsReadVector>{},
                                                           number<kDsReadVector>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return k_lds_block_desc;
        };
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;

        constexpr index_t kKVector = GetAlignmentK<Problem>();
        constexpr index_t OtherK   = kKPerBlock / kKVector;

        if constexpr(detail::IsPerfectHeaddimSize(kKPerBlock))
        // for kKPerBlock=32,64,128,256
        {
            static_assert((OtherK & (OtherK - 1)) == 0, "Check failed!");

            constexpr index_t KPerThread = kKVector;
            constexpr index_t KThreads   = OtherK;

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
        else // for kKPerBlock=96,160
        {
            static_assert((OtherK & (OtherK - 1)) != 0, "Check failed!");

            constexpr index_t KRepPerThread = (OtherK % 3 == 0) ? 3 : 5;
            constexpr index_t KThreads      = OtherK / KRepPerThread;

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
        };
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t NumVLdsBuffers = GetNumVLdsBuffers<Problem>();
        constexpr index_t kBlockSize     = Problem::kBlockSize;
        constexpr index_t kNPerBlock     = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock     = Problem::HstuAttentionTileSetting::kK1;

        if constexpr(!kUseTrLoad)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            // K2 is the vector size for storing shuffled tile to LDS
            constexpr index_t K2 = ElemPerThread / N1;

            constexpr index_t kDsReadVector = GetPVTWarpGemmBScalarPerVector<Problem>();

            static_assert(kDsReadVector >= K2, "Check failed!");

            using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG              = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr auto PaddingCfg =
                detail::GetLdsPaddingConfigForNormalRead<WG, false /*inputB */, kKPerBlock>();

            constexpr auto PadInterval = PaddingCfg[number<0>{}];
            constexpr auto PadLength   = PaddingCfg[number<1>{}];

            return detail::MakeRowMajorLdsPaddedBlockDescriptor<NumVLdsBuffers,
                                                                kNPerBlock,
                                                                kKPerBlock,
                                                                PadInterval,
                                                                PadLength>();
        }
        else
        {
            // With trload read,  16 threads per cycle access the [4Tl, 4Tm*4E] block and cross-bar
            // transpose it to [4E, 4Tm*4Tl] layout suitable for mfma. For hdim128, kK1=32, [32,
            // 128] = [2R*4Th*4Tl, 8R*4Tm*4E],  8R columns swizzled by 32 rows, for each value of
            // 8R, the 4Tl rows is mapped to separate bigger bank-groups (each has kSwizzleUnit
            // elements), able to guarantee 16 threads (4Tl*4Tm) in one cycle hitting to separate
            // smaller bank-groups (each has 4 elements, two dwords)
            using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG              = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr auto kSwizzleUnit =
                detail::GetSwizzleUnitForTrLoadRead<WG, false /*inputB*/>();

            constexpr auto desc_native = detail::MakeSwizzledNativeDesc<Problem,
                                                                        NumVLdsBuffers,
                                                                        kKPerBlock, // kN
                                                                        kNPerBlock, // kK
                                                                        kSwizzleUnit>();

            // merge: NumVLdsBuffers * [kK1, kVHeaddim] -> [kN0, kVHeaddim]
            return transform_tensor_descriptor(
                desc_native,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumVLdsBuffers>{}, number<kKPerBlock>{})),
                           make_pass_through_transform(number<kNPerBlock>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        };
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        if constexpr(!kUseTrLoad)
        {
            constexpr index_t NPerThread = GetAlignmentV<Problem>();
            constexpr index_t NThreads   = kNPerBlock / NPerThread;

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            constexpr index_t KPerThread     = ElemPerThread / NPerThread;
            constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NThreads, NPerThread>,
                                                 sequence<NumWarps, KThreadPerWarp, KPerThread>>,
                                           tuple<sequence<2>, sequence<2, 1>>,
                                           tuple<sequence<0>, sequence<1, 0>>,
                                           sequence<2, 1>,
                                           sequence<2, 1>>{});
        }
        else
        {
            constexpr index_t NPerThread = GetAlignmentV<Problem, true>();
            constexpr index_t NThreads   = kNPerBlock / NPerThread;

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            constexpr index_t KPerThread     = ElemPerThread / NPerThread;
            constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NumWarps, KThreadPerWarp, KPerThread>,
                                                 sequence<NThreads, NPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0>>,
                                           sequence<1, 2>,
                                           sequence<2, 1>>{});
        };
    }

#if !HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeShuffledVRegTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        constexpr index_t NPerThread = GetAlignmentV<Problem>();
        constexpr index_t NThreads   = kNPerBlock / NPerThread;

        constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

        constexpr index_t KPerThread     = ElemPerThread / NPerThread;
        constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NThreads, NPerThread>,
                                             sequence<NumWarps, KThreadPerWarp, KPerThread>>,
                                       tuple<sequence<2>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 2>>{});
    }
#endif

    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetQKBlockGemmSingleRepM()
    {
        return Problem::HstuAttentionTileSetting::Gemm0BlockWarps::at(number<0>{}) *
               Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{});
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm0Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN0Sub,
                                   Problem::HstuAttentionTileSetting::kQKHeaddim>,
                          typename Problem::HstuAttentionTileSetting::Gemm0BlockWarps,
                          typename Problem::HstuAttentionTileSetting::Gemm0WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QKVDataType, half_t> ||
                          std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<2>{});

#if defined(__hstu_gfx125__)
                static_assert(WarpGemmM == 16 && WarpGemmK == 32, "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
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
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{}),
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<1>{}),
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<2>{}),
                    true,
                    false,
                    false,
                    WGAttrNumAccessEnum::Default>{};
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
            typename Problem::HstuAttentionTileSetting::Gemm0BlockWarps,
            decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPVTBlockGemmSingleRepN()
    {
        return Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}) *
               Problem::HstuAttentionTileSetting::Gemm1BlockWarps::at(number<1>{});
    };

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetPVTBlockGemm()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::kNumGemm1Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN1,
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

#if defined(__hstu_gfx125__)
                static_assert(WarpGemmM == 16 && WarpGemmK == 32, "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
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
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}),
                        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{}),
                        true,
                        false,
                        false,
#if defined(__hstu_gfx95__)
                        WGAttrNumAccessEnum::Double
#else
                        WGAttrNumAccessEnum::Default
#endif
                        >{};
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
                        WGAttrNumAccessEnum::Default>{};
            }
            else
            {
                static_assert(false, "Not supported data types!");
            }
        }();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
            WarpGemm>;

        if constexpr(!kUseTrLoad)
        {
            return BlockGemmARegBSmemCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
        }
        else
        {
            return BlockGemmARegBSmemTrLoadCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
        };
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, kUseTrLoad>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem, bool kPipelineUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        constexpr auto actual_bytes =
            MakeKLdsBlockDescriptor<Problem, kPipelineUseTrLoad>().get_element_space_size() *
            sizeof(typename Problem::QKVDataType);

        return (actual_bytes + 63) / 64 * 64;
    };

    template <typename Problem, bool kPipelineUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        constexpr auto actual_bytes =
            MakeVLdsBlockDescriptor<Problem, kPipelineUseTrLoad>().get_element_space_size() *
            sizeof(typename Problem::QKVDataType);

        return (actual_bytes + 63) / 64 * 64;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout()
    {
        return 0;
    };

    template <typename Problem, bool kPipelineUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GetSmemSizeK<Problem, kPipelineUseTrLoad>() +
               GetSmemSizeV<Problem, kPipelineUseTrLoad>() + GetSmemSizeDropout<Problem>();
    }

    // Lds sizes for the Tdm staged buffers. Same rounding as GetSmemSizeK/V(), over the
    // plain descriptors.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKPlain()
    {
        constexpr auto actual_bytes =
            MakeKLdsPlainBlockDescriptor<Problem>().get_element_space_size() *
            sizeof(typename Problem::QKVDataType);

        return (actual_bytes + 63) / 64 * 64;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeVPlain()
    {
        constexpr auto actual_bytes =
            MakeVLdsPlainBlockDescriptor<Problem>().get_element_space_size() *
            sizeof(typename Problem::QKVDataType);

        return (actual_bytes + 63) / 64 * 64;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizePlain()
    {
        return GetSmemSizeKPlain<Problem>() + GetSmemSizeVPlain<Problem>() +
               GetSmemSizeDropout<Problem>();
    }
};

} // namespace ck_tile
