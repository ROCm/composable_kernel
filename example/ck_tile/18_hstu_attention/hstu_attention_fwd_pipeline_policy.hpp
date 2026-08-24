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

struct HstuAttentionFwdPipelineQRKSVSPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumKVLdsBuffers()
    {
        return 4;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeABlockTileDistribution<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetQKWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::kKPerThread;
    };

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetPVTWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, kUseTrLoad>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::kKPerThread;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasDramTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        constexpr auto bias_block_dstr_encode = BlockGemm::template MakeCBlockDistributionEncode<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kN0>();
        constexpr auto bias_block_dstr = make_static_tile_distribution(bias_block_dstr_encode);

        return bias_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBias()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
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
        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
            return 8;
        else
            return 4;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        return Problem::GetKDramTileAccessMaxVectorSize();
    }

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        if constexpr(GetPVTWarpGemmKPerThreadSize<Problem, kUseTrLoad>() >= 8)
            return 8;
        else
            return 4;
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

            return kVecLoad;
        }
        else
        {
            return Problem::GetVDramTileAccessMaxVectorSize();
        };
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKSingleSmemElementSpaceSize()
    {
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKPack     = GetSmemKPackK<Problem>();
        constexpr index_t kKVector   = GetAlignmentK<Problem>();

        // for hdim96 and hdim160
        if constexpr(!detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            return kKPerBlock * kNPerBlock;
        }
        else if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
        {
            static_assert(kKVector == kKPack);

            return kKPerBlock * kNPerBlock;
        }
        else
        {
            static_assert(kKVector % kKPack == 0);

            return kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector;
        };
    };

    template <typename Problem, bool kUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetVSingleSmemElementSpaceSize()
    {
        constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

        if constexpr(!kUseTrLoad)
        {
            constexpr index_t N1     = GetAlignmentV<Problem>();
            constexpr index_t N0     = kNPerBlock / N1;
            constexpr index_t kKPack = GetPVTWarpGemmKPerThreadSize<Problem>();

            return N0 * (N1 * kKPerBlock + kKPack);
        }
        else
        {
            return kNPerBlock * kKPerBlock;
        };
    };

    template <typename Problem, bool kPipelineUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        return max(GetKSingleSmemElementSpaceSize<Problem>(),
                   GetVSingleSmemElementSpaceSize<Problem, kPipelineUseTrLoad>());
    };

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

    template <typename Problem, bool kPipelineUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t NumKLdsBuffers = GetNumKVLdsBuffers<Problem>();
        constexpr index_t kNPerBlock     = Problem::HstuAttentionTileSetting::kN0Sub;
        constexpr index_t kKPerBlock     = Problem::HstuAttentionTileSetting::kQKHeaddim;
        constexpr index_t kKPack         = GetSmemKPackK<Problem>();
        constexpr index_t kKVector       = GetAlignmentK<Problem>();

        constexpr index_t SingleSmemElementSpaceSize =
            GetSingleSmemElementSpaceSize<Problem, kPipelineUseTrLoad>();

        // for hdim96 and hdim160, use simplest layout
        if constexpr(!detail::IsPerfectHeaddimSize(kKPerBlock))
        {
            constexpr index_t KSingleSmemElementSpaceSize = kNPerBlock * kKPerBlock;

            static_assert(KSingleSmemElementSpaceSize == GetKSingleSmemElementSpaceSize<Problem>());

            constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{}, number<kKPerBlock>{}),
                make_tuple(number<SingleSmemElementSpaceSize>{}, number<kKPerBlock>{}, number<1>{}),
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
        else if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
        { // This path can only be reached if WarpGemm is 16x16x32 or 32x32x16

            constexpr auto desc_native =
                MakeSwizzledNativeDesc<Problem, NumKLdsBuffers, kNPerBlock, kKPerBlock, kKPack>();

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
            static_assert(kKVector % kKPack == 0);

            constexpr index_t KSingleSmemElementSpaceSize =
                kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector;

            static_assert(KSingleSmemElementSpaceSize == GetKSingleSmemElementSpaceSize<Problem>());

            constexpr auto k_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<NumKLdsBuffers>{},
                                                        number<kKPerBlock / kKVector>{},
                                                        number<kKVector / kKPack>{},
                                                        number<kNPerBlock>{},
                                                        number<kKPack>{}),
                                             make_tuple(number<SingleSmemElementSpaceSize>{},
                                                        number<kNPerBlock * kKVector + kKPack>{},
                                                        number<kNPerBlock * kKPack>{},
                                                        number<kKPack>{},
                                                        number<1>{}),
                                             number<kKPack>{},
                                             number<1>{});

            constexpr auto k_lds_block_desc = transform_tensor_descriptor(
                k_lds_block_desc_0,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                           make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                           number<kKVector / kKPack>{},
                                                           number<kKPack>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return k_lds_block_desc;
        };
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
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
        constexpr index_t NumVLdsBuffers = GetNumKVLdsBuffers<Problem>();
        constexpr index_t kBlockSize     = Problem::kBlockSize;
        constexpr index_t kNPerBlock     = Problem::HstuAttentionTileSetting::kN1;
        constexpr index_t kKPerBlock     = Problem::HstuAttentionTileSetting::kK1;

        constexpr index_t SingleSmemElementSpaceSize =
            GetSingleSmemElementSpaceSize<Problem, kUseTrLoad>();

        if constexpr(!kUseTrLoad)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();
            constexpr index_t N0 = kNPerBlock / N1;

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            // K2 is the vector size for storing shuffled tile to LDS
            constexpr index_t K2 = ElemPerThread / N1;

            // GetSmemKPackV() is the vector size for loading from LDS by BlockGemm
            constexpr index_t kKPack = GetSmemKPackV<Problem>();

            static_assert(kKPack >= K2, "Check failed!");

            constexpr index_t VSingleSmemElementSpaceSize = N0 * (N1 * kKPerBlock + kKPack);

            static_assert(VSingleSmemElementSpaceSize == GetVSingleSmemElementSpaceSize<Problem>());

            constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(
                    number<NumVLdsBuffers>{}, number<N0>{}, number<N1>{}, number<kKPerBlock>{}),
                make_tuple(number<SingleSmemElementSpaceSize>{},
                           number<N1 * kKPerBlock + kKPack>{},
                           number<kKPerBlock>{},
                           number<1>{}),
                number<kKPack>{},
                number<1>{});

            constexpr auto v_lds_block_desc = transform_tensor_descriptor(
                v_lds_block_desc_0,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumVLdsBuffers>{}, number<N0>{}, number<N1>{})),
                           make_pass_through_transform(number<kKPerBlock>{})),
                make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return v_lds_block_desc;
        }
        else
        {
            constexpr index_t kKPack = GetSmemKPackV<Problem, true>();

            constexpr auto XorGroupSize =
                Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{});

            constexpr index_t VSingleSmemElementSpaceSize = kNPerBlock * kKPerBlock;

            static_assert(VSingleSmemElementSpaceSize ==
                          GetVSingleSmemElementSpaceSize<Problem, true>());

            constexpr auto v_lds_block_desc_naive =
                make_naive_tensor_descriptor(make_tuple(number<NumVLdsBuffers>{},
                                                        number<kKPerBlock>{},
                                                        number<kNPerBlock / XorGroupSize>{},
                                                        number<XorGroupSize>{}),
                                             make_tuple(number<SingleSmemElementSpaceSize>{},
                                                        number<kNPerBlock>{},
                                                        number<XorGroupSize>{},
                                                        number<1>{}),
                                             number<kKPack>{},
                                             number<1>{});

            constexpr auto v_lds_block_desc_permuted = transform_tensor_descriptor(
                v_lds_block_desc_naive,
                make_tuple(make_pass_through_transform(number<NumVLdsBuffers>{}),
                           make_xor_transform(make_tuple(number<kKPerBlock>{},
                                                         number<kNPerBlock / XorGroupSize>{})),
                           make_pass_through_transform(number<XorGroupSize>{})),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            return transform_tensor_descriptor(
                v_lds_block_desc_permuted,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumVLdsBuffers>{}, number<kKPerBlock>{})),
                           make_merge_transform_v3_division_mod(make_tuple(
                               number<kNPerBlock / XorGroupSize>{}, number<XorGroupSize>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
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

    // used when kUseTrLoad is false
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegTileDistribution()
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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQKBlockGemmSingleRepM()
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
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{}),
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<1>{}),
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<2>{}),
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
            typename Problem::HstuAttentionTileSetting::Gemm0BlockWarps,
            decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
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
            Problem::kNumGemm0Warps * get_warp_size(),
            TileGemmShape<sequence<Problem::HstuAttentionTileSetting::kM0,
                                   Problem::HstuAttentionTileSetting::kN0,
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
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<0>{}),
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<1>{}),
                    Problem::HstuAttentionTileSetting::Gemm0WarpTile::at(number<2>{}),
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
            typename Problem::HstuAttentionTileSetting::Gemm0BlockWarps,
            decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPVTBlockGemmSingleRepN()
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
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVTBlockGemm<Problem, kUseTrLoad>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem, bool kPipelineUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV()
    {
        constexpr index_t num_kv_lds_buffers = GetNumKVLdsBuffers<Problem>();

        return num_kv_lds_buffers * GetSingleSmemElementSpaceSize<Problem, kPipelineUseTrLoad>() *
               sizeof(typename Problem::QKVDataType);
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout()
    {
        if constexpr(Problem::kHasDropout)
        {
            using BlockGemm          = remove_cvref_t<decltype(GetQKCombinedBlockGemm<Problem>())>;
            constexpr auto config    = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG                 = remove_cvref_t<decltype(config.template at<0>())>;
            constexpr bool IsWG32    = WG::kM == 32;
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

    template <typename Problem, bool kPipelineUseTrLoad = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GetSmemSizeKV<Problem, kPipelineUseTrLoad>() + GetSmemSizeDropout<Problem>();
    }
};

} // namespace ck_tile
