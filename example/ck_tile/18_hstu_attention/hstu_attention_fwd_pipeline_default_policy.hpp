// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

#include "block_gemm_areg_bsmem_creg_v2_hack_0.hpp"
#include "block_gemm_areg_bsmem_creg_v2_hack_1.hpp"

namespace ck_tile {

struct HstuAttentionFwdPipelineQRKSVSDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ false,
                                          /* NumPrefetchK = */ -1,
                                          /* NumPrefetchV = */ 1>
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumKVLdsBuffers()
    {
        return 3;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegSingleRepMTileDistribution()
    {
        using BlockGemm               = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr index_t kBlockGemmM = GetQKBlockGemmSingleRepM<Problem>();

        return BlockGemm::
            template MakeABlockTileDistribution<kBlockGemmM, Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeABlockTileDistribution<
            Problem::BlockFmhaShape::kM0,
            Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetQKWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::kKPerThread;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetKVWarpGemmKPerThreadSize()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::kKPerThread;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
            return 8;
        else
            return 4;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = GetQKBlockGemmSingleRepM<Problem>();
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(QDataType);
        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;

        return min(MaxVectorSize, ElemPerThread);
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
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(KDataType);
        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

        return min(MaxVectorSize, ElemPerThread);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        if constexpr(GetKVWarpGemmKPerThreadSize<Problem>() >= 8)
            return 8;
        else
            return 4;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        using VLayout   = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

        // Need special consideration for RowMajor since shuffling is needed to write LDS in dwords
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t MaxVectorSize = 16 / sizeof(VDataType);
            constexpr index_t kMaxVecLoad   = min(ElemPerThread, MaxVectorSize);
            constexpr index_t kMinVecLoad   = 4 / sizeof(VDataType);

            constexpr index_t kVecLoad = ((ElemPerThread / kMaxVecLoad) >= kMinVecLoad)
                                             ? kMaxVecLoad
                                             : (ElemPerThread / kMinVecLoad);

            return kVecLoad;
        }
        else // Similar to GetAlignmentK()
        {
            constexpr index_t MaxVectorSize = 16 / sizeof(VDataType);
            return min(ElemPerThread, MaxVectorSize);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKSingleSmemElementSpaceSize()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack     = GetSmemKPackK<Problem>();
        constexpr index_t kKVector   = GetAlignmentK<Problem>();

        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
        {
            static_assert(kKVector == kKPack);

            return kKPerBlock * kNPerBlock + kKPerBlock;
        }
        else
        {
            static_assert(kKVector % kKPack == 0);

            return kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector;
        };
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVSingleSmemElementSpaceSize()
    {
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        // Need special consideration for RowMajor since shuffling is needed to write LDS in dwords
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1     = GetAlignmentV<Problem>();
            constexpr index_t N0     = kNPerBlock / N1;
            constexpr index_t kKPack = GetKVWarpGemmKPerThreadSize<Problem>();

            return N0 * (N1 * kKPerBlock + kKPack);
        }
        else // similar to GetKSingleSmemElementSpaceSize()
        {
            constexpr index_t kKPack   = GetSmemKPackV<Problem>();
            constexpr index_t kKVector = GetAlignmentV<Problem>();

            if constexpr(GetKVWarpGemmKPerThreadSize<Problem>() >= 8)
            {
                static_assert(kKVector == kKPack);

                return kKPerBlock * kNPerBlock + kKPerBlock;
            }
            else
            {
                static_assert(kKVector % kKPack == 0);

                return kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector;
            };
        };
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        return max(GetKSingleSmemElementSpaceSize<Problem>(),
                   GetVSingleSmemElementSpaceSize<Problem>());
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = GetQKBlockGemmSingleRepM<Problem>();
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack     = GetSmemKPackQ<Problem>();
        constexpr index_t kKVector   = GetAlignmentQ<Problem>();

        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
        {
            static_assert(kKVector == kKPack);

            constexpr auto q_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kKPerBlock / kKPack>{}, number<kMPerBlock>{}, number<kKPack>{}),
                make_tuple(number<kMPerBlock * kKPack + kKPack>{}, number<kKPack>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            constexpr auto q_lds_block_desc = transform_tensor_descriptor(
                q_lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<kMPerBlock>{}),
                           make_merge_transform(
                               make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return q_lds_block_desc;
        }
        else
        {
            static_assert(kKVector % kKPack == 0);

            constexpr auto q_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<kKPerBlock / kKVector>{},
                                                        number<kKVector / kKPack>{},
                                                        number<kMPerBlock>{},
                                                        number<kKPack>{}),
                                             make_tuple(number<kMPerBlock * kKVector + kKPack>{},
                                                        number<kMPerBlock * kKPack>{},
                                                        number<kKPack>{},
                                                        number<1>{}),
                                             number<kKPack>{},
                                             number<1>{});

            constexpr auto q_lds_block_desc = transform_tensor_descriptor(
                q_lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<kMPerBlock>{}),
                           make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                           number<kKVector / kKPack>{},
                                                           number<kKPack>{}))),
                make_tuple(sequence<2>{}, sequence<0, 1, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return q_lds_block_desc;
        };
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramSingleRepMTileDistribution()
    {
        using QKVDataType = remove_cvref_t<typename Problem::QKVDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = GetQKBlockGemmSingleRepM<Problem>();
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(QKVDataType);

        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

        constexpr index_t KPerThread     = kMaxVecLoad;
        constexpr index_t KThreads       = kKPerBlock / KPerThread;
        constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

        // for Q-Tile [64, 128], the encoding is [4W * 4T * 4E,   16T * 8E]
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NumWarps, MThreadPerWarp, MPerThread>,
                                             sequence<KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t NumKLdsBuffers = GetNumKVLdsBuffers<Problem>();
        constexpr index_t kNPerBlock     = Problem::BlockFmhaShape::kK1;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack         = GetSmemKPackK<Problem>();
        constexpr index_t kKVector       = GetAlignmentK<Problem>();

        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
        {
            static_assert(kKVector == kKPack);

            constexpr index_t KSingleSmemElementSpaceSize = kKPerBlock * kNPerBlock + kKPerBlock;

            static_assert(KSingleSmemElementSpaceSize == GetKSingleSmemElementSpaceSize<Problem>());

            constexpr index_t SingleSmemElementSpaceSize = GetSingleSmemElementSpaceSize<Problem>();

            constexpr auto k_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<NumKLdsBuffers>{},
                                                        number<kKPerBlock / kKPack>{},
                                                        number<kNPerBlock>{},
                                                        number<kKPack>{}),
                                             make_tuple(number<SingleSmemElementSpaceSize>{},
                                                        number<kNPerBlock * kKPack + kKPack>{},
                                                        number<kKPack>{},
                                                        number<1>{}),
                                             number<kKPack>{},
                                             number<1>{});

            constexpr auto k_lds_block_desc = transform_tensor_descriptor(
                k_lds_block_desc_0,
                make_tuple(make_merge_transform(
                               make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                           make_merge_transform(
                               make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<0, 2>{}, sequence<1, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return k_lds_block_desc;
        }
        else
        {
            static_assert(kKVector % kKPack == 0);

            constexpr index_t KSingleSmemElementSpaceSize =
                kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector;

            static_assert(KSingleSmemElementSpaceSize == GetKSingleSmemElementSpaceSize<Problem>());

            constexpr index_t SingleSmemElementSpaceSize = GetSingleSmemElementSpaceSize<Problem>();

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
        using QKVDataType = remove_cvref_t<typename Problem::QKVDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(QKVDataType);
        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

        constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

        constexpr index_t KPerThread     = kMaxVecLoad;
        constexpr index_t KThreads       = kKPerBlock / KPerThread;
        constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NThreadPerWarp, NumWarps>,
                                             sequence<KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<2>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;

        constexpr index_t NumVLdsBuffers = GetNumKVLdsBuffers<Problem>();
        constexpr index_t kBlockSize     = Problem::kBlockSize;
        constexpr index_t kNPerBlock     = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kK1;

        // Need special consideration for RowMajor since shuffling is needed to write LDS in dwords
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
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

            constexpr index_t SingleSmemElementSpaceSize = GetSingleSmemElementSpaceSize<Problem>();

            constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(
                    number<NumVLdsBuffers>{}, number<N0>{}, number<N1>{}, number<kKPerBlock>{}),
                make_tuple(number<SingleSmemElementSpaceSize>{},
                           number<N1 * kKPerBlock + kKPack>{},
                           number<kKPerBlock>{},
                           number<1>{}),
                number<8>{},
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
        else // Similar to MakeKLdsBlockDescriptor()
        {
            constexpr index_t kKPack   = GetSmemKPackV<Problem>();
            constexpr index_t kKVector = GetAlignmentV<Problem>();

            if constexpr(GetKVWarpGemmKPerThreadSize<Problem>() >= 8)
            {
                static_assert(kKVector == kKPack);

                constexpr index_t VSingleSmemElementSpaceSize =
                    kKPerBlock * kNPerBlock + kKPerBlock;

                static_assert(VSingleSmemElementSpaceSize ==
                              GetVSingleSmemElementSpaceSize<Problem>());

                constexpr index_t SingleSmemElementSpaceSize =
                    GetSingleSmemElementSpaceSize<Problem>();

                constexpr auto v_lds_block_desc_0 =
                    make_naive_tensor_descriptor(make_tuple(number<NumVLdsBuffers>{},
                                                            number<kKPerBlock / kKPack>{},
                                                            number<kNPerBlock>{},
                                                            number<kKPack>{}),
                                                 make_tuple(number<SingleSmemElementSpaceSize>{},
                                                            number<kNPerBlock * kKPack + kKPack>{},
                                                            number<kKPack>{},
                                                            number<1>{}),
                                                 number<kKPack>{},
                                                 number<1>{});

                constexpr auto v_lds_block_desc = transform_tensor_descriptor(
                    v_lds_block_desc_0,
                    make_tuple(make_merge_transform(
                                   make_tuple(number<NumVLdsBuffers>{}, number<kNPerBlock>{})),
                               make_merge_transform(
                                   make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
                    make_tuple(sequence<0, 2>{}, sequence<1, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return v_lds_block_desc;
            }
            else
            {
                static_assert(kKVector % kKPack == 0);

                constexpr index_t VSingleSmemElementSpaceSize =
                    kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector;

                static_assert(VSingleSmemElementSpaceSize ==
                              GetVSingleSmemElementSpaceSize<Problem>());

                constexpr index_t SingleSmemElementSpaceSize =
                    GetSingleSmemElementSpaceSize<Problem>();

                constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<NumVLdsBuffers>{},
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

                constexpr auto v_lds_block_desc = transform_tensor_descriptor(
                    v_lds_block_desc_0,
                    make_tuple(make_merge_transform(
                                   make_tuple(number<NumVLdsBuffers>{}, number<kNPerBlock>{})),
                               make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                               number<kKVector / kKPack>{},
                                                               number<kKPack>{}))),
                    make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return v_lds_block_desc;
            };
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        // Need special consideration for RowMajor since shuffling is needed to write LDS in dwords
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();
            constexpr index_t N0 = kNPerBlock / N1;

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

            static_assert(ElemPerThread % N1 == 0);

            constexpr index_t K2 = ElemPerThread / N1;
            constexpr index_t K1 = get_warp_size() / N0;
            constexpr index_t K0 = kBlockSize / get_warp_size();

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2>>,
                                           tuple<sequence<2>, sequence<2, 1>>,
                                           tuple<sequence<0>, sequence<1, 0>>,
                                           sequence<2, 1>,
                                           sequence<2, 1>>{});
        }
        else // Similar to MakeKDramTileDistribution()
        {
            using QKVDataType = remove_cvref_t<typename Problem::QKVDataType>;

            constexpr index_t MaxVectorSize = 16 / sizeof(QKVDataType);
            constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

            constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

            constexpr index_t KPerThread     = kMaxVecLoad;
            constexpr index_t KThreads       = kKPerBlock / KPerThread;
            constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NPerThread, NThreadPerWarp, NumWarps>,
                                                 sequence<KThreads, KPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<2>, sequence<1, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegTileDistribution()
    {
        // This tile-distribuiton only used when V layout is RowMajor
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>);

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t N1 = GetAlignmentV<Problem>();
        constexpr index_t N0 = kNPerBlock / N1;

        constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

        static_assert(ElemPerThread % N1 == 0);

        constexpr index_t K2 = ElemPerThread / N1;
        constexpr index_t K1 = get_warp_size() / N0;
        constexpr index_t K0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1>, sequence<K0, K1, K2>>,
                                       tuple<sequence<2>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQKBlockGemmSingleRepM()
    {
        return Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{}) *
               Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QKVDataType,
                             typename Problem::QKVDataType,
                             typename Problem::GemmAccDataType,
                             Problem::kNumGemm0Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kK1,
                                                    Problem::BlockFmhaShape::kQKHeaddim>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr auto warp_gemm = []() {
            constexpr index_t WarpGemmM = Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
            constexpr index_t WarpGemmK = Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{});
            static_assert(WarpGemmM == 4 || WarpGemmM == 16 || WarpGemmM == 32);

            if constexpr(std::is_same_v<typename Problem::QKVDataType, half_t> &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                {
                    if constexpr(WarpGemmK == 32)
                        return WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution{};
                    else
                        return WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
                }
                else // WarpGemmM == 4
                    return WarpGemmMfmaF16F16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QKVDataType, bf16_t> &&
                              std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                if constexpr(WarpGemmM == 32)
                    return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(WarpGemmM == 16)
                {
                    if constexpr(WarpGemmK == 32)
                        return WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution{};
                    else
                        return WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
                }
                else // WarpGemmM == 4
                    return WarpGemmMfmaBf16Bf16F32M4N64K16{};
            }
            else if constexpr(std::is_same_v<typename Problem::QKVDataType, fp8_t> &&
                              std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                static_assert(WarpGemmM == 32);

                // TODO: hard coded here. Otherwise, it may incorrect result
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            } // TODO - bf8_t
        }();

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QKVDataType,
                                                 typename Problem::QKVDataType,
                                                 typename Problem::GemmAccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemmSingleRepN()
    {
        return Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}) *
               Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QKVDataType,
                             typename Problem::QKVDataType,
                             typename Problem::GemmAccDataType,
                             Problem::kNumGemm1Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN1,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr(std::is_same_v<typename Problem::QKVDataType, fp8_t> &&
                         std::is_same_v<typename Problem::GemmAccDataType, float>)
            {
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<>{};
                // return
                // WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
                //         WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<typename
                //         Problem::PDataType, typename Problem::VDataType>>>{};
            }
            else
            {
                return WarpGemmMfmaDispatcher<
                    typename Problem::QKVDataType,
                    typename Problem::QKVDataType,
                    typename Problem::GemmAccDataType,
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                    true>{};
            }
        }();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QKVDataType,
                                                 typename Problem::QKVDataType,
                                                 typename Problem::GemmAccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t IsFirstKLdsBufferOverlapLastVLdsBuffer()
    {
        using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

        constexpr index_t k1_loops           = BlockFmhaShape::kN0 / BlockFmhaShape::kK1;
        constexpr index_t num_kv_lds_buffers = GetNumKVLdsBuffers<Problem>();

        return (k1_loops - 1 + 1) % num_kv_lds_buffers == 0;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QKVDataType);
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV()
    {
        constexpr index_t num_kv_lds_buffers = GetNumKVLdsBuffers<Problem>();

        return num_kv_lds_buffers * GetSingleSmemElementSpaceSize<Problem>() *
               sizeof(typename Problem::QKVDataType);
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return max(GetSmemSizeKV<Problem>() + GetSmemSizeDropout<Problem>(0),
                   GetSmemSizeQ<Problem>());
    }
};

} // namespace ck_tile
