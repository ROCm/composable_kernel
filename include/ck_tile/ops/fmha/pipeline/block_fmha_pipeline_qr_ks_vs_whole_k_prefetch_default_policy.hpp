// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_prefetch_k.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_prefetch_n.hpp"

namespace ck_tile {

struct BlockFmhaPipelineQRKSVSWholeKPrefetchDefaultPolicy
{
    static constexpr bool QLoadOnce = true;  // needed by the kernel
    static constexpr bool AsyncCopy = false; // needed by the kernel

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t IsPreloadWholeNextIterationK()
    {
        return Problem::BlockFmhaShape::kM0 <= 64;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetNumPrefetchV()
    {
        constexpr index_t k1_loops = Problem::BlockFmhaShape::kN0 / Problem::BlockFmhaShape::kK1;

        // usually kN0 is 128, kK1 is 32/16
        static_assert(k1_loops >= 2, "Check failed!");

        return 2;
    };

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumKVLdsBuffers()
    {
        return 4;
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
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasDramTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeCBlockTile<Problem::BlockFmhaShape::kM0,
                                                  Problem::BlockFmhaShape::kN0>()
            .get_tile_distribution();
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
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

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
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0Sub;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

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
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;

        constexpr index_t MaxVectorSize = 16 / sizeof(VDataType);
        constexpr index_t kMaxVecLoad   = min(ElemPerThread, MaxVectorSize);
        constexpr index_t kMinVecLoad   = 4 / sizeof(VDataType);

        constexpr index_t kVecLoad = ((ElemPerThread / kMaxVecLoad) >= kMinVecLoad)
                                         ? kMaxVecLoad
                                         : (ElemPerThread / kMinVecLoad);

        return kVecLoad;
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
    CK_TILE_HOST_DEVICE static constexpr auto GetKSingleSmemElementSpaceSize()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0Sub;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;
        constexpr index_t kKPack     = GetSmemKPackK<Problem>();
        constexpr index_t kKVector   = GetAlignmentK<Problem>();

        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVSingleSmemElementSpaceSize()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t N1     = GetAlignmentV<Problem>();
        constexpr index_t N0     = kNPerBlock / N1;
        constexpr index_t kKPack = GetKVWarpGemmKPerThreadSize<Problem>();

        return N0 * (N1 * kKPerBlock + kKPack);
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
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;
        constexpr index_t kKPack     = GetSmemKPackQ<Problem>();
        constexpr index_t kKVector   = GetAlignmentQ<Problem>();

        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
        {
            static_assert(kKVector == kKPack);

            using QDataType = remove_cvref_t<typename Problem::QDataType>;

            constexpr index_t DataTypeSize = sizeof(QDataType);

            // 128 contiguous bytes mapped to 32 banks with each bank 4 contiguous bytes
            constexpr auto MLdsLayer =
                (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

            constexpr auto q_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kMPerBlock / MLdsLayer>{},
                           number<kKPerBlock / kKPack * MLdsLayer>{},
                           number<kKPack>{}),
                make_tuple(number<kKPerBlock * MLdsLayer>{}, number<kKPack>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            constexpr auto q_lds_block_desc_permuted = transform_tensor_descriptor(
                q_lds_block_desc_0,
                make_tuple(
                    make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                                  number<kKPerBlock / kKPack * MLdsLayer>{})),
                    make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));

            constexpr auto q_lds_block_desc_k0_mldslayer_m_k1 = transform_tensor_descriptor(
                q_lds_block_desc_permuted,
                make_tuple(make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
                           make_unmerge_transform(
                               make_tuple(number<kKPerBlock / kKPack>{}, number<MLdsLayer>{})),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            constexpr auto q_lds_block_desc = transform_tensor_descriptor(
                q_lds_block_desc_k0_mldslayer_m_k1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<0, 2>{}, sequence<1, 3>{}),
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
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = GetQKBlockGemmSingleRepM<Problem>();
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t kKVector = GetAlignmentQ<Problem>();

        constexpr index_t KPerThread     = kKVector;
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
        constexpr index_t kNPerBlock     = Problem::BlockFmhaShape::kN0Sub;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kSubQKHeaddim;
        constexpr index_t kKPack         = GetSmemKPackK<Problem>();
        constexpr index_t kKVector       = GetAlignmentK<Problem>();

        if constexpr(GetQKWarpGemmKPerThreadSize<Problem>() >= 8)
        {
            static_assert(kKVector == kKPack);

            using KDataType = remove_cvref_t<typename Problem::KDataType>;

            constexpr index_t DataTypeSize = sizeof(KDataType);

            // 128 contiguous bytes mapped to 32 banks with each bank 4 contiguous bytes
            constexpr auto NLdsLayer =
                (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

            constexpr auto k_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<NumKLdsBuffers>{},
                                                        number<kNPerBlock / NLdsLayer>{},
                                                        number<kKPerBlock / kKPack * NLdsLayer>{},
                                                        number<kKPack>{}),
                                             make_tuple(number<kKPerBlock * kNPerBlock>{},
                                                        number<kKPerBlock * NLdsLayer>{},
                                                        number<kKPack>{},
                                                        number<1>{}),
                                             number<kKPack>{},
                                             number<1>{});

            constexpr auto k_lds_block_desc_permuted = transform_tensor_descriptor(
                k_lds_block_desc_0,
                make_tuple(
                    make_pass_through_transform(number<NumKLdsBuffers>{}),
                    make_xor_transform(make_tuple(number<kNPerBlock / NLdsLayer>{},
                                                  number<kKPerBlock / kKPack * NLdsLayer>{})),
                    make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            constexpr auto k_lds_block_desc_k0_nldslayer_n_k1 = transform_tensor_descriptor(
                k_lds_block_desc_permuted,
                make_tuple(make_pass_through_transform(number<NumKLdsBuffers>{}),
                           make_pass_through_transform(number<kNPerBlock / NLdsLayer>{}),
                           make_unmerge_transform(
                               make_tuple(number<kKPerBlock / kKPack>{}, number<NLdsLayer>{})),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}));

            constexpr auto k_lds_block_desc = transform_tensor_descriptor(
                k_lds_block_desc_k0_nldslayer_n_k1,
                make_tuple(
                    make_merge_transform_v3_division_mod(
                        make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                    make_merge_transform_v3_division_mod(make_tuple(number<NumKLdsBuffers>{},
                                                                    number<kKPerBlock / kKPack>{},
                                                                    number<kKPack>{}))),
                make_tuple(sequence<1, 3>{}, sequence<0, 2, 4>{}),
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
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0Sub;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t kKVector = GetAlignmentK<Problem>();

        constexpr index_t KPerThread     = kKVector;
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
        constexpr index_t NumVLdsBuffers = GetNumKVLdsBuffers<Problem>();
        constexpr index_t kBlockSize     = Problem::kBlockSize;
        constexpr index_t kNPerBlock     = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kK1;

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
            make_tuple(number<NumVLdsBuffers>{}, number<N0>{}, number<N1>{}, number<kKPerBlock>{}),
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

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
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
                                       sequence<2, 1>,
                                       sequence<2, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegTileDistribution()
    {
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
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kNumGemm0Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0Sub,
                                                    Problem::BlockFmhaShape::kQKHeaddim>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QDataType, half_t> ||
                          std::is_same_v<typename Problem::QDataType, bf16_t>)&&std::
                             is_same_v<typename Problem::SaccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{});

#ifdef __gfx950__
                static_assert((WarpGemmM == 16 && WarpGemmK == 32) ||
                                  (WarpGemmM == 32 && WarpGemmK == 16),
                              "Not supported WarpGemm sizes!");
#else
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");
#endif

                return WarpGemmDispatcher<typename Problem::QDataType,
                                          typename Problem::KDataType,
                                          typename Problem::SaccDataType,
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
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

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 WarpGemm>;

        if constexpr(1 < Problem::kNumGemm0Warps)
            return BlockGemmARegBSmemCRegV2PrefetchK<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
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

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::VDataType, half_t> ||
                          std::is_same_v<typename Problem::VDataType, bf16_t>)&&std::
                             is_same_v<typename Problem::OaccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{});

                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");

                if constexpr((WarpGemmM == 16 && WarpGemmK == 32) ||
                             (WarpGemmM == 32 && WarpGemmK == 16))
                    return WarpGemmDispatcher<
                        typename Problem::PDataType,
                        typename Problem::VDataType,
                        typename Problem::OaccDataType,
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                        true,
                        false,
                        false,
                        WGAttrNumAccessEnum::Double>{};
                else
                    return WarpGemmDispatcher<
                        typename Problem::PDataType,
                        typename Problem::VDataType,
                        typename Problem::OaccDataType,
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
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

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::OaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;

        if constexpr(1 < Problem::kNumGemm1Warps)
            return BlockGemmARegBSmemCRegV2PrefetchN<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV()
    {
        constexpr index_t num_kv_lds_buffers = GetNumKVLdsBuffers<Problem>();

        return num_kv_lds_buffers * GetSingleSmemElementSpaceSize<Problem>() *
               max(sizeof(typename Problem::KDataType), sizeof(typename Problem::VDataType));
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout()
    {
        return 0;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return max(GetSmemSizeKV<Problem>() + GetSmemSizeDropout<Problem>(),
                   GetSmemSizeQ<Problem>());
    }
};

} // namespace ck_tile
