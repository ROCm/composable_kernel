// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

struct BlockFmhaPipelineQRKSVSWholeKPrefetchDefaultPolicy
{
    static constexpr bool QLoadOnce = true;  // needed by the kernel
    static constexpr bool AsyncCopy = false; // needed by the kernel

    static constexpr index_t NumPrefetchV = 2;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t IsPreloadWholeNextIterationK()
    {
        return Problem::BlockFmhaShape::kM0 <= 64;
    };

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumKLdsBuffers()
    {
        return 2;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumPrefetchV()
    {
        using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

        constexpr index_t kN0 = BlockFmhaShape::kN0;
        constexpr index_t kK1 = BlockFmhaShape::kK1;

        constexpr index_t k1_loops = kN0 / kK1;

        return min(NumPrefetchV, k1_loops);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetNumVLdsBuffers()
    {
        return 2;
    };

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

        return BlockGemm::MakeCBlockTile().get_tile_distribution();
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
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

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
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kK1;
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
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t NumKLdsBuffers = GetNumKLdsBuffers<Problem>();
        constexpr index_t kNPerBlock     = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kK0;
        constexpr index_t kKPack         = GetSmemKPackK<Problem>();
        constexpr index_t kKVector       = GetAlignmentK<Problem>();

        static_assert(kKVector % kKPack == 0);

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NumKLdsBuffers>{},
                       number<kKPerBlock / kKVector>{},
                       number<kKVector / kKPack>{},
                       number<kNPerBlock>{},
                       number<kKPack>{}),
            make_tuple(number<kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector>{},
                       number<kNPerBlock * kKVector + kKPack>{},
                       number<kNPerBlock * kKPack>{},
                       number<kKPack>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                number<kKVector / kKPack>{},
                                                number<kKPack>{}))),
            make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(KDataType);

        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
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
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t NumVLdsBuffers = GetNumVLdsBuffers<Problem>();

        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
        constexpr index_t kKPack       = GetSmemKPackV<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr index_t VSingleSmemElementSpaceSize =
            (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NumVLdsBuffers>{},
                       number<kKPerBlock / kKPack>{},
                       number<kNPerBlock / NPerRow>{},
                       number<NPerRow>{},
                       number<kKPack>{}),
            make_tuple(number<VSingleSmemElementSpaceSize>{},
                       number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       number<PixelsPerRow + kKPack>{},
                       number<kKPack>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(
                    number<NumVLdsBuffers>{}, number<kNPerBlock / NPerRow>{}, number<NPerRow>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
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

        auto warp_gemm = [&]() {
            if constexpr((std::is_same_v<typename Problem::QDataType, half_t> ||
                          std::is_same_v<typename Problem::QDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                constexpr index_t WarpGemmM =
                    Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
                constexpr index_t WarpGemmK =
                    Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{});

#ifdef __gfx950__
                static_assert((WarpGemmM == 16 && WarpGemmK == 32) ||
                                  (WarpGemmM == 32 && WarpGemmK == 16),
                              "Not supported WarpGemm sizes!");

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
#else
                static_assert((WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                                  (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
                              "Not supported WarpGemm sizes!");

                return WarpGemmDispatcher<typename Problem::QDataType,
                                          typename Problem::KVDataType,
                                          typename Problem::SaccDataType,
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                          true,
                                          false,
                                          false,
                                          WGAttrNumAccessEnum::Single>{};
#endif
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
            return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
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
                          std::is_same_v<typename Problem::VDataType, bf16_t>) &&
                         std::is_same_v<typename Problem::OaccDataType, float>)
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
            return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    // leave some exclusive space so that the second v_lds buffer will nenver
    // overlap with the first k_lds bufffer
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetExclusiveKLdsBytes()
    {
        constexpr index_t single_k_lds_buffer_size =
            GetSmemSizeK<Problem>() / GetNumKLdsBuffers<Problem>();
        constexpr index_t single_v_lds_buffer_size =
            GetSmemSizeV<Problem>() / GetNumVLdsBuffers<Problem>();

        if constexpr(single_k_lds_buffer_size <= single_v_lds_buffer_size)
            return 0;
        else
            return integer_least_multiple(single_k_lds_buffer_size - single_v_lds_buffer_size, 64);
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t IsFirstKLdsBufferOverlapLastVLdsBuffer()
    {
        using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

        constexpr index_t k1_loops          = BlockFmhaShape::kN0 / BlockFmhaShape::kK1;
        constexpr index_t num_k_lds_buffers = GetNumKLdsBuffers<Problem>();
        constexpr index_t num_v_lds_buffers = GetNumVLdsBuffers<Problem>();

        constexpr index_t last_v_lds_buffer_offset =
            MakeVLdsBlockDescriptor<Problem>().get_element_space_size() / num_v_lds_buffers *
            ((k1_loops - 1) % num_v_lds_buffers) * sizeof(typename Problem::VDataType);

        constexpr index_t first_k_lds_buffer_size =
            MakeKLdsBlockDescriptor<Problem>().get_element_space_size() / num_k_lds_buffers *
            sizeof(typename Problem::KDataType);

        return GetExclusiveKLdsBytes<Problem>() + last_v_lds_buffer_offset <
               first_k_lds_buffer_size;
    };

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
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout()
    {
        return 0;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // assume V can reuse the other shared memory by K except the first
        // assume Dropout can reuse the shared memory by V
        return GetExclusiveKLdsBytes<Problem>() +
               max(GetSmemSizeK<Problem>() - GetExclusiveKLdsBytes<Problem>(),
                   max(GetSmemSizeV<Problem>(), GetSmemSizeDropout<Problem>()));
    }
};

} // namespace ck_tile
