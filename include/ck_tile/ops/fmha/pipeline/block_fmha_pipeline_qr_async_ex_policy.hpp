// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/block_gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp"

// TODO: remove this
// #define K_LDS_LOAD_USE_OFFSET_TRANSFORM 0

namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaPipelineQRAsyncEx
{
    static constexpr index_t NumPrefetchK = 2;
    static constexpr index_t NumPrefetchV = 2;
    static constexpr bool AsyncCopyK      = true;
    static constexpr bool AsyncCopyV      = true;
    static constexpr bool QLoadOnce       = true;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_Q()
    {
        using WG = GetWarpGemm_0<Problem>();
        return WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalDesc_Q()
    {
        using WG = GetWarpGemm_0<Problem>();
        constexpr index_t MWarp =
            Problem::BlockFmhaShape::Gemm0BlockWarps; //   config.template at<1>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0BlockLength;

        constexpr index_t K2 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K0 = kKPerBlock / (K1 * K2);

        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
                                       tuple<sequence<1>, sequence<2, 1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemm_0()
    {
        constexpr auto warp_gemm = []() {
            if constexpr(std::is_same_v<typename Problem::QDataType, half_t> &&
                         std::is_same_v<typename Problem::KDataType, half_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                // return WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                return WarpGemmImpl<
                    WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Raw_vaa>,
                        2>>;
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                              std::is_same_v<typename Problem::KDataType, bf16_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                // return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                return WarpGemmImpl<
                    WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Raw_vaa>,
                        2>>;
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, fp8_t> &&
                              std::is_same_v<typename Problem::KDataType, fp8_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                // TODO: hard coded here. Otherwise, it may incorrect result
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            } // TODO - bf8_t
        }();

        return warp_gemm;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm_0()
    {
        using BlockGemmProblem = BlockGemmPipelineProblem<
            typename Problem::QDataType,
            typename Problem::KDataType,
            typename Problem::SaccDataType,
            TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                   Problem::BlockFmhaShape::kN0,
                                   Problem::BlockFmhaShape::kK0>,
                          typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                          typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr auto warp_gemm = GetWarpGemm_0<Problem>();

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 decltype(warp_gemm)>;

        return BlockGemmARegBSmemCRegV2<BlockGemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack_K()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_K()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        if constexpr(AsyncCopyK)
        {
            return 4 / sizeof(KDataType);
        }
        else
        {
            return 16 / sizeof(KDataType);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack_V()
    {
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_V()
    {
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        if constexpr(AsyncCopyV)
        {
            return 4 / sizeof(VDataType);
        }
        else
        {
            return 16 / sizeof(VDataType);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_Bias()
    {
        using WG        = GetWarpGemm_0<Problem>();
        using CWarpDstr = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths().at(number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_O()
    {
        using WG        = GetWarpGemm_1<Problem>();
        using CWarpDstr = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths().at(number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    // template <typename Problem>
    template <index_t kNPerBlock,
              index_t kKPerBlock,
              index_t NumWarps,
              index_t KPack,
              index_t KVector>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemSize()
    {
        constexpr index_t warpSize = ck_tile::get_warp_size();

        constexpr index_t kPad = KPack;

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector;
        constexpr index_t LaneGroups = warpSize / LanesPerK;
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

        return NumIssues * NumWarps * (warpSize * KVector + kPad);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemSize_K()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;

        constexpr index_t KPack   = GetSmemKPack_K<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignment_K<Problem>(); // this is for global load
        return GetSingleSmemSize<kNPerBlock, kKPerBlock, NumWarps, KPack, KVector>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemSize_V()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;

        constexpr index_t KPack   = GetSmemKPack_V<Problem>(); // this is for lds
        constexpr index_t KVector = GetAlignment_V<Problem>(); // this is for global load
        return GetSingleSmemSize<kNPerBlock, kKPerBlock, NumWarps, KPack, KVector>();
    }

    // common function for B matrix decriptor for lds used in asyn load
    template <index_t kNPerBlock,
              index_t kKPerBlock,
              index_t kBlockSize,
              index_t NumWarps,
              index_t KPack,
              index_t KVector /*alignment*/,
              index_t SingleSmemSize,
              index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAsyncSmemStoreDesc(number<IBuf> = number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t warpSize = ck_tile::get_warp_size();

        constexpr index_t kPad =
            KPack; // for async-copy, this pad is between warps. Optimize this for lds_read speed

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK =
            kKPerBlock / KVector; // how many lane (within a wave) to load K
        constexpr index_t LaneGroups =
            warpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr index_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr auto desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},  // n0
                       number<LaneGroups>{}, // n1
                       number<NumWarps>{},   // n2
                       number<LanesPerK>{},  // k0
                       number<KVector>{}),   // k1
            make_tuple(number<NumWarps*(warpSize * KVector + kPad)>{},
                       number<kKPerBlock>{},
                       number<warpSize * KVector + kPad>{},
                       number<KVector>{},
                       number<1>{}),
            number<IBuf * SingleSmemSize>{},
            number<KVector>{},
            number<1>{});

        // TODO this layout is hard coded, and will be used in async copy buffer view load
        // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
        constexpr auto desc_issues_warps_lanes = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<NumIssues>{}),
                       make_pass_through_transform(number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        return desc_issues_warps_lanes;
    }

    template <typename Problem, index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSmemStoreDesc_K(number<IBuf> = number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;

        constexpr index_t KPack          = GetSmemKPack_K<Problem>(); // this is for lds
        constexpr index_t KVector        = GetAlignment_K<Problem>(); // this is for global load
        constexpr index_t SingleSmemSize = GetSingleSmemSize_K<Problem>();
        return MakeAsyncSmemStoreDesc<kNPerBlock,
                                      kKPerBlock,
                                      kBlockSize,
                                      NumWarps,
                                      KPack,
                                      KVector,
                                      SingleSmemSize>(number<IBuf>{});
    }

    template <typename Problem, index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSmemStoreDesc_V(number<IBuf> = number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;

        constexpr index_t KPack          = GetSmemKPack_V<Problem>(); // this is for lds
        constexpr index_t KVector        = GetAlignment_V<Problem>(); // this is for global load
        constexpr index_t SingleSmemSize = GetSingleSmemSize_V<Problem>();
        return MakeAsyncSmemStoreDesc<kNPerBlock,
                                      kKPerBlock,
                                      kBlockSize,
                                      NumWarps,
                                      KPack,
                                      KVector,
                                      SingleSmemSize>(number<IBuf>{});
    }

    template <index_t kNPerBlock,
              index_t kKPerBlock,
              index_t kBlockSize,
              index_t NumWarps,
              index_t KPack,
              index_t KVector /*alignment*/,
              index_t SingleSmemSize,
              index_t NumPrefetch>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAsyncSmemLoadDesc()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t warpSize = ck_tile::get_warp_size();
        constexpr index_t kPad     = KPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));
        // constexpr index_t SingleKSize = NumIssues * NumWarps * (warpSize * KVector + kPad);
        // constexpr index_t SingleVSize =
        // MakeVLdsBlockDescriptor<Problem>().get_element_space_size();
        constexpr index_t BufferSize = SingleSmemSize;

        constexpr auto desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumPrefetch>{},        // num_buffers
                                                    number<NumIssues>{},          // n0
                                                    number<NumWarps>{},           // n2
                                                    number<LaneGroups>{},         // n1
                                                    number<kKPerBlock / KPack>{}, // k0
                                                    number<KPack>{}),             // k1
                                         make_tuple(number<BufferSize>{},
                                                    number<NumWarps*(warpSize * KVector + kPad)>{},
                                                    number<warpSize * KVector + kPad>{},
                                                    number<kKPerBlock>{},
                                                    number<KPack>{},
                                                    number<1>{}),
                                         number<KPack>{},
                                         number<1>{});

        constexpr auto desc_ = transform_tensor_descriptor(
            desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<NumPrefetch>{},
                                                number<NumIssues>{},
                                                number<LaneGroups>{},
                                                number<NumWarps>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<0, 1, 3, 2>{}, sequence<4, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return desc_;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSmemLoadDesc_K()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;

        constexpr index_t KPack          = GetSmemKPack_K<Problem>(); // this is for lds
        constexpr index_t KVector        = GetAlignment_K<Problem>(); // this is for global load
        constexpr index_t SingleSmemSize = GetSingleSmemSize_K<Problem>();
        constexpr index_t NumPrefetch    = NumPrefetch_K;
        return MakeAsyncSmemLoadDesc<kNPerBlock,
                                     kKPerBlock,
                                     kBlockSize,
                                     NumWarps,
                                     KPack,
                                     KVector,
                                     SingleSmemSize,
                                     NumPrefetch>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSmemLoadDesc_V()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;

        constexpr index_t KPack          = GetSmemKPack_V<Problem>(); // this is for lds
        constexpr index_t KVector        = GetAlignment_V<Problem>(); // this is for global load
        constexpr index_t SingleSmemSize = GetSingleSmemSize_V<Problem>();
        constexpr index_t NumPrefetch    = NumPrefetch_V;
        return MakeAsyncSmemLoadDesc<kNPerBlock,
                                     kKPerBlock,
                                     kBlockSize,
                                     NumWarps,
                                     KPack,
                                     KVector,
                                     SingleSmemSize,
                                     NumPrefetch>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize_KV()
    {
        // TODO: no K/V Smem overlap
        return NumPrefetchK * GetSingleSmemSize_K() * sizeof(typename Problem::KDataType) +
               NumPrefetchV * GetSingleSmemSize_V() * sizeof(typename Problem::VDataType)
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GetSmemSize_KV<Problem>() + GetSmemSize_Dropout<Problem>(0);
    }

    // this method is only available when Problem::kHasDropout is present
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr std::
        enable_if_t<std::is_convertible_v<decltype(Problem::kHasDropout), bool>, ck_tile::index_t>
        GetSmemSize_Dropout(int)
    {
        if constexpr(Problem::kHasDropout)
        {
            constexpr auto gemm_0 = QXPolicy::template GetBlockGemm_0<Problem>();
            constexpr auto config =
                decltype(gemm_0)::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG                    = remove_cvref_t<decltype(config.template at<0>())>;
            constexpr index_t MWarp     = config.template at<1>();
            constexpr index_t kMPerStep = MWarp * WG::kM;
            constexpr index_t kNPerStep = WG::kN;

            return (kMPerStep + 1) * kNPerStep * sizeof(uint8_t);
        }
        else
        {
            return 0;
        }
    }

    // fallback version if Problem::kHasDropout is not exist
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize_Dropout(...)
    {
        return 0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalDesc_K()
    {
        // async
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck_tile::get_warp_size();

        constexpr index_t KVector = GetAlignment_K<Problem>(); // this is for global load

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr index_t N0 = NumIssues;
        constexpr index_t N1 = LaneGroups;
        constexpr index_t N2 = NumWarps;
        constexpr index_t K0 = LanesPerK;
        constexpr index_t K1 = KVector;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<2>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeGlobalDesc_V()
    {
        // async
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t warpSize   = ck_tile::get_warp_size();

        constexpr index_t KVector = GetAlignment_V<Problem>(); // this is for global load

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr index_t N0 = NumIssues;
        constexpr index_t N1 = LaneGroups;
        constexpr index_t N2 = NumWarps;
        constexpr index_t K0 = LanesPerK;
        constexpr index_t K1 = KVector;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<2>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem, typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalDesc_Bias()
    {
        constexpr index_t MPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t NPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        // Construct C-Block-HostTensor
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        return c_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemm_1()
    {
        auto warp_gemm = [&]() {
            if constexpr(std::is_same_v<typename Problem::KDataType, fp8_t> &&
                         std::is_same_v<typename Problem::VDataType, fp8_t> &&
                         std::is_same_v<typename Problem::OaccDataType, float>)
            {
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<>{};
                // return
                // WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
                //         WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<typename
                //         Problem::PDataType, typename Problem::VDataType>>>{};
            }
            else
            {
                // return WarpGemmMfmaDispatcher<
                //     typename Problem::PDataType,
                //     typename Problem::VDataType,
                //     typename Problem::OaccDataType,
                //     Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                //     Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                //     Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                //     true>{};
                if constexpr(std::is_same_v<typename Problem::PDataType, half_t> &&
                             std::is_same_v<typename Problem::VDataType, half_t> &&
                             std::is_same_v<typename Problem::OaccDataType, float>)
                {
                    // return WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                    return WarpGemmImpl<
                        WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                            WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Raw_vaa>,
                            2>>;
                }
                else if constexpr(std::is_same_v<typename Problem::PDataType, bf16_t> &&
                                  std::is_same_v<typename Problem::VDataType, bf16_t> &&
                                  std::is_same_v<typename Problem::OaccDataType, float>)
                {
                    // return WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                    return WarpGemmImpl<
                        WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                            WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Raw_vaa>,
                            2>>;
                }
            }
        }();
        return warp_gemm;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm_1()
    {
        using BlockGemmProblem = BlockGemmPipelineProblem<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                   Problem::BlockFmhaShape::kN1,
                                   Problem::BlockFmhaShape::kK1>,
                          typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                          typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        auto warp_gemm = GetWarpGemm_1<Problem>();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::PDataType,
                                                 typename Problem::VDataType,
                                                 typename Problem::OaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV2<BlockGemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
