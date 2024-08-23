// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

struct FusedMoePipelinePolicy
{
    CK_TILE_HOST_DEVICE static constexpr index_t GetAsyncCopyDwords()
    {
        // TODO:
        return 1;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_A()
    {
        // using async
        static constexpr index_t copy_bytes = 4 * GetAsyncCopyDwords();
        static constexpr index_t data_bytes = sizeof(typename Problem::ADataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_G()
    {
        static constexpr index_t copy_bytes = [&]() { return 16; }();
        static constexpr index_t data_bytes = sizeof(typename Problem::GDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_U()
    {
        static constexpr index_t copy_bytes = [&]() { return 16; }();
        static constexpr index_t data_bytes = sizeof(typename Problem::UDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_D()
    {
        static constexpr index_t copy_bytes = [&]() { return 16; }();
        static constexpr index_t data_bytes = sizeof(typename Problem::DDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename DataType_>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack()
    {
        // TODO: this is for 3d layout
        return 16 / sizeof(remove_cvref_t<typename Problem::DataType_>);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack_A()
    {
        return GetSmemKPack<typename Problem::ADataType>();
    }

    template <index_t MPerBlock, index_t KPerBlock, index_t NumWarps, index_t Alignment>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_SimpleMxK()
    {
        constexpr index_t K_vec = Alignment constexpr index_t K_rem = KPerBlock / K_vec;

        if constexpr(get_warp_size() < K_rem)
        {
            static_assert(K_rem % get_warp_size() == 0);
            constexpr index_t K_lan = get_warp_size(); // lane within same wave is along gemm-k
            constexpr index_t K_wav = K_rem / get_warp_size();
            static_assert(K_wav <= NumWarps, "not not support thread has repeat along K yet");
            constexpr index_t M_wav = NumWarps / K_wav;
            static_assert(MPerBlock % M_wav == 0, "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / M_wav;

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_wav>, sequence<K_wav, K_lan, K_vec>>,
                    tuple<sequence<1, 2>, sequence<2>>,
                    tuple<sequence<1, 0>, sequence<1>>,
                    sequence<1, 2>,
                    sequence<0, 2>>{});
        }
        else
        {
            constexpr index_t K_lan = K_rem;
            constexpr index_t M_lan = get_warp_size() / K_lan;
            constexpr index_t M_wav = NumWarps;
            static_assert(MPerBlock % (M_lan * M_wav) == 0,
                          "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / (M_lan * M_wav);
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_wav, M_lan>, sequence<K_lan, K_vec>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<1>, sequence<2, 0>>,
                    sequence<1, 2>,
                    sequence<0, 1>>{});
        }
    }

    // optimized version for async
    template <index_t MPerBlock, index_t KPerBlock, index_t NumWarps, index_t Alignment>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_SimpleMxK_Async()
    {
        constexpr index_t K_vec = Alignment constexpr index_t K_rem = KPerBlock / K_vec;

        if constexpr(get_warp_size() < K_rem)
        {
            static_assert(K_rem % get_warp_size() == 0);
            constexpr index_t K_lan = get_warp_size(); // lane within same wave is along gemm-k
            constexpr index_t K_wav = K_rem / get_warp_size();
            static_assert(K_wav <= NumWarps, "not not support thread has repeat along K yet");
            constexpr index_t M_wav = NumWarps / K_wav;
            static_assert(MPerBlock % M_wav == 0, "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / M_wav;
            // NOTE: no swap, but hard to avoid LDS bank conflict
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_wav>, sequence<K_wav, K_lan, K_vec>>,
                    tuple<sequence<1, 2>, sequence<2>>,
                    tuple<sequence<1, 0>, sequence<1>>,
                    sequence<1, 2>,
                    sequence<0, 2>>{});
        }
        else
        {
            constexpr index_t K_lan = K_rem;
            constexpr index_t M_lan = get_warp_size() / K_lan;
            constexpr index_t M_wav = NumWarps;
            static_assert(MPerBlock % (M_lan * M_wav) == 0,
                          "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / (M_lan * M_wav);
            // NOTE: swapped for LDS load bank conflict free
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_lan, M_wav>, sequence<K_lan, K_vec>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<2>, sequence<1, 0>>,
                    sequence<1, 2>,
                    sequence<0, 1>>{});
        }
    }

    // Caution: this will require global memory pre-shuffled to follow the mfma layout
    // to maximize the L1/L2 channel while skip LDS
    template <index_t NPerBlock,
              index_t KPerBlock,
              index_t WavesPerBlock_N,
              index_t WavesPerBlock_K,
              typename WarpGemm,
              index_t Alignment,
              FusedMoePermuteStyle PermuteStyle = FusedMoePermuteStyle::permute_b_nr_kr_kw_nw_kv>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_MatrixCore_Swizzled()
    {
        static_assert(Alignment % WarpGemm::WarpGemmAttribute::Impl::kABKPerLane == 0);

        if constexpr(PermuteStyle == FusedMoePermuteStyle::permute_b_nr_kr_kw_nw_kv)
        {
            // permute_b_nr_kr_kw_nw_kv or permute_b_nr_kr_waveflatten
            constexpr index_t Kv = Alignment;
            constexpr index_t Nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t Kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

            static_assert(KPerBlock % (K1 * K2) == 0);
            constexpr index_t Nr = NPerBlock / Nw;
            constexpr index_t Kr = KPerBlock / (Kv * Kw);

            constexpr index_t Nr_p = WavesPerBlock_N;
            constexpr index_t Kr_p = WavesPerBlock_K;
            constexpr index_t Nr_y = Nr / Nr_p;
            constexpr index_t Kr_y = Kr / Kr_p;

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>, // 0
                    // major       1                     2                     3
                    // minor       0     1               0     1               0   1   2
                    tuple<sequence<Nr_y, Nr_p>, sequence<Kr_y, Kr_p>, sequence<Kw, Nw, Kv>>,

                    //            Nr_p, Kr_p         Kw Nw
                    tuple<sequence<1, 2>, sequence<3, 3>>,
                    tuple<sequence<1, 1>, sequence<0, 1>>,

                    //       Nr_y Kr_y Kv
                    sequence<1, 2, 3>,
                    sequence<0, 0, 2>>{});
            // clang-format on
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_A()
    {
        constexpr index_t kMPerBlock = Problem::FusedMoeTileShape::kM_a;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment  = GetAlignment_A<Problem>();
        return MakeGlobalTileDistribution_SimpleMxK_Async<kMPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_G()
    {
        constexpr auto PermuteStype = Problem::Traits::PermuteStyle;
        if constexpr(PermuteStype == FusedMoePermuteStyle::permute_b_nr_kr_kw_nw_kv)
        {
            constexpr index_t kNPerBlock      = Problem::FusedMoeTileShape::kN_u;
            constexpr index_t kKPerBlock      = Problem::FusedMoeTileShape::kK_a;
            constexpr index_t WavesPerBlock_N = Problem::Gemm0BlockWarps {}
            ::at(number<1>{});
            constexpr index_t WavesPerBlock_K = Problem::Gemm0BlockWarps {}
            ::at(number<2>{});
            using WarpGemm              = remove_cvref_t<GetWarpGemm0<Problem>()>;
            constexpr index_t Alignment = GetAlignment_G<Problem>();
            return MakeGlobalTileDistribution_MatrixCore_Swizzled<kNPerBlock,
                                                                  kKPerBlock,
                                                                  WavesPerBlock_N,
                                                                  WavesPerBlock_K,
                                                                  WarpGemm,
                                                                  Alignment,
                                                                  PermuteStype>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_U()
    {
        constexpr auto PermuteStype = Problem::Traits::PermuteStyle;
        if constexpr(PermuteStype == FusedMoePermuteStyle::permute_b_nr_kr_kw_nw_kv)
        {
            constexpr index_t kNPerBlock      = Problem::FusedMoeTileShape::kN_u;
            constexpr index_t kKPerBlock      = Problem::FusedMoeTileShape::kK_a;
            constexpr index_t WavesPerBlock_N = Problem::Gemm0BlockWarps {}
            ::at(number<1>{});
            constexpr index_t WavesPerBlock_K = Problem::Gemm0BlockWarps {}
            ::at(number<2>{});
            using WarpGemm              = remove_cvref_t<GetWarpGemm0<Problem>()>;
            constexpr index_t Alignment = GetAlignment_U<Problem>();
            return MakeGlobalTileDistribution_MatrixCore_Swizzled<kNPerBlock,
                                                                  kKPerBlock,
                                                                  WavesPerBlock_N,
                                                                  WavesPerBlock_K,
                                                                  WarpGemm,
                                                                  Alignment,
                                                                  PermuteStype>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_D()
    {
        constexpr auto PermuteStype = Problem::Traits::PermuteStyle;
        if constexpr(PermuteStype == FusedMoePermuteStyle::permute_b_nr_kr_kw_nw_kv)
        {
            constexpr index_t kNPerBlock      = Problem::FusedMoeTileShape::kN_d;
            constexpr index_t kKPerBlock      = Problem::FusedMoeTileShape::kK_y;
            constexpr index_t WavesPerBlock_N = Problem::Gemm1BlockWarps {}
            ::at(number<1>{});
            constexpr index_t WavesPerBlock_K = Problem::Gemm1BlockWarps {}
            ::at(number<2>{});
            using WarpGemm              = remove_cvref_t<GetWarpGemm1<Problem>()>;
            constexpr index_t Alignment = GetAlignment_D<Problem>();
            return MakeGlobalTileDistribution_MatrixCore_Swizzled<kNPerBlock,
                                                                  kKPerBlock,
                                                                  WavesPerBlock_N,
                                                                  WavesPerBlock_K,
                                                                  WarpGemm,
                                                                  Alignment,
                                                                  PermuteStype>();
        }
    }

    template <index_t MPerBlock,
              index_t KPerBlock,
              index_t NumWarps,
              index_t Alignment,
              index_t KPack,
              index_t NumPrefetch>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSmemLoadTileDescriptor_SimpleMxK_Async()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr index_t kBlockSize = ck_tile::get_warp_size() * NumWarps; // Problem::kBlockSize;
        constexpr index_t warpSize   = ck_tile::get_warp_size();

        constexpr index_t KVector = Alignment; // this is for global load
        constexpr index_t kPad    = KPack;     // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= KPerBlock && warpSize * KVector % KPerBlock == 0);
        constexpr index_t LanesPerK  = KPerBlock / KVector;  // within a wave
        constexpr index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr index_t NumIssues  = MPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == MPerBlock * KPerBlock / (kBlockSize * KVector));

        constexpr index_t BufferSize = NumIssues * NumWarps * (warpSize * KVector + kPad);

        constexpr auto lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumPrefetch>{},       // num_buffers
                                                    number<NumIssues>{},         // n0
                                                    number<NumWarps>{},          // n2
                                                    number<LaneGroups>{},        // n1
                                                    number<KPerBlock / KPack>{}, // k0
                                                    number<KPack>{}),            // k1
                                         make_tuple(number<BufferSize>{},
                                                    number<NumWarps*(warpSize * KVector + kPad)>{},
                                                    number<warpSize * KVector + kPad>{},
                                                    number<KPerBlock>{},
                                                    number<KPack>{},
                                                    number<1>{}),
                                         number<KPack>{},
                                         number<1>{});

        constexpr auto lds_block_desc = transform_tensor_descriptor(
            lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<NumPrefetch>{},
                                                number<NumIssues>{},
                                                number<LaneGroups>{},
                                                number<NumWarps>{})),
                make_merge_transform(make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<0, 1, 3, 2>{}, sequence<4, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsStoreBlockDescriptor_A()
    {
        // A async->LDS
        constexpr index_t kMPerBlock = Problem::FusedMoeTileShape::kM_a;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t warpSize   = ck_tile::get_warp_size();
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;

        constexpr index_t KPack   = GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t kVector = GetAlignment_A<Problem>(); // async copy 1 dword
        constexpr index_t kPad    = KPack;                     // pad between warps

        static_assert(kKPerBlock % kVector == 0);
        constexpr index_t LanesPerK = kKPerBlock / kVector; // how many thread loading K
        if constexpr(LanesPerK > warpSize)
        {
            // need multiple waves to load K
            static_assert(LanesPerK % warpSize == 0);
            constexpr index_t wavesPerK = LanesPerK / warpSize;
            if constexpr(wavesPerK > NumWarps)
            {
                // TODO: need multiple issues along K to load all data
            }
            else
            {
                constexpr index_t wavesPerM     = NumWarps / wavesPerK;
                constexpr index_t NumIssues     = kMPerBlock / wavesPerM;
                constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<NumIssues>{},                             // m0
                               number<wavesPerM>{},                             // m1
                               number<wavesPerK>{},                             // k0
                               number<warpSize>{},                              // k1
                               number<KVector>{}),                              // k2
                    make_tuple(number<NumWarps*(warpSize * KVector + kPad)>{},  // m0
                               number<wavesPerK*(warpSize * KVector + kPad)>{}, // m1
                               number<warpSize * KVector + kPad>{},             // k0
                               number<KVector>{},                               // k1
                               number<1>{}),                                    // k2
                    number<KVector>{}, // lds store vector(actually no explicit store)
                    number<1>{});

                constexpr auto lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
                    lds_block_desc_0,
                    make_tuple(
                        make_pass_through_transform(number<NumIssues>{}),
                        make_merge_transform(make_tuple(number<wavesPerM>{}, number<wavesPerK>{})),
                        make_merge_transform(make_tuple(number<warpSize>{}, number<KVector>{}))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                return lds_block_desc_issues_warps_lanes;
            }
        }
        else
        {
            // lanes within a wave load different M but same K
            static_assert(warpSize % LanesPerK == 0);
            constexpr index_t LaneGroups = warpSize / LanesPerK; // along m
            constexpr index_t NumIssues  = kMPerBlock / (LaneGroups * NumWarps);

            constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumIssues>{},                            // m0
                           number<LaneGroups>{},                           // m1
                           number<NumWarps>{},                             // m2
                           number<LanesPerK>{},                            // k0
                           number<KVector>{}),                             // k1
                make_tuple(number<NumWarps*(warpSize * KVector + kPad)>{}, // m0
                           number<kKPerBlock>{},                           // m1
                           number<warpSize * KVector + kPad>{},            // m2
                           number<KVector>{},                              // k0
                           number<1>{}),                                   // k1
                number<KVector>{}, // lds store vector(actually no explicit store)
                number<1>{});

            constexpr auto lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
                lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<NumIssues>{}),
                           make_pass_through_transform(number<NumWarps>{}),
                           make_merge_transform(make_tuple(
                               number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
                make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

            return lds_block_desc_issues_warps_lanes;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSmemLoadTileDistribution_A()
    {
        // A async->LDS
        // Note that, this descriptor is only to construct the layout inside LDS
        // in real Gemm pipeline, ds_read may not follow this pattern
        // (may follow that in tile_distribution)
        // below code is almost the same as SmemStore dist, with difference:
        //  1). modify the GuaranteedLastDimensionVectorLength of naive tensor desc
        //  2). return discriptor is in NxK 2d layout
        constexpr index_t kMPerBlock = Problem::FusedMoeTileShape::kM_a;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t warpSize   = ck_tile::get_warp_size();
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;

        constexpr index_t KPack   = GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t kVector = GetAlignment_A<Problem>(); // async copy 1 dword
        constexpr index_t kPad    = KPack;                     // pad between warps

        static_assert(kKPerBlock % kVector == 0);
        constexpr index_t LanesPerK = kKPerBlock / kVector; // how many thread loading K
        if constexpr(LanesPerK > warpSize)
        {
            // need multiple waves to load K
            static_assert(LanesPerK % warpSize == 0);
            constexpr index_t wavesPerK = LanesPerK / warpSize;
            if constexpr(wavesPerK > NumWarps)
            {
                // TODO: need multiple issues along K to load all data
            }
            else
            {
                constexpr index_t wavesPerM     = NumWarps / wavesPerK;
                constexpr index_t NumIssues     = kMPerBlock / wavesPerM;
                constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<NumIssues>{},                             // m0
                               number<wavesPerM>{},                             // m1
                               number<wavesPerK>{},                             // k0
                               number<warpSize>{},                              // k1
                               number<KVector>{}),                              // k2
                    make_tuple(number<NumWarps*(warpSize * KVector + kPad)>{},  // m0
                               number<wavesPerK*(warpSize * KVector + kPad)>{}, // m1
                               number<warpSize * KVector + kPad>{},             // k0
                               number<KVector>{},                               // k1
                               number<1>{}),                                    // k2
                    number<KPack>{},                                            // lds load vector
                    number<1>{});

                constexpr auto lds_desc_m_k = transform_tensor_descriptor(
                    lds_block_desc_0,
                    make_tuple(
                        make_merge_transform(make_tuple(number<NumIssues>{}, number<wavesPerM>{})),
                        make_merge_transform(make_tuple(
                            number<wavesPerK>{}, number<warpSize>{}, number<KVector>{}))),
                    make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return lds_desc_m_k;
            }
        }
        else
        {
            // lanes within a wave load different M but same K
            static_assert(warpSize % LanesPerK == 0);
            constexpr index_t LaneGroups = warpSize / LanesPerK; // along m
            constexpr index_t NumIssues  = kMPerBlock / (LaneGroups * NumWarps);

            constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumIssues>{},                            // m0
                           number<LaneGroups>{},                           // m1
                           number<NumWarps>{},                             // m2
                           number<LanesPerK>{},                            // k0
                           number<KVector>{}),                             // k1
                make_tuple(number<NumWarps*(warpSize * KVector + kPad)>{}, // m0
                           number<kKPerBlock>{},                           // m1
                           number<warpSize * KVector + kPad>{},            // m2
                           number<KVector>{},                              // k0
                           number<1>{}),                                   // k1
                number<KPack>{},                                           // lds load vector
                number<1>{});

            constexpr auto lds_desc_m_k = transform_tensor_descriptor(
                lds_block_desc_0,
                make_tuple(
                    make_merge_transform(
                        make_tuple(number<NumIssues>{}, number<LaneGroups>{}, number<NumWarps>{})),
                    make_merge_transform(make_tuple(number<LanesPerK>{}, number<KVector>{}))),
                make_tuple(sequence<0, 1, 2>{}, sequence<3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc_m_k;
        }
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeASmemStoreTileDistribution()
    {
        constexpr index_t kMPerBlock  = Problem::FusedMoeTileShape::kM_a;
        constexpr index_t kKPerBlock  = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps    = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment   = GetAlignment_A<Problem>();
        constexpr index_t KPack       = GetSmemKPack_A<Problem>();
        constexpr index_t NumPrefetch = Problem::Traits::NumPrefetchA;

        return MakeSmemStoreBlockDescriptor_SimpleMxK_Async<kMperBlock,
                                                            kKPerBlock,
                                                            kBlockSize,
                                                            NumWarps,
                                                            KPack,
                                                            Alignment>();
    }

#if 0
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGSmemLoadTileDistribution()
    {
        constexpr index_t kNPerBlock  = Problem::FusedMoeTileShape::kN_g;
        constexpr index_t kKPerBlock  = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps    = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment   = GetAlignment_G<Problem>();
        constexpr index_t KPack       = GetSmemKPackG<Problem>();
        constexpr index_t NumPrefetch = Problem::Traits::NumPrefetchG;

        return MakeSmemLoadTileDescriptor_SimpleMxK_Async<kNPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment,
                                                          KPack,
                                                          NumPrefetch>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGSmemStoreTileDistribution()
    {
        constexpr index_t kNPerBlock  = Problem::FusedMoeTileShape::kN_g;
        constexpr index_t kKPerBlock  = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps    = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment   = GetAlignment_G<Problem>();
        constexpr index_t KPack       = GetSmemKPackG<Problem>();
        constexpr index_t NumPrefetch = Problem::Traits::NumPrefetchG;

        return MakeSmemStoreTileDescriptor_SimpleMxK_Async<kNPerBlock,
                                                           kKPerBlock,
                                                           NumWarps,
                                                           Alignment,
                                                           KPack,
                                                           NumPrefetch>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeUSmemLoadTileDistribution()
    {
        constexpr index_t kNPerBlock  = Problem::FusedMoeTileShape::kN_u;
        constexpr index_t kKPerBlock  = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps    = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment   = GetAlignment_U<Problem>();
        constexpr index_t KPack       = GetSmemKPackU<Problem>();
        constexpr index_t NumPrefetch = Problem::Traits::NumPrefetchU;

        return MakeSmemLoadTileDescriptor_SimpleMxK_Async<kNPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment,
                                                          KPack,
                                                          NumPrefetch>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDSmemLoadTileDistribution()
    {
        constexpr index_t kNPerBlock  = Problem::FusedMoeTileShape::kN_d;
        constexpr index_t kKPerBlock  = Problem::FusedMoeTileShape::kK_y;
        constexpr index_t NumWarps    = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment   = GetAlignment_D<Problem>();
        constexpr index_t KPack       = GetSmemKPackD<Problem>();
        constexpr index_t NumPrefetch = Problem::Traits::NumPrefetchD;

        return MakeSmemLoadTileDescriptor_SimpleMxK_Async<kNPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment,
                                                          KPack,
                                                          NumPrefetch>();
    }
#endif

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemm0()
    {
        return WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                      typename Problem::GDataType,
                                      typename Problem::AccDataType,
                                      Problem::FusedMoeTileShape::Gemm0WarpTile::at(number<0>{}),
                                      Problem::FusedMoeTileShape::Gemm0WarpTile::at(number<1>{}),
                                      Problem::FusedMoeTileShape::Gemm0WarpTile::at(number<2>{}),
                                      true /*TransposeC*/>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemm1()
    {
        return WarpGemmMfmaDispatcher<typename Problem::YDataType,
                                      typename Problem::DDataType,
                                      typename Problem::AccDataType,
                                      Problem::FusedMoeTileShape::Gemm1WarpTile::at(number<0>{}),
                                      Problem::FusedMoeTileShape::Gemm1WarpTile::at(number<1>{}),
                                      Problem::FusedMoeTileShape::Gemm1WarpTile::at(number<2>{}),
                                      true /*TransposeC*/>{};
    }
#if 0
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetGemm0()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::ADataType,
                                     typename Problem::GDataType, // UDataType is the same
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::FusedMoeTileShape::kM_a,
                                                   Problem::FusedMoeTileShape::kN_g * 2,
                                                   Problem::FusedMoeTileShape::kK_a>>;

        constexpr auto warp_gemm = []() {
            return WarpGemmMfmaDispatcher<
                typename Problem::ADataType,
                typename Problem::GDataType,
                typename Problem::AccDataType,
                Problem::FusedMoeTileShape::Gemm0WarpTile::at(number<0>{}),
                Problem::FusedMoeTileShape::Gemm0WarpTile::at(number<1>{}),
                Problem::FusedMoeTileShape::Gemm0WarpTile::at(number<2>{}),
                true /*TransposeC*/>{};
        }();

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<
            typename Problem::ADataType,
            typename Problem::GDataType,
            typename Problem::AccDataType,
            typename Problem::FusedMoeTileShape::Gemm0BlockWarps,
            decltype(warp_gemm)>;

        return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetGemm1()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::YDataType,
                                     typename Problem::DDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::FusedMoeTileShape::kM_a,
                                                   Problem::FusedMoeTileShape::kN_d,
                                                   Problem::FusedMoeTileShape::kK_y>>;

        constexpr auto warp_gemm = []() {
            return WarpGemmMfmaDispatcher<
                typename Problem::YDataType,
                typename Problem::DDataType,
                typename Problem::AccDataType,
                Problem::FusedMoeTileShape::Gemm1WarpTile::at(number<0>{}),
                Problem::FusedMoeTileShape::Gemm1WarpTile::at(number<1>{}),
                Problem::FusedMoeTileShape::Gemm1WarpTile::at(number<2>{}),
                true /*TransposeC*/>{};
        }();

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<
            typename Problem::YDataType,
            typename Problem::DDataType,
            typename Problem::AccDataType,
            typename Problem::FusedMoeTileShape::Gemm1BlockWarps,
            decltype(warp_gemm)>;

        return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }
#endif
};
} // namespace ck_tile
