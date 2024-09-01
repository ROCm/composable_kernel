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

struct FusedMoePipelineNSplit2Policy
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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_O()
    {
        if constexpr(Problem::Traits::OAtomic == 0)
        {
            // pack fp16/bf16 atomic
            static_assert(sizeof(typename Problem::ODataType) == 2);
            return 2;
        }
        else if constexpr(Problem::Traits::OAtomic == 1)
        {
            // fp32 atomic
            return 1;
        }
        else
        {
            return 16 / sizeof(typename Problem::ODataType);
        }
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

#if 0
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWaveFlattenShape()
    {
        using WarpGemm = GetWarpGemm0<Problem>{}; // assume warpgemm0/1 are the same

        constexpr index_t Kv = GetAlignment_G<{Problem}>();
        constexpr index_t Nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t Kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
        return sequence<Kw, Nw, Kv>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockTileNrKr()
    {
        using WarpGemm = GetWarpGemm0<Problem>{}; // assume warpgemm0/1 are the same

        constexpr index_t Kv = GetAlignment_G<{Problem}>();
        constexpr index_t Nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t Kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
        return sequence<Problem::FusedMoeTileShape::kBlockK_0 / Nw,
                        Problem::FusedMoeTileShape::kBlockK_0 / (Kw * Kv)>{};
    }
#endif

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeSingleBuffer()
    {
        constexpr a_sld_desc = MakeLdsLoadDesc_A<Problem>();
        constexpr a_sst_desc = MakeLdsStoreDesc_A<Problem>();
        static_assert(a_sld_desc.get_element_space_size() == a_sst_desc.get_element_space_size());
        return a_sld_desc.get_element_space_size();
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
        constexpr index_t K_vec = Alignment;
        constexpr index_t K_rem = KPerBlock / K_vec;

        if constexpr(get_warp_size() <= K_rem)
        {
            static_assert(K_rem % get_warp_size() == 0);
            constexpr index_t K_lan = get_warp_size(); // lane within same wave is along gemm-k
            constexpr index_t K_wav = K_rem / get_warp_size();
            static_assert(K_wav <= NumWarps, "do not support thread has repeat along K yet");
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
                    tuple<sequence<M_rep, M_wav, M_lan>, sequence<K_lan, K_vec>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<1>, sequence<2, 0>>,
                    sequence<1, 2>,
                    sequence<0, 1>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetMatrixCoreSwizzledBlockTIle_0()
    {
        if constexpr(Problem::Traits::PermuteStyle ==
                     FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv)
        {
            using WarpGemm = GetWarpGemm0<Problem>{}; // assume warpgemm0/1 are the same
            constexpr index_t NPerBlock = Problem::FusedMoeTileShape::kBlockN_0;
            constexpr index_t KPerBlock = Problem::FusedMoeTileShape::kBlockK_0;

            constexpr index_t Kv = GetAlignment_G<{Problem}>();
            constexpr index_t Nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t Kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

            static_assert(KPerBlock % (K1 * K2) == 0);
            constexpr index_t Nr = NPerBlock / Nw;
            constexpr index_t Kr = KPerBlock / (Kv * Kw);

            return sequence<Nr, Kr, Kw * Nw * Kv>{}; // 3D
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
              FusedMoeWeightPermuteEnum PermuteStyle =
                  FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_MatrixCore_Swizzled()
    {
        static_assert(Alignment % WarpGemm::WarpGemmAttribute::Impl::kABKPerLane == 0);

        if constexpr(PermuteStyle == FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv)
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

    template <typename Problem, index_t NSplits = 2>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_G(number<NSplits> = {})
    {
        constexpr auto PermuteStype = Problem::Traits::PermuteStyle;
        if constexpr(PermuteStype == FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv)
        {
            constexpr index_t kNPerBlock      = Problem::FusedMoeTileShape::kBlockN_0;
            constexpr index_t kKPerBlock      = Problem::FusedMoeTileShape::kBlockK_0;
            constexpr index_t WavesPerBlock_N = Problem::FusedMoeTileShape::kBlockWarpsN_0;
            constexpr index_t WavesPerBlock_K = Problem::FusedMoeTileShape::kBlockWarpsK_0;
            using WarpGemm                    = remove_cvref_t<GetWarpGemm0<Problem>()>;
            constexpr index_t Alignment       = GetAlignment_G<Problem>();
            return MakeGlobalTileDistribution_MatrixCore_Swizzled<kNPerBlock,
                                                                  kKPerBlock,
                                                                  WavesPerBlock_N,
                                                                  WavesPerBlock_K,
                                                                  WarpGemm,
                                                                  Alignment,
                                                                  PermuteStype>();
        }
    }

    template <typename Problem, index_t NSplits = 2>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_U(number<NSplits> = {})
    {
        constexpr auto PermuteStype = Problem::Traits::PermuteStyle;
        if constexpr(PermuteStype == FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv)
        {
            constexpr index_t kNPerBlock      = Problem::FusedMoeTileShape::kBlockN_0;
            constexpr index_t kKPerBlock      = Problem::FusedMoeTileShape::kBlockK_0;
            constexpr index_t WavesPerBlock_N = Problem::FusedMoeTileShape::kBlockWarpsN_0;
            constexpr index_t WavesPerBlock_K = Problem::FusedMoeTileShape::kBlockWarpsK_0;
            using WarpGemm                    = remove_cvref_t<GetWarpGemm0<Problem>()>;
            constexpr index_t Alignment       = GetAlignment_U<Problem>();
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
        if constexpr(PermuteStype == FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv)
        {
            constexpr index_t kNPerBlock      = Problem::FusedMoeTileShape::kBlockN_1;
            constexpr index_t kKPerBlock      = Problem::FusedMoeTileShape::kBlockK_1;
            constexpr index_t WavesPerBlock_N = Problem::FusedMoeTileShape::kBlockWarpsN_1;
            constexpr index_t WavesPerBlock_K = Problem::FusedMoeTileShape::kBlockWarpsK_1;
            using WarpGemm                    = remove_cvref_t<GetWarpGemm1<Problem>()>;
            constexpr index_t Alignment       = GetAlignment_D<Problem>();
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
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsStoreDesc_A()
    {
        // A async->LDS
        constexpr index_t kMPerBlock = Problem::FusedMoeTileShape::kBlockM_0;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kBlockK_0;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t warpSize   = ck_tile::get_warp_size();
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;

        constexpr index_t KPack   = GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t kVector = GetAlignment_A<Problem>(); // async copy 1 dword
        constexpr index_t kPad    = KPack;                     // pad between warps

        static_assert(kKPerBlock % kVector == 0);
        constexpr index_t LanesPerK = kKPerBlock / kVector; // how many thread loading K
        if constexpr(LanesPerK >= warpSize)
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
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsLoadDesc_A()
    {
        // A async->LDS
        // Note that, this descriptor is only to construct the layout inside LDS
        // in real Gemm pipeline, ds_read may not follow this pattern
        // (may follow that in tile_distribution)
        // below code is almost the same as SmemStore dist, with difference:
        //  1). modify the GuaranteedLastDimensionVectorLength of naive tensor desc
        //  2). return discriptor is in NxK 2d layout
        constexpr index_t kMPerBlock = Problem::FusedMoeTileShape::kBlockM_0;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kBlockK_0;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t warpSize   = ck_tile::get_warp_size();
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;

        constexpr index_t KPack   = GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t kVector = GetAlignment_A<Problem>(); // async copy 1 dword
        constexpr index_t kPad    = KPack;                     // pad between warps

        static_assert(kKPerBlock % kVector == 0);
        constexpr index_t LanesPerK = kKPerBlock / kVector; // how many thread loading K
        if constexpr(LanesPerK >= warpSize)
        {
            // need multiple waves to load K
            static_assert(LanesPerK % warpSize == 0);
            constexpr index_t wavesPerK = LanesPerK / warpSize;
            if constexpr(wavesPerK >= NumWarps)
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

    template <typename Problem, index_t NSplits = 2>
    CK_TILE_HOST_DEVICE constexpr auto MakeCBlockTile_Gemm0(number<NSplits> = {}) const
    {
        using TileShape = remove_cvref_t<typename Problem::FusedMoeTileShape>;

        constexpr index_t BlockWarpsM = TileShape::kBlockWarpsM_0;
        constexpr index_t BlockWarpsN = TileShape::kBlockWarpsN_0;
        constexpr index_t WarpRepeatM = TileShape::kWarpRepeatM_0;
        constexpr index_t WarpRepeatN = TileShape::kWarpRepeatN_0;
        using WarpGemm                = remove_cvref_t<decltype(GetWarpGemm0<Problem>())>;
        static_assert(WarpRepeatN % NSplits == 0);

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<WarpRepeatM, BlockWarpsM>, sequence<WarpRepeatN / NSplits, BlockWarpsN>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE constexpr auto MakeCBlockTile_Gemm1() const
    {
        using TileShape = remove_cvref_t<typename Problem::FusedMoeTileShape>;

        constexpr index_t BlockWarpsM = TileShape::kBlockWarpsM_1;
        constexpr index_t BlockWarpsN = TileShape::kBlockWarpsN_1;
        constexpr index_t WarpRepeatM = TileShape::kWarpRepeatM_1;
        constexpr index_t WarpRepeatN = TileShape::kWarpRepeatN_1;
        using WarpGemm                = remove_cvref_t<decltype(GetWarpGemm1<Problem>())>;

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<WarpRepeatM, BlockWarpsM>, sequence<WarpRepeatN, BlockWarpsN>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetMatrixCoreSwizzledBlockTIle_0()
    {
        if constexpr(Problem::Traits::PermuteStyle ==
                     FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv)
        {
            using WarpGemm = GetWarpGemm0<Problem>{}; // assume warpgemm0/1 are the same
            constexpr index_t NPerBlock = Problem::FusedMoeTileShape::kBlockN_0;
            constexpr index_t KPerBlock = Problem::FusedMoeTileShape::kBlockK_0;

            constexpr index_t Kv = GetAlignment_G<{Problem}>();
            constexpr index_t Nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t Kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

            static_assert(KPerBlock % (K1 * K2) == 0);
            constexpr index_t Nr = NPerBlock / Nw;
            constexpr index_t Kr = KPerBlock / (Kv * Kw);

            return sequence<Nr, Kr, Kw * Nw * Kv>{}; // 3D
        }
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetMatrixCoreSwizzledBlockTIle_1()
    {
        if constexpr(Problem::Traits::PermuteStyle ==
                     FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv)
        {
            using WarpGemm = GetWarpGemm1<Problem>{}; // assume warpgemm0/1 are the same
            constexpr index_t NPerBlock = Problem::FusedMoeTileShape::kBlockN_1;
            constexpr index_t KPerBlock = Problem::FusedMoeTileShape::kBlockK_1;

            constexpr index_t Kv = GetAlignment_G<{Problem}>();
            constexpr index_t Nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t Kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

            static_assert(KPerBlock % (K1 * K2) == 0);
            constexpr index_t Nr = NPerBlock / Nw;
            constexpr index_t Kr = KPerBlock / (Kv * Kw);

            return sequence<Nr, Kr, Kw * Nw * Kv>{}; // 3D
        }
    }
};
} // namespace ck_tile
