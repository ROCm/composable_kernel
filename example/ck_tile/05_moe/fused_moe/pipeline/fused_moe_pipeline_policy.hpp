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
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA()
    {
        // using async
        static constexpr index_t copy_bytes = 4 * GetAsyncCopyDwords();
        static constexpr index_t data_bytes = sizeof(typename Problem::ADataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentG()
    {
        static constexpr index_t copy_bytes = [&]() {
            if constexpr(Problem::Traits::GateUpPreShuffled)
            {
                return 4 * 4;
            }
            else
            {
                return 4 * GetAsyncCopyDwords();
            }
        }();
        static constexpr index_t data_bytes = sizeof(typename Problem::GDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentU()
    {
        static constexpr index_t copy_bytes = [&]() {
            if constexpr(Problem::Traits::GateUpPreShuffled)
            {
                return 4 * 4;
            }
            else
            {
                return 4 * GetAsyncCopyDwords();
            }
        }();
        static constexpr index_t data_bytes = sizeof(typename Problem::UDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentD()
    {
        static constexpr index_t copy_bytes = [&]() {
            if constexpr(Problem::Traits::DownPreShuffled)
            {
                return 4 * 4;
            }
            else
            {
                return 4 * GetAsyncCopyDwords();
            }
        }();
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
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackA()
    {
        return GetSmemKPack<typename Problem::ADataType>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackG()
    {
        return GetSmemKPack<typename Problem::GDataType>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackU()
    {
        return GetSmemKPack<typename Problem::UDataType>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackD()
    {
        return GetSmemKPack<typename Problem::DDataType>();
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
    /*

        (b) n0 n1 n2 k0 k1 k2

                     klanes
                     |
            nr 4  kr 4  16 8
        (b) n0 n1 k0 k1 n2 k2 -> kthreads
               |        |
               V        V
               waves   nlanes

                    klanes
                     |
            nr kr  4 4  16 8
        (b) n0 k0 n1 k1 n2 k2 -> kthreads
                   |    |
                   V    V
               waves   nlanes
    */
    template <typename BlockTile, typename BlockWarps, typename WarpGemm, index_t Alignment>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_MatrixCore_Swizzled_NxK()
    {
        static_assert(Alignment % WarpGemm::WarpGemmAttribute::Impl::kABKPerLane == 0);
        static_assert(BlockWarps{}.at(number<0>{}) == 1 && BlockWarps{}.at(number<2>{}) == 1);
        static constexpr index_t NumWarps =
            reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});

        constexpr index_t NPerBlock = BlockTile{}.at(number<1>{});
        constexpr index_t KPerBlock = BlockTile{}.at(number<2>{});

        constexpr index_t K2 = Alignment;
        constexpr index_t N2 = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t K1 = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t N1 = NumWarps;

        static_assert(NPerBlock % (N1 * N2) == 0);
        static_assert(KPerBlock % (K1 * K2) == 0);

        constexpr index_t K0 = KPerBlock / (K1 * K2);
        constexpr index_t N0 = NPerBlock / (N1 * N2);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M_rep, M_wav>, sequence<K_wav, K_lan, K_vec>>,
                                       tuple<sequence<1, 2>, sequence<2>>,
                                       tuple<sequence<1, 0>, sequence<1>>,
                                       sequence<1, 2>,
                                       sequence<0, 2>>{});

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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAGlobalTileDistribution()
    {
        constexpr index_t kMPerBlock = Problem::FusedMoeTileShape::kM_a;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment  = GetAlignmentA<Problem>();
        return MakeGlobalTileDistribution_SimpleMxK_Async<kMPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGGlobalTileDistribution()
    {
        constexpr index_t kNPerBlock = Problem::FusedMoeTileShape::kN_g;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment  = GetAlignmentG<Problem>();
        return MakeGlobalTileDistribution_SimpleMxK_Async<kNPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeUGlobalTileDistribution()
    {
        constexpr index_t kNPerBlock = Problem::FusedMoeTileShape::kN_u;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment  = GetAlignmentU<Problem>();
        return MakeGlobalTileDistribution_SimpleMxK_Async<kNPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDGlobalTileDistribution()
    {
        constexpr index_t kNPerBlock = Problem::FusedMoeTileShape::kN_d;
        constexpr index_t kKPerBlock = Problem::FusedMoeTileShape::kK_y;
        constexpr index_t NumWarps   = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment  = GetAlignmentD<Problem>();
        return MakeGlobalTileDistribution_SimpleMxK_Async<kNPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment>();
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

        // constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        constexpr index_t KVector =
            Alignment;                  // GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad = KPack; // for async-copy, this pad is between warps

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

    template <index_t MPerBlock,
              index_t KPerBlock,
              index_t NumWarps,
              index_t KPack,
              index_t Alignement,
              index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto
        MakeSmemStoreBlockDescriptor_SimpleMxK_Async(number<IBuf> = number<0>{})
    {
        constexpr index_t kBlockSize = ck_tile::get_warp_size() * NumWarps; // Problem::kBlockSize;
        constexpr index_t warpSize   = ck_tile::get_warp_size();

        // constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
        // constexpr index_t Alignement = GetAlignmentK<Problem>(); // this is for global load
        constexpr index_t kPad =
            KPack; // for async-copy, this pad is between warps. Optimize this for lds_read speed

        static_assert(warpSize * Alignement >= KPerBlock && warpSize * Alignement % KPerBlock == 0);
        constexpr index_t LanesPerK =
            KPerBlock / Alignement; // how many lane (within a wave) to load K
        constexpr index_t LaneGroups =
            warpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr index_t NumIssues = MPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == MPerBlock * KPerBlock / (BlockSize * Alignement));

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},   // n0
                       number<LaneGroups>{},  // n1
                       number<NumWarps>{},    // n2
                       number<LanesPerK>{},   // k0
                       number<Alignement>{}), // k1
            make_tuple(number<NumWarps*(warpSize * Alignement + kPad)>{},
                       number<KPerBlock>{},
                       number<warpSize * Alignement + kPad>{},
                       number<Alignement>{},
                       number<1>{}),
            number<IBuf * GetSingleSmemElementSpaceSize<Problem>()>{},
            number<Alignement>{},
            number<1>{});

        // TODO this layout is hard coded, and will be used in async copy buffer view load
        // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
        constexpr auto k_lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<NumIssues>{}),
                       make_pass_through_transform(number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           number<LaneGroups>{}, number<LanesPerK>{}, number<Alignement>{}))),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        return k_lds_block_desc_issues_warps_lanes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeASmemLoadTileDistribution()
    {
        constexpr index_t kMPerBlock  = Problem::FusedMoeTileShape::kM_a;
        constexpr index_t kKPerBlock  = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps    = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment   = GetAlignmentA<Problem>();
        constexpr index_t KPack       = GetSmemKPackA<Problem>();
        constexpr index_t NumPrefetch = Problem::Traits::NumPrefetchA;

        return MakeSmemLoadTileDescriptor_SimpleMxK_Async<kMPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment,
                                                          KPack,
                                                          NumPrefetch>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeASmemStoreTileDistribution()
    {
        constexpr index_t kMPerBlock  = Problem::FusedMoeTileShape::kM_a;
        constexpr index_t kKPerBlock  = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps    = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment   = GetAlignmentA<Problem>();
        constexpr index_t KPack       = GetSmemKPackA<Problem>();
        constexpr index_t NumPrefetch = Problem::Traits::NumPrefetchA;

        return MakeSmemStoreBlockDescriptor_SimpleMxK_Async<kMperBlock,
                                                            kKPerBlock,
                                                            kBlockSize,
                                                            NumWarps,
                                                            KPack,
                                                            Alignment>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGSmemLoadTileDistribution()
    {
        constexpr index_t kNPerBlock  = Problem::FusedMoeTileShape::kN_g;
        constexpr index_t kKPerBlock  = Problem::FusedMoeTileShape::kK_a;
        constexpr index_t NumWarps    = Problem::FusedMoeTileShape::NumWarps;
        constexpr index_t Alignment   = GetAlignmentG<Problem>();
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
        constexpr index_t Alignment   = GetAlignmentG<Problem>();
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
        constexpr index_t Alignment   = GetAlignmentU<Problem>();
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
        constexpr index_t Alignment   = GetAlignmentD<Problem>();
        constexpr index_t KPack       = GetSmemKPackD<Problem>();
        constexpr index_t NumPrefetch = Problem::Traits::NumPrefetchD;

        return MakeSmemLoadTileDescriptor_SimpleMxK_Async<kNPerBlock,
                                                          kKPerBlock,
                                                          NumWarps,
                                                          Alignment,
                                                          KPack,
                                                          NumPrefetch>();
    }

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
};
} // namespace ck_tile
