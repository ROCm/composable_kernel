// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"

#include "hstu_attention_fwd_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem_,
          typename Traits_,
          typename Policy_ = HstuAttentionFwdPipelineQRKSVSDefaultPolicy>
struct HstuAttentionWithSoftmaxFwdPipelineQRKSVS
{
    using Problem         = remove_cvref_t<Problem_>;
    using Traits          = remove_cvref_t<Traits_>;
    using Policy          = remove_cvref_t<Policy_>;
    using QKVDataType     = remove_cvref_t<typename Problem::InOutDataType>;
    using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
    using CompDataType    = remove_cvref_t<typename Problem::CompDataType>;
    using BiasDataType    = remove_cvref_t<typename Problem::BiasDataType>;
    using PDataType       = remove_cvref_t<typename Problem::InOutDataType>;
    using ODataType       = remove_cvref_t<typename Problem::InOutDataType>;

    using HstuAttentionTileSetting = remove_cvref_t<typename Problem::HstuAttentionTileSetting>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = HstuAttentionTileSetting::kM0;
    static constexpr index_t kN0           = HstuAttentionTileSetting::kN0;
    static constexpr index_t kN0Sub        = HstuAttentionTileSetting::kN0Sub;
    static constexpr index_t kN1           = HstuAttentionTileSetting::kN1;
    static constexpr index_t kK1           = HstuAttentionTileSetting::kK1;
    static constexpr index_t kQKHeaddim    = HstuAttentionTileSetting::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = HstuAttentionTileSetting::kSubQKHeaddim;

    static_assert(kQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static_assert(Problem::kUseSoftmax == true, "This pipeline only works with using softmax");

    static constexpr bool kIsJagged   = Problem::kIsJagged;
    static constexpr auto kHasBias    = Problem::kHasBias;
    static constexpr bool kHasDropout = Problem::kHasDropout;
    static constexpr bool kHasCausal  = Problem::kHasCausal;

    static_assert(Problem::kUseTrLoad == false, "Check failed!");

    static constexpr bool kUseTrLoad = false;

    static constexpr bool kPadSeqLenQ   = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQK = Traits::kPadHeadDimQK;
    static constexpr bool kPadHeadDimV  = Traits::kPadHeadDimV;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        Traits::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    // used by NRepetitions2DEpilogue
    static constexpr index_t kGemm1SingleRepN =
        Policy::template GetKVBlockGemmSingleRepN<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Traits::kBlockPerCu != -1)
            return Traits::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim == 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim == 64)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim == 96 || kQKHeaddim == 128)
            {
                if constexpr(kHasBias)
                    return 2;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim == 256)
            {
                return 1;
            }
            else
            {
                return 1;
            };
        }
    }();

    static constexpr const char* name = "qr_hstu";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename QElementFunction,
              typename BiasElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename HstuMask>
    CK_TILE_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*kQKHeaddim tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*kQKHeaddim tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               HstuMask& mask,
               float scale_s, // scaling value exerted on the immediate Q@K result
               float scale_p, // scaling value exerted on the SiLu result
               void* smem_ptr,
               DropoutType& dropout) const
    {
        ignore = scale_p;

        static_assert(
            std::is_same_v<QKVDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QKVDataType,
                               remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QKVDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kQKHeaddim == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr index_t n0_loops = kN0 / kN0Sub;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(n0_loops >= 2, "n0_loops >= 2 required by this pipeline");
        static_assert(k1_loops >= 2, "k1_loops >= 2 required by this pipeline");

        constexpr auto NumKVLdsBuffers = Policy::template GetNumKVLdsBuffers<Problem>();

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        // SaccBlockTile size is [kM0, kN0Sub]
        // PcompBlockTile size is [kM0, kN0]
        using SaccBlockTileType        = decltype(gemm_0.template MakeCBlockTile<kM0, kN0Sub>());
        using CombineSaccBlockTileType = decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
        using PcompBlockTileType = decltype(cast_tile<CompDataType>(CombineSaccBlockTileType{}));

        SaccBlockTileType sacc_tile;
        PcompBlockTileType pcomp_tile;

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using MLBlockTileType = decltype(block_tile_reduce<CompDataType>(
            PcompBlockTileType{}, sequence<1>{}, f_max, CompDataType{0}));

        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
        OaccBlockTileType o_acc;

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        auto q_tile = load_tile(q_dram_window);

        const auto q_origin = q_dram_window.get_window_origin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
                             {seqlen_k_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem>());

        using k_tile_type = decltype(load_tile(k_dram_window));

        constexpr index_t NumPrefetchK = 2;

        static_assert(n0_loops >= NumPrefetchK, "Check failed!");

        // only prefetch two k tiles to save vgprs consumption
        statically_indexed_array<k_tile_type, NumPrefetchK> k_tiles;

        static_for<0, NumPrefetchK, 1>{}([&](auto i_n0) {
            k_tiles[i_n0] = load_tile(k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
        });

        __builtin_amdgcn_sched_barrier(0);

        // provide partition_index for LDS tile window so that warp_id is in vgpr
        array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

        // K tile in LDS
        QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
        auto k_lds             = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window = make_tile_window(
            k_lds, Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using k_lds_window_type = decltype(get_slice_tile(
            k_lds_window, sequence<0, 0>{}, sequence<kN0Sub, kQKHeaddim>{}));

        statically_indexed_array<k_lds_window_type, NumKVLdsBuffers> k_lds_windows;

        static_for<0, NumKVLdsBuffers, 1>{}([&](auto i_buf) {
            k_lds_windows[i_buf] = get_slice_tile(k_lds_window,
                                                  sequence<i_buf * kN0Sub, 0>{},
                                                  sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
        });

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QKVDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using v_lds_window_type =
            decltype(get_slice_tile(v_lds_window, sequence<0, 0>{}, sequence<kN1, kK1>{}));

        statically_indexed_array<v_lds_window_type, NumKVLdsBuffers> v_lds_windows;

        static_for<0, NumKVLdsBuffers, 1>{}([&](auto i_buf) {
            v_lds_windows[i_buf] = get_slice_tile(
                v_lds_window, sequence<i_buf * kN1, 0>{}, sequence<(i_buf + 1) * kN1, kK1>{});
        });

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start},
                             Policy::template MakeVDramTileDistribution<Problem>());

        const auto f_exp = [&](CompDataType x) {
            if constexpr(std::is_same_v<CompDataType, float>)
            {
                return __expf(x);
            }
            else
            {
                return exp(x);
            }
        };

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK1>{}),
                             {bias_origin.at(number<0>{}), seqlen_k_start}, // M/N
                             Policy::template MakeBiasDramTileDistribution<Problem>());

        auto null_randval_window = [&]() {
            if constexpr(kHasDropout)
            {
                const auto null_randval_dram = [&]() {
                    const auto null_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        static_cast<uint8_t*>(nullptr),
                        make_tuple(1, 1),
                        make_tuple(1, 1),
                        number<1>{},
                        number<1>{});

                    return pad_tensor_view(null_dram_naive,
                                           make_tuple(number<1>{}, number<1>{}),
                                           sequence<true, true>{});
                }();

                return make_tile_window(
                    null_randval_dram, make_tuple(number<1>{}, number<1>{}), {0, 0});
            }
            else
                return make_null_tile_window(make_tuple(number<1>{}, number<1>{}));
        }();

        clear_tile(o_acc);
        set_tile(m, -numeric<CompDataType>::infinity());
        clear_tile(l);

        q_tile = tile_elementwise_in(q_element_func, q_tile);

        auto seqlen_k_curr = seqlen_k_start;

        constexpr index_t NumPrefetchV = 2;

        static_assert(NumPrefetchV >= NumPrefetchK);

        using v_tile_type = decltype(load_tile(v_dram_window));

        statically_indexed_array<v_tile_type, NumPrefetchV> v_tiles;

        do
        {
            // STAGE 1, Gemm_0 ( S = Q@K )
            static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                store_tile(k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}],
                           k_tiles[number<i_n0 % NumPrefetchK>{}],
                           partition_index);

                __builtin_amdgcn_sched_barrier(0x00000001);

                if constexpr(i_n0 < n0_loops - NumPrefetchK)
                {
                    k_tiles[number<i_n0 % NumPrefetchK>{}] = load_tile(k_dram_window);
                    move_tile_window(k_dram_window, {kN0Sub, 0});
                }
                else
                {
                    // Since NumPrefetchV >= NumPrefetchK, we are able to have NumPrefetchK
                    // prefetchings of v_tile arranged in n0_loops

                    v_tiles[number<i_n0 - (n0_loops - NumPrefetchK)>{}] = load_tile(v_dram_window);
                    move_tile_window(v_dram_window, {0, kK1});
                };

                __builtin_amdgcn_sched_barrier(0x00000001);

                block_sync_lds();

                // execute current unroll of gemm_0
                gemm_0(sacc_tile, q_tile, k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}]);

                sacc_tile = tile_elementwise_in(s_acc_element_func, sacc_tile);

                auto tmp_tile = cast_tile<CompDataType>(sacc_tile);

                set_slice_tile(pcomp_tile,
                               tmp_tile,
                               sequence<0, i_n0 * kN0Sub>{},
                               sequence<kM0, (i_n0 + 1) * kN0Sub>{});
            });

            // STAGE 2, scale_s, add bias, mask, siLU
            if constexpr(kHasBias)
            {
                const auto bias_tile = load_tile(bias_dram_window);

                tile_elementwise_inout(
                    [&scale_s, &bias_element_func](auto& x, const auto& y) {
                        x = x * scale_s + type_convert<CompDataType>(bias_element_func(y));
                    },
                    pcomp_tile,
                    bias_tile);

                move_tile_window(bias_dram_window, {0, kN0});
            }
            else
            {
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, pcomp_tile);
            }

            if(!mask.IsFullTileInsideMask(
                   q_origin.at(number<0>{}), seqlen_k_curr, number<kN0>{}, number<kM0>{}))
            {
                constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            pcomp_tile.get_tile_distribution(),
                            make_tuple(idx0, idx1),
                            partition_index);

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        if(!mask.IsTokenPairInsideMask(row, col) || col >= seqlen_k_end)
                        {
                            pcomp_tile(i_j_idx) = -numeric<CompDataType>::infinity();
                        };
                    });
                });
            }
            else
            {
                constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            pcomp_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto col         = seqlen_k_curr + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        if(col >= seqlen_k_end)
                        {
                            pcomp_tile(i_j_idx) = -numeric<CompDataType>::infinity();
                        };
                    });
                });
            };

            __builtin_amdgcn_sched_barrier(0x00000001);

            using v_shuffled_tile_type = decltype(make_static_distributed_tensor<QKVDataType>(
                Policy::template MakeShuffledVRegTileDistribution<Problem>()));

            v_shuffled_tile_type v_shuffled_tile;

            shuffle_tile(v_shuffled_tile, v_tiles[number<0>{}]);

            // check whether first V-LdsBufer overlap with last K-LdsBuffer,
            // this does not occur when n0_loops == 2/4 and NumKVLdsBuffers == 4
            if constexpr((n0_loops - 1) % NumKVLdsBuffers == 2 % NumKVLdsBuffers)
            {
                __builtin_amdgcn_s_barrier();
            };

            store_tile(
                v_lds_windows[number<2 % NumKVLdsBuffers>{}], v_shuffled_tile, partition_index);

            __builtin_amdgcn_sched_barrier(0x00000001);

            static_for<NumPrefetchK, NumPrefetchV, 1>{}([&](auto i_k1) {
                v_tiles[i_k1] = load_tile(v_dram_window);
                move_tile_window(v_dram_window, {0, kK1});
            });

            __builtin_amdgcn_sched_barrier(0x00000001);

            auto m_local = block_tile_reduce<CompDataType>(
                pcomp_tile, sequence<1>{}, f_max, -numeric<CompDataType>::infinity());
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m;

            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);

            constexpr auto p_spans = decltype(pcomp_tile)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                if(m[i_idx] == -numeric<CompDataType>::infinity())
                {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        pcomp_tile(i_j_idx)    = type_convert<CompDataType>(0.0f);
                    });
                }
                else
                {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        pcomp_tile(i_j_idx)    = f_exp(pcomp_tile[i_j_idx] - m[i_idx]);
                    });
                }
            });

            auto rowsum_p =
                block_tile_reduce<CompDataType>(pcomp_tile, sequence<1>{}, f_sum, CompDataType{0});

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

            // adjust o_acc[] according to the update between m and m_old
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                if(m[i_idx] == -numeric<CompDataType>::infinity())
                {
                    l(i_idx) = rowsum_p[i_idx];
                }
                else
                {
                    const auto tmp = f_exp(m_old[i_idx] - m[i_idx]);
                    l(i_idx)       = tmp * l[i_idx] + rowsum_p[i_idx];
                    sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= tmp;
                    });
                }
            });

            seqlen_k_curr += kN0;

            if constexpr(kHasDropout)
            {
                auto randval_lds_ptr =
                    reinterpret_cast<char*>(smem_ptr) + Policy::template GetSmemSizeKV<Problem>();

                dropout.template Run<decltype(gemm_0), CompDataType, uint8_t>(
                    randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);
            }

            auto p = cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, pcomp_tile));

            __builtin_amdgcn_sched_barrier(0x00000001);

            // STAGE 3, Gemm_1 ( O = P@V )
            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                if constexpr(i_k1 < k1_loops - NumPrefetchV)
                {
                    v_tiles[number<i_k1 % NumPrefetchV>{}] = load_tile(v_dram_window);
                    move_tile_window(v_dram_window, {0, kK1});
                };

                if constexpr((i_k1 >= k1_loops - NumPrefetchV) &&
                             (i_k1 - (k1_loops - NumPrefetchV) < NumPrefetchK))
                {
                    k_tiles[number<i_k1 - (k1_loops - NumPrefetchV)>{}] = load_tile(k_dram_window);
                    move_tile_window(k_dram_window, {kN0Sub, 0});
                };

                __builtin_amdgcn_sched_barrier(0x00000001);

                block_sync_lds();

                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                    v_lds_windows[number<(i_k1 + 2) % NumKVLdsBuffers>{}]);

                if constexpr(i_k1 < k1_loops - 1)
                {
                    __builtin_amdgcn_sched_barrier(0x00000001);

                    shuffle_tile(v_shuffled_tile, v_tiles[number<(i_k1 + 1) % NumPrefetchV>{}]);
                    store_tile(v_lds_windows[number<(i_k1 + 3) % NumKVLdsBuffers>{}],
                               v_shuffled_tile,
                               partition_index);

                    __builtin_amdgcn_sched_barrier(0x00000001);
                };
            });

            // check whether last V-LdsBuffer overlap with first K-LdsBuffer,
            // this does not occur when k1_loops == 2 and NumKVLdsBuffers == 4
            if constexpr((k1_loops - 1 + 2) % NumKVLdsBuffers == 0)
            {
                __builtin_amdgcn_s_barrier();
            };
        } while(seqlen_k_curr < seqlen_k_end);

        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                if(m[i_idx] == -numeric<CompDataType>::infinity())
                    o_acc(i_j_idx) = 0.0f;
                else
                    o_acc(i_j_idx) *= 1.0f / l[i_idx];
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename HstuMask>
    CK_TILE_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*kQKHeaddim tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*KSubQKHeaddim tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               HstuMask mask,
               float scale_s, // scaling value exerted on the immediate Q@K result
               float scale_p, // scaling value exerted on the SiLU result
               void* smem_ptr,
               DropoutType& dropout) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          v_dram_block_window_tmp,
                          bias_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          scale_s,
                          scale_p,
                          smem_ptr,
                          dropout);
    }
};

} // namespace ck_tile
