// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_fwd_pipeline_policy.hpp"
#include "hstu_attention_kernel_util.hpp"

namespace ck_tile {

template <typename Problem_,
          typename Traits_,
          typename Policy_ = HstuAttentionFwdPipelineQRKSVSPolicy>
// Tdm variant of the no-softmax forward pipeline, for gfx1250.
//
// Same structure as HstuAttentionNoSoftmaxFwdPipelineQRKSVSTrLoad; what differs is only how
// K and V reach Lds. Here the Tdm engine copies them global -> Lds directly, so there is no
// register staging and no store_tile, and the Lds layout must be the plain row-major one the
// engine writes rather than an xor-swizzled one.
//
// Q is not staged through Tdm: hstu loads Q once straight into registers and has no Q Lds
// descriptor at all.
struct HstuAttentionNoSoftmaxFwdPipelineQRKSVSTdm
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

    static constexpr index_t kM0        = HstuAttentionTileSetting::kM0;
    static constexpr index_t kN0        = HstuAttentionTileSetting::kN0;
    static constexpr index_t kN0Sub     = HstuAttentionTileSetting::kN0Sub;
    static constexpr index_t kN1        = HstuAttentionTileSetting::kN1;
    static constexpr index_t kK1        = HstuAttentionTileSetting::kK1;
    static constexpr index_t kQKHeaddim = HstuAttentionTileSetting::kQKHeaddim;

    static_assert(kQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static_assert(Problem::kUseSoftmax == false, "This pipeline only works with not-using softmax");

    static constexpr bool kIsJagged   = Problem::kIsJagged;
    static constexpr auto kHasBias    = Problem::kHasBias;
    static constexpr bool kHasDropout = Problem::kHasDropout;
    static constexpr bool kHasCausal  = Problem::kHasCausal;

    // gemm_1 still reads V with ds_read_tr, so the trload block-gemm stays selected.
    static constexpr bool kUseTrLoad = true;
    // ... but the Lds descriptors must be the un-swizzled, row-padded ones, and the K/V Dram
    // windows must carry MakeK/VDramTdmTileDistribution().
    // kUseTrLoad is shared with the trload pipeline, so the kernel needs a separate marker
    // to recognise this one and skip the head-dim padding of the K/V Dram views.
    static constexpr bool kUseTdm = true;

    static constexpr bool kPadSeqLenQ   = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQK = Traits::kPadHeadDimQK;
    static constexpr bool kPadHeadDimV  = Traits::kPadHeadDimV;

    // This pipeline implements one schedule only: a single n0 sub-loop with K and V each
    // double-buffered, so that the next kv-tile's descriptors are in flight for the whole of
    // the current iteration. Assert the preconditions rather than silently degrading.
    static_assert(kN0 == kN0Sub && kN0 == kK1,
                  "the Tdm pipeline needs kN0 == kN0Sub == kK1, i.e. a single n0/k1 sub-loop");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        Traits::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem, true /*kUseTrLoad*/>();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem, true /*kUseTrLoad */>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    // used by NRepetitions2DEpilogue
    static constexpr index_t kGemm1SingleRepN =
        Policy::template GetPVTBlockGemmSingleRepN<Problem>();

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

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSizePlain<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename HstuMask>
    CK_TILE_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*kQKHeaddim tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*kQKHeaddim tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               index_t seqlen_k_start,
               index_t seqlen_k_end,
               HstuMask& mask,
               float scale_s, // scaling value exerted on the immediate Q@K result
               float scale_p, // scaling value exerted on the SiLu result
               void* smem_ptr,
               DropoutType& dropout) const
    {
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

        static_assert(n0_loops == k1_loops, "n0_loops == k1_loops required by this pipeline");

        constexpr auto NumKLdsBuffers = Policy::template GetNumKLdsBuffers<Problem>();
        constexpr auto NumVLdsBuffers = Policy::template GetNumVLdsBuffers<Problem>();

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVTBlockGemm<Problem, true /*kUseTrLoad*/>();

        // SaccBlockTile size is [kM0, kN0Sub]
        // PcompBlockTile size is [kM0, kN0]
        using SaccBlockTileType        = decltype(gemm_0.template MakeCBlockTile<kM0, kN0Sub>());
        using CombineSaccBlockTileType = decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
        using PcompBlockTileType = decltype(cast_tile<CompDataType>(CombineSaccBlockTileType{}));

        SaccBlockTileType sacc_tile;
        PcompBlockTileType pcomp_tile;

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
        OaccBlockTileType o_acc;

        if(seqlen_k_end <= seqlen_k_start)
        {
            clear_tile(o_acc);

            return o_acc;
        };

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        auto q_tile = load_tile(q_dram_window);

        const auto q_origin = q_dram_window.get_window_origin();

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
                             {seqlen_k_start, 0},
                             Policy::template MakeKDramTdmTileDistribution<Problem>());

        // Pad configuration for the K/V global -> Lds copies. The same
        // (PadIntervalBytes, PadLengthBytes) pair that MakeK/VLdsBlockDescriptor() used to
        // build the reader-side descriptors is converted here into the encoded fields the
        // Tdm descriptor carries, so writer and reader cannot disagree about the row stride.
        constexpr auto k_pad_config = Policy::template GetTdmPadConfigK<Problem>();
        constexpr auto v_pad_config = Policy::template GetTdmPadConfigV<Problem>();

        TDMConfig tdm_config_k{};
        tdm_config_k.pad_enable              = true;
        tdm_config_k.pad_config.pad_interval = k_pad_config.at(number<0>{});
        tdm_config_k.pad_config.pad_amount   = k_pad_config.at(number<1>{});

        TDMConfig tdm_config_v{};
        tdm_config_v.pad_enable              = true;
        tdm_config_v.pad_config.pad_interval = v_pad_config.at(number<0>{});
        tdm_config_v.pad_config.pad_amount   = v_pad_config.at(number<1>{});

        __builtin_amdgcn_sched_barrier(0);

        // provide partition_index for LDS tile window so that warp_id is in vgpr
        array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

        // K tile in LDS
        QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
        auto k_lds             = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsPlainBlockDescriptor<Problem>());
        auto k_lds_monolithic_window = make_tile_window(
            k_lds, Policy::template MakeKLdsPlainBlockDescriptor<Problem>().get_lengths(), {0, 0});

        static_assert(
            Policy::template MakeKLdsPlainBlockDescriptor<Problem>().get_lengths()[number<0>{}] ==
                NumKLdsBuffers * kN0Sub,
            "Check failed!");
        static_assert(
            Policy::template MakeKLdsPlainBlockDescriptor<Problem>().get_lengths()[number<1>{}] ==
                kQKHeaddim,
            "Check failed!");

        using k_lds_window_type = decltype(get_slice_tile(
            k_lds_monolithic_window, sequence<0, 0>{}, sequence<kN0Sub, kQKHeaddim>{}));

        statically_indexed_array<k_lds_window_type, NumKLdsBuffers> k_lds_windows;

        static_for<0, NumKLdsBuffers, 1>{}([&](auto i_buf) {
            k_lds_windows[i_buf] = get_slice_tile(k_lds_monolithic_window,
                                                  sequence<i_buf * kN0Sub, 0>{},
                                                  sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
        });

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QKVDataType*>(reinterpret_cast<char*>(smem_ptr) +
                                           Policy::template GetSmemSizeKPlain<Problem>()),
            Policy::template MakeVLdsPlainBlockDescriptor<Problem>());
        auto v_lds_monolithic_window = make_tile_window(
            v_lds, Policy::template MakeVLdsPlainBlockDescriptor<Problem>().get_lengths(), {0, 0});

        static_assert(
            Policy::template MakeVLdsPlainBlockDescriptor<Problem>().get_lengths()[number<0>{}] ==
                NumVLdsBuffers * kK1,
            "Check failed!");
        static_assert(
            Policy::template MakeVLdsPlainBlockDescriptor<Problem>().get_lengths()[number<1>{}] ==
                kN1,
            "Check failed!");

        // v_lds is written by Tdm and read back with ds_read_tr
        using v_lds_trload_window_type = decltype(get_slice_tile(
            v_lds_monolithic_window, sequence<0, 0>{}, sequence<kK1, kN1>{}));

        statically_indexed_array<v_lds_trload_window_type, NumVLdsBuffers> v_lds_trload_windows;

        static_for<0, NumVLdsBuffers, 1>{}([&](auto i_buf) {
            v_lds_trload_windows[i_buf] = get_slice_tile(v_lds_monolithic_window,
                                                         sequence<i_buf * kK1, 0>{},
                                                         sequence<(i_buf + 1) * kK1, kN1>{});
        });

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0},
                             Policy::template MakeVDramTdmTileDistribution<Problem>());

        // reduction function for softmax
        const auto f_silu = [&](CompDataType& x) {
            const auto one = ck_tile::type_convert<CompDataType>(1.0f);

            if constexpr(std::is_same_v<CompDataType, float>)
            {
                x = x * __builtin_amdgcn_rcpf(one + __expf(-x));
            }
            else
            {
                x = x / (one + exp(-x));
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

        auto seqlen_k_curr = seqlen_k_start;

        // K and V are each double-buffered, and the kv-tile for the next iteration is issued
        // right after the barrier that retires the current one, so two Tdm descriptors are in
        // flight for the whole of every iteration. On a standalone Tdm micro-benchmark going
        // from one to two outstanding descriptors nearly doubled the transfer rate (+79%
        // cache-resident, +99% Dram-bound).
        static_assert(NumKLdsBuffers >= 2 && NumVLdsBuffers >= 2,
                      "the Tdm pipeline needs K and V to be at least double-buffered");

        auto k_lds_cur  = k_lds_windows[number<0>{}];
        auto k_lds_next = k_lds_windows[number<1>{}];
        auto v_lds_cur  = v_lds_trload_windows[number<0>{}];
        auto v_lds_next = v_lds_trload_windows[number<1>{}];

        // prologue: issue the first kv-tile
        load_tile_tdm(tdm_config_k, k_lds_cur, k_dram_window);
        move_tile_window(k_dram_window, {kN0Sub, 0});
        load_tile_tdm(tdm_config_v, v_lds_cur, v_dram_window);
        move_tile_window(v_dram_window, {kK1, 0});

        do
        {
            // STAGE 1, Gemm_0 ( S = Q@K )
            static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                // Retire the K[t]/V[t] copies issued one kv-tile ago. This is the only
                // barrier of the iteration: every reader of the spare half is a gemm of
                // iteration t-1, so it is also the WAR fence for the writes issued below.
                s_wait_tensorcnt_barrier<0>();

                // issue the next kv-tile into the spare half
                load_tile_tdm(tdm_config_k, k_lds_next, k_dram_window);
                move_tile_window(k_dram_window, {kN0Sub, 0});
                load_tile_tdm(tdm_config_v, v_lds_next, v_dram_window);
                move_tile_window(v_dram_window, {kK1, 0});

                __builtin_amdgcn_sched_barrier(0x00000001);

                gemm_0(sacc_tile, q_tile, k_lds_cur);

                auto tmp_tile = cast_tile<CompDataType>(sacc_tile);

                set_slice_tile(pcomp_tile,
                               tmp_tile,
                               sequence<0, i_n0 * kN0Sub>{},
                               sequence<kM0, (i_n0 + 1) * kN0Sub>{});
            });

            __builtin_amdgcn_sched_barrier(0x00000001);

            // STAGE 2, scale_s, add bias, mask, siLU
            if constexpr(kHasBias)
            {
                const auto bias_tile = load_tile(bias_dram_window);

                tile_elementwise_inout(
                    [&scale_s](auto& x, const auto& y) {
                        x = x * scale_s + type_convert<CompDataType>(y);
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

                        if(!mask.IsTokenPairInsideMask(row, col))
                        {
                            pcomp_tile(i_j_idx) = type_convert<CompDataType>(0.0f);
                        };
                    });
                });
            }

            tile_elementwise_inout(f_silu, pcomp_tile);

            detail::scale_tile_in_pack(pcomp_tile, scale_p);

            seqlen_k_curr += kN0;

            if constexpr(kHasDropout)
            {
                auto randval_lds_ptr = reinterpret_cast<char*>(smem_ptr) +
                                       Policy::template GetSmemSizeKPlain<Problem>() +
                                       Policy::template GetSmemSizeVPlain<Problem>();

                dropout.template Run<decltype(gemm_0), CompDataType, uint8_t>(
                    randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);
            }

            auto p = cast_tile<PDataType>(pcomp_tile);

            // STAGE 3, Gemm_1 ( O = P@V )
            // V[t] was retired by the same barrier as K[t] in stage 1, and V[t+1] is already
            // in flight, so there is nothing to store and nothing to wait for here.
            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                    v_lds_cur);
            });

            // the half just consumed becomes the Tdm target for the round after next
            auto k_lds_tmp = k_lds_cur;
            k_lds_cur      = k_lds_next;
            k_lds_next     = k_lds_tmp;

            auto v_lds_tmp = v_lds_cur;
            v_lds_cur      = v_lds_next;
            v_lds_next     = v_lds_tmp;
        } while(seqlen_k_curr < seqlen_k_end);

        return o_acc;
    }
};

} // namespace ck_tile
