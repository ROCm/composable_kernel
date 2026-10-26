// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_fwd_pipeline_policy.hpp"

namespace ck_tile {

template <typename Problem_,
          typename Traits_,
          typename Policy_ = HstuAttentionFwdPipelineQRKSVSPolicy>
// Tdm variant of the with-softmax forward pipeline, for gfx1250.
//
// Same structure as HstuAttentionWithSoftmaxFwdPipelineQRKSVSTrLoad; the main difference is how
// K and V reach Lds. Here the Tdm engine copies them global -> Lds directly, so there is no
// register staging and no store_tile for them, and the Lds layout must be the plain row-major
// one the engine writes rather than an xor-swizzled one.
//
// Q is loaded once into registers. Where Policy::IsQInLdsTdm() each wave then stores its rows
// of Q to Lds once (plain row-major, behind the K/V copies) and gemm_0 reads Q back from there.
//
// The NumPrefetchK/NumPrefetchV software pipelining of the trload version is replaced by a
// plain double buffer on each of K and V: the Tdm engine keeps the copies in flight by
// itself, so the loads no longer need to be interleaved into the gemm loops by hand.
struct HstuAttentionWithSoftmaxFwdPipelineQRKSVSTdmTrLoad
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

    static_assert(Problem::kUseSoftmax == true, "This pipeline only works with using softmax");

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

    // This pipeline implements one schedule only: a single n0/k1 sub-loop with K and V each
    // double-buffered, so that the next kv-tile is in flight for the whole of the current
    // iteration.
    static_assert(kN0 == kN0Sub && kN0 == kK1,
                  "the Tdm pipeline needs kN0 == kN0Sub == kK1, i.e. a single n0/k1 sub-loop");

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
        Traits::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem, true /*kUseTrLoad*/>();

    // since dim num_splits is the fastest dim and each workgroup accesses only one value in this
    // dim
    static constexpr index_t kAlignmentLSEacc = 1;

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem, true /*kUseTrLoad*/>();
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
            else if constexpr(Policy::template IsQInLdsTdm<Problem>())
            {
                // hdim128 Tdm M128 tile with Q in Lds: 3 workgroups (= 3 waves/SIMD) per CU,
                // VGPR <= 336
                return 3;
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
              typename LSEorLSEaccDramBlockWindow,
              typename HstuMask>
    CK_TILE_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,           // M0*kQKHeaddim tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,           // N0*kQKHeaddim tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,           // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,     // M0*N0 tile
               LSEorLSEaccDramBlockWindow& lse_or_lse_acc_dram_block_window, // M0 tile
               index_t seqlen_k_start,
               index_t seqlen_k_end,
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

        // The trload pipeline requires n0_loops >= 2 and k1_loops >= 2 because its
        // NumPrefetchK/NumPrefetchV scheme spreads the loads over the sub-loops. Tdm keeps
        // the copies in flight itself, so this one wants the opposite: exactly one sub-loop,
        // with K and V double-buffered across kv-tiles instead of within one.
        static_assert(n0_loops == 1 && k1_loops == 1,
                      "the Tdm pipeline needs a single n0/k1 sub-loop");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemmTdm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVTBlockGemmTdm<Problem>();

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

        if(seqlen_k_end <= seqlen_k_start)
        {
            clear_tile(o_acc);

            if constexpr(!is_null_tile_window_v<LSEorLSEaccDramBlockWindow>)
            {
                auto lse_or_lse_acc =
                    make_static_distributed_tensor<CompDataType>(m.get_tile_distribution());

                set_tile(lse_or_lse_acc, -numeric<CompDataType>::infinity());

                store_tile(lse_or_lse_acc_dram_block_window, lse_or_lse_acc);
            }

            return o_acc;
        };

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        // The Tdm Lds layout reserves no space for the dropout randval buffer (GetSmemSizeDropout()
        // is 0, so the buffer at smem + 2 K + 2 V would overlap Q in Lds or run past the end of
        // the allocation): dropout is refused whether or not Q is in Lds. kBlockPerCu workgroups
        // per CU (above, or from Traits) need kBlockPerCu copies of the whole Lds layout. The Tdm
        // pipelines only run on gfx1250, so its Lds size is named directly: get_lds_size() would
        // pick the arch from the device macros, which the host pass of a HIP build does not define.
        static_assert(!kHasDropout,
                      "the Tdm pipelines reserve no Lds for the dropout randval buffer");
        static_assert(kBlockPerCu * GetSmemSize() <= detail::get_lds_size(gfx125_t{}),
                      "kBlockPerCu x GetSmemSize() does not fit the Lds of a CU");

        // The Q operand of gemm_0: the register tile or, where Policy::IsQInLdsTdm(), an Lds
        // window over a copy of Q that every wave stores once here (each wave reads back only
        // the rows it wrote, so no barrier is needed).
        auto q_tile = [&]() {
            if constexpr(Policy::template IsQInLdsTdm<Problem>())
            {
                auto q_lds = make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<QKVDataType*>(
                        static_cast<char*>(smem_ptr) +
                        Policy::template GetQLdsByteOffsetPlain<Problem>()),
                    Policy::template MakeQLdsPlainBlockDescriptor<Problem>());

                auto q_lds_store_window =
                    make_tile_window(q_lds,
                                     make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                     {0, 0},
                                     Policy::template MakeQRegTileDistribution<Problem>());

                store_tile(q_lds_store_window, load_tile(q_dram_window));

                return make_tile_window(
                    q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});
            }
            else
                return load_tile(q_dram_window);
        }();

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

        // Lds layout: see Policy::GetSmemSizePlain()
        constexpr index_t k_smem_size = Policy::template GetSmemSizeKPlain<Problem>();
        constexpr index_t v_smem_size = Policy::template GetSmemSizeVPlain<Problem>();

        // K tile in LDS
        QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
        auto k_lds             = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsPlainBlockDescriptor<Problem>());
        auto k_lds_monolithic_window_0 = make_tile_window(
            k_lds, Policy::template MakeKLdsPlainBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto k_lds_monolithic_window_1 = k_lds_monolithic_window_0;
        k_lds_monolithic_window_1.set_bottom_tensor_view_data_ptr(
            k_lds_ptr + k_smem_size / sizeof(QKVDataType));

        static_assert(
            Policy::template MakeKLdsPlainBlockDescriptor<Problem>().get_lengths()[number<0>{}] ==
                kN0,
            "Check failed!");
        static_assert(
            Policy::template MakeKLdsPlainBlockDescriptor<Problem>().get_lengths()[number<1>{}] ==
                kQKHeaddim,
            "Check failed!");

        using k_lds_window_type = decltype(get_slice_tile(
            k_lds_monolithic_window_0, sequence<0, 0>{}, sequence<kN0Sub, kQKHeaddim>{}));

        statically_indexed_array<k_lds_window_type, n0_loops> k_lds_windows_0;
        statically_indexed_array<k_lds_window_type, n0_loops> k_lds_windows_1;

        static_for<0, n0_loops, 1>{}([&](auto i_n0) {
            k_lds_windows_0[i_n0] = get_slice_tile(k_lds_monolithic_window_0,
                                                   sequence<i_n0 * kN0Sub, 0>{},
                                                   sequence<(i_n0 + 1) * kN0Sub, kQKHeaddim>{});
            k_lds_windows_1[i_n0] = get_slice_tile(k_lds_monolithic_window_1,
                                                   sequence<i_n0 * kN0Sub, 0>{},
                                                   sequence<(i_n0 + 1) * kN0Sub, kQKHeaddim>{});
        });

        // V tile in LDS. Unlike K, one view spans both V copies: viewing one copy and moving
        // the pointer by v_smem_size left two extra VGPR spill reloads in the mask sweep of the
        // no_softmax M128 kernel on gfx1250 (~10% slower at seqlen 4k).
        QKVDataType* v_lds_ptr =
            reinterpret_cast<QKVDataType*>(reinterpret_cast<char*>(smem_ptr) + k_smem_size * 2);
        constexpr auto v_lds_desc = Policy::template MakeVLdsPlainBlockDescriptor<Problem, 2>();
        auto v_lds = make_tensor_view<address_space_enum::lds>(v_lds_ptr, v_lds_desc);
        auto v_lds_monolithic_window_0 = make_tile_window(v_lds, v_lds_desc.get_lengths(), {0, 0});

        auto v_lds_monolithic_window_1 = v_lds_monolithic_window_0;
        v_lds_monolithic_window_1.set_bottom_tensor_view_data_ptr(
            v_lds_ptr + Policy::template GetVLdsCopy1Offset<Problem>());

        static_assert(
            Policy::template MakeVLdsPlainBlockDescriptor<Problem>().get_lengths()[number<0>{}] ==
                kN0,
            "Check failed!");
        static_assert(
            Policy::template MakeVLdsPlainBlockDescriptor<Problem>().get_lengths()[number<1>{}] ==
                kN1,
            "Check failed!");

        // v_lds for normal write and trload read
        using v_lds_trload_window_type = decltype(get_slice_tile(
            v_lds_monolithic_window_0, sequence<0, 0>{}, sequence<kK1, kN1>{}));

        statically_indexed_array<v_lds_trload_window_type, k1_loops> v_lds_windows_0;
        statically_indexed_array<v_lds_trload_window_type, k1_loops> v_lds_windows_1;

        static_for<0, k1_loops, 1>{}([&](auto i_k1) {
            v_lds_windows_0[i_k1] = get_slice_tile(v_lds_monolithic_window_0,
                                                   sequence<i_k1 * kK1, 0>{},
                                                   sequence<(i_k1 + 1) * kK1, kN1>{});
            v_lds_windows_1[i_k1] = get_slice_tile(v_lds_monolithic_window_1,
                                                   sequence<i_k1 * kK1, 0>{},
                                                   sequence<(i_k1 + 1) * kK1, kN1>{});
        });

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0},
                             Policy::template MakeVDramTdmTileDistribution<Problem>());

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

        // true: rescale o_acc behind one wave-uniform branch; false: always multiply.
        // Measured on gfx1250: the branch wins on the M128 tiles (many kv-tiles per row, m
        // rarely moves after the first ones) and on the hdim256 M64 tile (256 o_acc columns
        // per row: multiplying on every kv-tile costs little time at full clock, but the long
        // calls run at the power limit and get a lower clock); always multiplying wins on the
        // other M64 tiles.
        constexpr bool kUniformRescale = (kM0 >= 128) || (kQKHeaddim == 256);
        const auto m_keep_threshold    = type_convert<CompDataType>(-8.0f);

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

        auto seqlen_k_curr = seqlen_k_start;

        // K and V are each double-buffered and the next kv-tile is issued right after the
        // barrier that retires the current one, so two Tdm descriptors stay in flight for the
        // whole of every iteration. This replaces the NumPrefetchK/NumPrefetchV register
        // prefetching of the trload pipeline, which existed to hide the same latency by hand.
        // prologue: issue the first kv-tile
        load_tile_tdm(tdm_config_k, k_lds_windows_0[number<0>{}], k_dram_window);
        move_tile_window(k_dram_window, {kN0Sub, 0});
        load_tile_tdm(tdm_config_v, v_lds_windows_0[number<0>{}], v_dram_window);
        move_tile_window(v_dram_window, {kK1, 0});

        // the gemms read the cur copies while Tdm fills the next ones
        auto k_lds_cur  = k_lds_windows_0;
        auto k_lds_next = k_lds_windows_1;
        auto v_lds_cur  = v_lds_windows_0;
        auto v_lds_next = v_lds_windows_1;

        do
        {
            // STAGE 1, Gemm_0 ( S = Q@K )
            static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                // Retire the K[t]/V[t] copies issued one kv-tile ago. This is the only
                // barrier of the iteration: every reader of the other copy is a gemm of
                // iteration t-1, so it also acts as the WAR fence for the writes below.
                s_wait_tensorcnt_barrier<0>();

                // issue the next kv-tile into the other copy, then run gemm_0 on this one
                load_tile_tdm(tdm_config_k, k_lds_next[i_n0], k_dram_window);
                move_tile_window(k_dram_window, {kN0Sub, 0});
                load_tile_tdm(tdm_config_v, v_lds_next[i_n0], v_dram_window);
                move_tile_window(v_dram_window, {kK1, 0});

                __builtin_amdgcn_sched_barrier(0x00000001);

                gemm_0(sacc_tile, q_tile, k_lds_cur[i_n0]);

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
                // contextual_seqlen is a tile-invariant scalar, so hoist its test out of the
                // per-element sweep: inside the sweep it makes the index clamp depend on a
                // runtime value and the whole clamp gets replicated once per element.
                auto sweep_mask = [&](auto has_contextual) {
                    sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                            const auto tile_idx = get_x_indices_from_distributed_indices(
                                pcomp_tile.get_tile_distribution(),
                                make_tuple(idx0, idx1),
                                partition_index);

                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);

                            if(!mask.IsTokenPairInsideMask(row, col, has_contextual) ||
                               col >= seqlen_k_end)
                            {
                                pcomp_tile(i_j_idx) = -numeric<CompDataType>::infinity();
                            };
                        });
                    });
                };

                if(mask.contextual_seqlen > 0)
                    sweep_mask(bool_constant<true>{});
                else
                    sweep_mask(bool_constant<false>{});
            }

            __builtin_amdgcn_sched_barrier(0x00000001);

            // V[t] was retired by the same barrier as K[t] in stage 1 and V[t+1] is already
            // in flight, so there is nothing to store and nothing to prefetch here.

            const auto m_old = m;

            block_tile_reduce(m, pcomp_tile, sequence<1>{}, f_max);
            block_tile_reduce_sync(m, f_max, bool_constant<false>{});

            __builtin_amdgcn_sched_barrier(0x00000001);

            // Branch-free row stage. A row that is fully masked so far has m == -inf (and
            // then m_old == -inf and every s of the row is -inf), so subtracting 0 instead of
            // m gives exp(-inf) == 0 for the whole row with no per-row exec branch. (m also
            // stays -inf while the unmasked s of the row are only NaN and -inf, as f_max drops
            // NaN; a NaN s then gives a NaN P here. If m is still -inf at the end, the final
            // normalization sets the row to 0; if a finite s follows in a later kv-tile, the
            // row ends up NaN, as in the CPU reference.)
            constexpr auto p_spans = decltype(pcomp_tile)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                // use the m_old[i] as the m-for-stablization if m_old[i] - m[i] >= -8.0f
                // and still keep the m-for-stablization in m[]; for m == -inf the
                // difference is NaN and m stays -inf
                const CompDataType m_new = m[i_idx];
                m(i_idx) = (m_old[i_idx] - m_new >= m_keep_threshold) ? m_old[i_idx] : m_new;

                const CompDataType m_sub =
                    (m[i_idx] == -numeric<CompDataType>::infinity()) ? CompDataType{0} : m[i_idx];

                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    pcomp_tile(i_j_idx)    = f_exp(pcomp_tile[i_j_idx] - m_sub);
                });
            });

            auto rowsum_p =
                block_tile_reduce<CompDataType>(pcomp_tile, sequence<1>{}, f_sum, CompDataType{0});

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

            // adjust o_acc[] according to the update between m and m_old. m > m_old implies
            // a finite m; o_scale is exp(m_old - m) there (0 for m_old == -inf) and 1 for
            // every other row, including the ones with m == -inf (see the final normalization).
            auto o_scale = make_static_distributed_tensor<CompDataType>(m.get_tile_distribution());
            bool any_rescale       = false;
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const bool rescale = m[i_idx] > m_old[i_idx];
                const auto tmp     = f_exp(m_old[i_idx] - m[i_idx]);
                o_scale(i_idx)     = rescale ? tmp : CompDataType{1};
                l(i_idx)           = o_scale[i_idx] * l[i_idx] + rowsum_p[i_idx];
                any_rescale        = any_rescale || rescale;
            });

            // after the first kv-tiles m rarely moves (m_keep_threshold), so test once per
            // wave instead of once per row and lane
            if constexpr(kUniformRescale)
            {
                if(__builtin_amdgcn_ballot_w32(any_rescale) != 0)
                {
                    sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);
                            o_acc(i_j_idx) *= o_scale[i_idx];
                        });
                    });
                }
            }
            else
            {
                sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= o_scale[i_idx];
                    });
                });
            }

            seqlen_k_curr += kN0;

            if constexpr(kHasDropout)
            {
                auto randval_lds_ptr =
                    reinterpret_cast<char*>(smem_ptr) + k_smem_size * 2 + v_smem_size * 2;

                dropout.template Run<decltype(gemm_0), CompDataType, uint8_t>(
                    randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);
            }

            auto p = cast_tile<PDataType>(pcomp_tile);

            __builtin_amdgcn_sched_barrier(0x00000001);

            // STAGE 3, Gemm_1 ( O = P@V )
            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                    v_lds_cur[i_k1]);
            });

            // the copy just consumed becomes the Tdm target for the kv-tile after next
            auto k_lds_tmp = k_lds_cur;
            k_lds_cur      = k_lds_next;
            k_lds_next     = k_lds_tmp;

            auto v_lds_tmp = v_lds_cur;
            v_lds_cur      = v_lds_next;
            v_lds_next     = v_lds_tmp;
        } while(seqlen_k_curr < seqlen_k_end);

        // if pipeline is called from splitkv_kernel, the window shall not be null;
        // if pipeline is called from non-splitkv kernel, the window is null if kStoreLSE is false
        if constexpr(!is_null_tile_window_v<LSEorLSEaccDramBlockWindow>)
        {
            // store lse or lse_acc
            auto lse_or_lse_acc =
                make_static_distributed_tensor<CompDataType>(m.get_tile_distribution());

            constexpr auto lse_or_lse_acc_spans = decltype(lse_or_lse_acc)::get_distributed_spans();
            sweep_tile_span(lse_or_lse_acc_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                // a row with m == -inf stores -inf whatever its l is (see the final normalization)
                lse_or_lse_acc(i_idx) = (m_[i_idx] == -numeric<CompDataType>::infinity())
                                            ? -numeric<CompDataType>::infinity()
                                            : m_[i_idx] + log(l_[i_idx]);
            });

            store_tile(lse_or_lse_acc_dram_block_window, lse_or_lse_acc);
        }

        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        // one division per row instead of one per element, and no per-element exec branch.
        // A row whose m is still -inf (fully masked, or no unmasked score other than NaN and
        // -inf) is set to 0 by a select, not by a multiply, so an inf/NaN in its o_acc (0 * an
        // inf/NaN of V, or the NaN P of NaN scores) does not leak into the output or the
        // split-kv combine.
        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx     = make_tuple(idx0);
            const bool dead          = (m[i_idx] == -numeric<CompDataType>::infinity());
            const CompDataType inv_l = 1.0f / l[i_idx];
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx)         = dead ? CompDataType{0} : o_acc[i_j_idx] * inv_l;
            });
        });

        return o_acc;
    }
};

} // namespace ck_tile
