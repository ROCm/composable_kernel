// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_bwd_kernel_2_pipeline_policy.hpp"

namespace ck_tile {

// Kernel 2 backward pipeline -- Softmax path (kUseSoftmax == true).
// Iterates over Q blocks for a fixed K/V tile; accumulates dK and dV.
// Naming: KR = K and V are register-resident, QS = Q and dO staged through LDS.
//
// Four static_for loops:
//   Loop 1 (gemm0_k0_loops): Gemm0 (S = Q@K); Q prefetched and written to q_lds and qt_lds.
//   Loop 2 (gemm2_k0_loops): Gemm2 (dP = dO@V); dO prefetched and written to do_lds and dot_lds.
//   Loop 3 (k1_loops): Gemm1 (dV += P^T @ dO^T); P^T is converted from casted PComp
//   Loop 4 (k1_loops): Gemm3 (dK += dS^T @ Q^T); dS^T is converted from casted dS
//
// LDS layout (fix regions):
//   q_lds   : [kM0, kK0] x NumQOGradLdsBuffers
//   do_lds  : [kM0, kK0] x NumQOGradLdsBuffers
//   qt_lds  : [kQKHeaddim, kK1] x k1_loops
//   dot_lds : [kVHeaddim,  kK1] x k1_loops
template <typename Problem_,
          typename Traits_,
          typename Policy_ = HstuAttentionBwdKernel2PipelinePolicy>
struct HstuAttentionWithSoftmaxBwdPipelineKRVRQS_dK_dV
{
    using Problem          = remove_cvref_t<Problem_>;
    using Traits           = remove_cvref_t<Traits_>;
    using Policy           = remove_cvref_t<Policy_>;
    using QKVDataType      = remove_cvref_t<typename Problem::QKVDataType>;
    using GemmAccDataType  = remove_cvref_t<typename Problem::GemmAccDataType>;
    using CompDataType     = remove_cvref_t<typename Problem::CompDataType>;
    using OGradDataType    = remove_cvref_t<typename Problem::OGradDataType>;
    using KGradAccDataType = remove_cvref_t<typename Problem::KGradAccDataType>;
    using VGradAccDataType = remove_cvref_t<typename Problem::VGradAccDataType>;
    using PDataType        = remove_cvref_t<typename Problem::PDataType>;

    using HstuAttentionTileSetting = remove_cvref_t<typename Problem::HstuAttentionTileSetting>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0        = HstuAttentionTileSetting::kM0;
    static constexpr index_t kK0        = HstuAttentionTileSetting::kK0;
    static constexpr index_t kN0        = HstuAttentionTileSetting::kN0;
    static constexpr index_t kK1        = HstuAttentionTileSetting::kK1;
    static constexpr index_t kQKHeaddim = HstuAttentionTileSetting::kQKHeaddim;
    static constexpr index_t kVHeaddim  = kQKHeaddim; // V shares head dim with K in HSTU

    static constexpr bool IsWarpGemm32 = HstuAttentionTileSetting::IsWarpGemm32;

    static_assert(Problem::kUseSoftmax == true, "This pipeline only works with the softmax path");

    static constexpr bool kIsJagged   = Problem::kIsJagged;
    static constexpr bool kHasBias    = Problem::kHasBias;
    static constexpr bool kHasCausal  = Problem::kHasCausal;
    static constexpr bool kHasDropout = Problem::kHasDropout;

    static constexpr bool kPadSeqLenQ   = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQK = Traits::kPadHeadDimQK;
    static constexpr bool kPadHeadDimV  = Traits::kPadHeadDimV;

    static constexpr index_t kAlignmentQ =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentKGrad =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentKGrad<Problem>();
    static constexpr index_t kAlignmentVGrad =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentVGrad<Problem>();

    static constexpr index_t kGemm3SingleRepN =
        Policy::template GetSGradTQTBlockGemmSingleRepN<Problem>();
    static constexpr index_t kGemm1SingleRepN =
        Policy::template GetPTOGradTBlockGemmSingleRepN<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Traits::kBlockPerCuForKernel2 != -1)
            return Traits::kBlockPerCuForKernel2;
        else
        {
            if constexpr(kQKHeaddim <= 128)
                return 2;
            else
                return 1;
        }
    }();

    // Kernel-2's Gemm0 uses an M-major warp layout (block-warps <1,4,1>), producing a P tile whose
    // warp C-vector lies on the M axis. The forward BlockDropout::Run assumes an N-major P tile, so
    // use the backward (M-major) BlockDropoutBwd::Run here instead. This must match the dropout
    // instance constructed in HstuAttentionBwdKernel2. Template args:
    // <IsDropout=true, IsWG32 (don't-care; derived internally), IsStoreRandval=false>.
    using DropoutType = std::
        conditional_t<kHasDropout, BlockDropoutBwd<true, IsWarpGemm32, false>, NullBlockDropout>;

    CK_TILE_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    // operator() accumulates dK and dV into the provided accumulator tiles.
    //
    // Parameters:
    //   q_dram_block_window_tmp    : Q  tile window [kM0, kQKHeaddim]
    //   do_dram_block_window_tmp   : dO tile window [kM0, kVHeaddim]
    //   lse_dram_block_window_tmp  : LSE tile window [kM0] (for P = exp(S - LSE))
    //   delta_dram_block_window_tmp: delta tile window [kM0] (D[sq], read-only)
    //   bias_dram_block_window_tmp : optional bias [kM0, kN0]
    //   k_dram_block_window_tmp    : K  DRAM window [kN0, kQKHeaddim] (loaded into registers here)
    //   v_dram_block_window_tmp    : V  DRAM window [kN0, kVHeaddim]  (loaded into registers here)
    //   seqlen_q_start / seqlen_q_end
    //   i_n0                       : column origin in K/V sequence dimension
    //   mask / scale_s / smem_ptr
    template <typename QDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DeltaDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename NullRandValDramWindowTmp,
              typename HstuMask>
    CK_TILE_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                   const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
                                   const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
                                   const DeltaDramBlockWindowTmp& delta_dram_block_window_tmp,
                                   const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                   const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                   const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                   const NullRandValDramWindowTmp& null_randval_window_tmp,
                                   index_t seqlen_q_start,
                                   index_t seqlen_q_end,
                                   index_t i_n0,
                                   HstuMask& mask,
                                   float scale_s,
                                   void* smem_ptr,
                                   DropoutType& dropout) const
    {
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();

        using Gemm0 = decltype(Policy::template GetQKBlockGemm<Problem>());

        // ---- Load K and V into registers (register-resident for the entire Q loop) ----
        auto k_tile = load_tile(k_dram_block_window_tmp);
        auto v_tile = load_tile(v_dram_block_window_tmp);

        // ---- Initialize dK and dV accumulators ----
        auto dk_acc = gemm_3.MakeCBlockTile();
        auto dv_acc = gemm_1.MakeCBlockTile();
        clear_tile(dk_acc);
        clear_tile(dv_acc);

        if(seqlen_q_start >= seqlen_q_end)
            return make_tuple(dk_acc, dv_acc);

        constexpr index_t gemm0_k0_loops = kQKHeaddim / kK0;
        constexpr index_t gemm2_k0_loops = kVHeaddim / kK0;
        constexpr index_t k1_loops       = kM0 / kK1;

        constexpr auto NumQOGradPrefetches = Policy::template GetNumQOGradPrefetches<Problem>();
        constexpr auto NumQOGradLdsBuffers = Policy::template GetNumQOGradLdsBuffers<Problem>();

        static_assert(NumQOGradPrefetches <= gemm0_k0_loops, "Check failed!");
        static_assert(NumQOGradLdsBuffers <= gemm0_k0_loops, "Check failed!");
        static_assert(NumQOGradPrefetches <= gemm2_k0_loops, "Check failed!");
        static_assert(NumQOGradLdsBuffers <= gemm2_k0_loops, "Check failed!");

        // ---- Tile type declarations ----
        using SaccBlockTileType     = decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
        using PGradaccBlockTileType = decltype(gemm_2.template MakeCBlockTile<kM0, kN0>());
        using PcompBlockTileType    = decltype(cast_tile<CompDataType>(SaccBlockTileType{}));

        SaccBlockTileType sacc_tile;
        PGradaccBlockTileType dpacc_tile;

        auto pt_tile = make_static_distributed_tensor<QKVDataType>(
            Policy::template MakePTRegTileDistribution<Problem>());
        auto dst_tile = make_static_distributed_tensor<QKVDataType>(
            Policy::template MakeSGradTRegTileDistribution<Problem>());

        // ---- LDS setup ----
        // Four LDS regions in order:
        //   [q_lds | do_lds | qt_lds | dot_lds]
        // q_lds  and do_lds  : double-buffered [kM0, kK0], invariant view for write/read
        // qt_lds and dot_lds : complete-buffered [kN0, kK1], transposed view [kM0, kK0] for write
        constexpr index_t q_smem_size  = Policy::template GetSmemSizeQ<Problem>();
        constexpr index_t do_smem_size = Policy::template GetSmemSizeOGrad<Problem>();
        constexpr index_t qt_smem_size = Policy::template GetSmemSizeQT<Problem>();

        // q_lds, the same tensor_view for write/read
        QKVDataType* q_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
        auto q_lds             = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_monolithic_window = make_tile_window(
            q_lds, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using q_lds_window_type = decltype(get_slice_tile(
            q_lds_monolithic_window, sequence<0, 0>{}, sequence<kM0, kK0>{}));
        statically_indexed_array<q_lds_window_type, NumQOGradLdsBuffers> q_lds_windows;
        static_for<0, NumQOGradLdsBuffers, 1>{}([&](auto i_buf) {
            q_lds_windows[i_buf] = get_slice_tile(q_lds_monolithic_window,
                                                  sequence<0, i_buf * kK0>{},
                                                  sequence<kM0, (i_buf + 1) * kK0>{});
        });

        // do_lds, the same tensor_view for write/read
        QKVDataType* do_lds_ptr =
            reinterpret_cast<QKVDataType*>(static_cast<char*>(smem_ptr) + q_smem_size);
        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());
        auto do_lds_monolithic_window = make_tile_window(
            do_lds, Policy::template MakeOGradLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using do_lds_window_type = decltype(get_slice_tile(
            do_lds_monolithic_window, sequence<0, 0>{}, sequence<kM0, kK0>{}));
        statically_indexed_array<do_lds_window_type, NumQOGradLdsBuffers> do_lds_windows;
        static_for<0, NumQOGradLdsBuffers, 1>{}([&](auto i_buf) {
            do_lds_windows[i_buf] = get_slice_tile(do_lds_monolithic_window,
                                                   sequence<0, i_buf * kK0>{},
                                                   sequence<kM0, (i_buf + 1) * kK0>{});
        });

        // --- dot_lds (separate transposed layout for dO^T, used by Gemm1) ---
        QKVDataType* dot_lds_ptr = reinterpret_cast<QKVDataType*>(
            static_cast<char*>(smem_ptr) + q_smem_size + do_smem_size + qt_smem_size);
        auto dot_write_lds = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsWriteBlockDescriptor<Problem>());
        auto dot_lds_write_monolithic_window = make_tile_window(
            dot_write_lds,
            Policy::template MakeOGradTLdsWriteBlockDescriptor<Problem>().get_lengths(),
            {0, 0});
        using dot_lds_write_window_type = decltype(get_slice_tile(
            dot_lds_write_monolithic_window, sequence<0, 0>{}, sequence<kM0, kK0>{}));
        statically_indexed_array<dot_lds_write_window_type, gemm2_k0_loops> dot_lds_write_windows;
        static_for<0, gemm2_k0_loops, 1>{}([&](auto i_buf) {
            dot_lds_write_windows[i_buf] = get_slice_tile(dot_lds_write_monolithic_window,
                                                          sequence<0, i_buf * kK0>{},
                                                          sequence<kM0, (i_buf + 1) * kK0>{});
        });

        auto dot_read_lds = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsReadBlockDescriptor<Problem>());
        auto dot_lds_read_monolithic_window = make_tile_window(
            dot_read_lds,
            Policy::template MakeOGradTLdsReadBlockDescriptor<Problem>().get_lengths(),
            {0, 0});
        using dot_lds_read_window_type = decltype(get_slice_tile(
            dot_lds_read_monolithic_window, sequence<0, 0>{}, sequence<kVHeaddim, kK1>{}));
        statically_indexed_array<dot_lds_read_window_type, k1_loops> dot_lds_read_windows;
        static_for<0, k1_loops, 1>{}([&](auto i_buf) {
            dot_lds_read_windows[i_buf] = get_slice_tile(dot_lds_read_monolithic_window,
                                                         sequence<0, i_buf * kK1>{},
                                                         sequence<kVHeaddim, (i_buf + 1) * kK1>{});
        });

        // --- qt_lds (separate transposed layout for Q^T, used by Gemm3) ---
        QKVDataType* qt_lds_ptr = reinterpret_cast<QKVDataType*>(static_cast<char*>(smem_ptr) +
                                                                 q_smem_size + do_smem_size);

        auto qt_write_lds = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeQTLdsWriteBlockDescriptor<Problem>());
        auto qt_lds_write_monolithic_window = make_tile_window(
            qt_write_lds,
            Policy::template MakeQTLdsWriteBlockDescriptor<Problem>().get_lengths(),
            {0, 0});
        using qt_lds_write_window_type = decltype(get_slice_tile(
            qt_lds_write_monolithic_window, sequence<0, 0>{}, sequence<kM0, kK0>{}));
        statically_indexed_array<qt_lds_write_window_type, gemm0_k0_loops> qt_lds_write_windows;
        static_for<0, gemm0_k0_loops, 1>{}([&](auto i_buf) {
            qt_lds_write_windows[i_buf] = get_slice_tile(qt_lds_write_monolithic_window,
                                                         sequence<0, i_buf * kK0>{},
                                                         sequence<kM0, (i_buf + 1) * kK0>{});
        });

        auto qt_read_lds = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeQTLdsReadBlockDescriptor<Problem>());
        auto qt_lds_read_monolithic_window =
            make_tile_window(qt_read_lds,
                             Policy::template MakeQTLdsReadBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});
        using qt_lds_read_window_type = decltype(get_slice_tile(
            qt_lds_read_monolithic_window, sequence<0, 0>{}, sequence<kQKHeaddim, kK1>{}));
        statically_indexed_array<qt_lds_read_window_type, k1_loops> qt_lds_read_windows;
        static_for<0, k1_loops, 1>{}([&](auto i_buf) {
            qt_lds_read_windows[i_buf] = get_slice_tile(qt_lds_read_monolithic_window,
                                                        sequence<0, i_buf * kK1>{},
                                                        sequence<kQKHeaddim, (i_buf + 1) * kK1>{});
        });

        array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

        // ---- Q DRAM window (per-iteration tile [kM0, kK0]) ----
        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             {seqlen_q_start, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());

        // ---- dO DRAM window ----
        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             {seqlen_q_start, 0},
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        // ---- LSE DRAM window [kM0] -- loaded once per outer iteration ----
        auto lse_dram_window =
            make_tile_window(lse_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}),
                             {seqlen_q_start},
                             Policy::template MakeLSETileDistribution<Problem>());

        // ---- Delta DRAM window [kM0] -- loaded once per outer iteration ----
        auto delta_dram_window =
            make_tile_window(delta_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}),
                             {seqlen_q_start},
                             Policy::template MakeDeltaTileDistribution<Problem>());

        // ---- Bias DRAM window ----
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             {seqlen_q_start, bias_origin.at(number<1>{})},
                             Policy::template MakeBiasDramTileDistribution<Problem>());

        auto null_randval_window =
            dropout.template MakeRandvalDramWindow<Gemm0>(null_randval_window_tmp, i_n0);

        const auto f_exp = [](CompDataType x) -> CompDataType {
            if constexpr(std::is_same_v<CompDataType, float>)
                return __expf(x);
            else
                return exp(x);
        };

        // ---- Prefetch first round of Q tiles (for Loop 1 / Gemm0) ----
        using q_tile_type  = decltype(load_tile(q_dram_window));
        using do_tile_type = decltype(load_tile(do_dram_window));
        statically_indexed_array<q_tile_type, NumQOGradPrefetches> q_tiles;
        statically_indexed_array<do_tile_type, NumQOGradPrefetches> do_tiles;
        static_for<0, NumQOGradPrefetches, 1>{}([&](auto i_k0) {
            q_tiles[i_k0] = load_tile(q_dram_window);
            move_tile_window(q_dram_window, {0, kK0});
        });

        __builtin_amdgcn_sched_barrier(0);

        // Full [kM0] row-scalar tiles for LSE and delta
        using LSETileType   = decltype(block_tile_reduce<CompDataType>(
            decltype(gemm_0.template MakeCBlockTile<kM0, kN0>()){},
            sequence<1>{},
            [](auto a, auto b) { return a + b; },
            CompDataType{0}));
        using DeltaTileType = LSETileType;
        LSETileType lse_tile;
        DeltaTileType delta_tile;

        auto seqlen_q_curr = seqlen_q_start;

        do
        {
            // Load LSE and delta for the full kM0 tile
            lse_tile   = cast_tile<CompDataType>(load_tile(lse_dram_window));
            delta_tile = cast_tile<CompDataType>(load_tile(delta_dram_window));
            move_tile_window(lse_dram_window, {kM0});
            move_tile_window(delta_dram_window, {kM0});

            // =======================================================================
            // Loop 1: Gemm0 (S = Q@K); prefetch Q to LDS
            // =======================================================================
            // Gemm0 accumulates across the gemm0_k0_loops k-slices (BlockGemm is C += A@B),
            // so sacc_tile must be cleared at the start of every sq iteration.
            clear_tile(sacc_tile);
            static_for<0, gemm0_k0_loops, 1>{}([&](auto i_k0) {
                constexpr auto i_prefetch_buf = number<i_k0 % NumQOGradPrefetches>{};
                constexpr auto i_lds_buf_0    = number<i_k0 % NumQOGradLdsBuffers>{};
                constexpr auto i_lds_buf_1    = i_k0;

                // Store Q to q_lds and also to qt_lds (transposed, for Gemm3 dK)
                store_tile(q_lds_windows[i_lds_buf_0], q_tiles[i_prefetch_buf], partition_index);
                store_tile(
                    qt_lds_write_windows[i_lds_buf_1], q_tiles[i_prefetch_buf], partition_index);

                __builtin_amdgcn_sched_barrier(0x00000001);

                // Prefetch next Q tile while current stores are in flight
                if constexpr(NumQOGradPrefetches + i_k0 < gemm0_k0_loops)
                {
                    q_tiles[i_prefetch_buf] = load_tile(q_dram_window);
                    move_tile_window(q_dram_window, {0, kK0});
                }

                __builtin_amdgcn_sched_barrier(0x00000001);

                block_sync_lds();

                auto k_slice = get_slice_tile(
                    k_tile, sequence<0, i_k0 * kK0>{}, sequence<kN0, (i_k0 + 1) * kK0>{});

                gemm_0(sacc_tile, q_lds_windows[i_lds_buf_0], k_slice);
            });

            move_tile_window(q_dram_window, {kM0, -gemm0_k0_loops * kK0});

            // Prefetch first round of dO tiles (for Loop 2 / Gemm2)
            static_for<0, NumQOGradPrefetches, 1>{}([&](auto i_k0) {
                do_tiles[i_k0] = load_tile(do_dram_window);
                move_tile_window(do_dram_window, {0, kK0});
            });

            // =======================================================================
            // Loop 2: Gemm2 (dP = dO@V); prefetch dO to LDS
            // =======================================================================
            // Gemm2 accumulates across the gemm2_k0_loops k-slices (BlockGemm is C += A@B),
            // so dpacc_tile must be cleared at the start of every sq iteration.
            clear_tile(dpacc_tile);
            static_for<0, gemm2_k0_loops, 1>{}([&](auto i_k0) {
                constexpr auto i_prefetch_buf = number<i_k0 % NumQOGradPrefetches>{};
                constexpr auto i_lds_buf_0    = number<i_k0 % NumQOGradLdsBuffers>{};
                constexpr auto i_lds_buf_1    = i_k0;

                // Store dO to do_lds and also to dot_lds (transposed, for Gemm1 dV)
                store_tile(do_lds_windows[i_lds_buf_0], do_tiles[i_prefetch_buf], partition_index);
                store_tile(
                    dot_lds_write_windows[i_lds_buf_1], do_tiles[i_prefetch_buf], partition_index);

                __builtin_amdgcn_sched_barrier(0x00000001);

                // Prefetch next dO tile while current stores are in flight
                if constexpr(NumQOGradPrefetches + i_k0 < gemm2_k0_loops)
                {
                    do_tiles[i_prefetch_buf] = load_tile(do_dram_window);
                    move_tile_window(do_dram_window, {0, kK0});
                }

                __builtin_amdgcn_sched_barrier(0x00000001);

                block_sync_lds();

                auto v_slice = get_slice_tile(
                    v_tile, sequence<0, i_k0 * kK0>{}, sequence<kN0, (i_k0 + 1) * kK0>{});

                gemm_2(dpacc_tile, do_lds_windows[i_lds_buf_0], v_slice);
            });

            move_tile_window(do_dram_window, {kM0, -gemm2_k0_loops * kK0});

            auto pcomp_tile  = cast_tile<CompDataType>(sacc_tile);
            auto dscomp_tile = cast_tile<CompDataType>(dpacc_tile);

            __builtin_amdgcn_sched_barrier(0x00000001);

            // ---- Scale, optional bias, mask ----
            if constexpr(kHasBias)
            {
                const auto bias_tile = load_tile(bias_dram_window);
                tile_elementwise_inout(
                    [&scale_s](auto& x, const auto& y) {
                        x = x * scale_s + type_convert<CompDataType>(y);
                    },
                    pcomp_tile,
                    bias_tile);
                move_tile_window(bias_dram_window, {kM0, 0});
            }
            else
            {
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, pcomp_tile);
            }

            // Softmax path mask: set masked-out S to -inf.
            // The `row >= seqlen_q_end` guard excludes padded tail Q rows of the last tile
            // (seqlen_q not a multiple of kM0). Kernel 2 reduces over Q (rows), so a padded row
            // that slips through the mask (IsTokenPairInsideMask clamps and may return true for
            // it) would pollute every dK/dV column; forcing S=-inf makes P=0 (hence dS=0) there.
            if(!mask.IsFullTileInsideMask(seqlen_q_curr, i_n0, number<kN0>{}, number<kM0>{}))
            {
                constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            pcomp_tile.get_tile_distribution(),
                            make_tuple(idx0, idx1),
                            partition_index);
                        const auto row    = seqlen_q_curr + tile_idx.at(number<0>{});
                        const auto col    = i_n0 + tile_idx.at(number<1>{});
                        constexpr auto ij = make_tuple(idx0, idx1);
                        if(!mask.IsTokenPairInsideMask(row, col) || row >= seqlen_q_end)
                            pcomp_tile(ij) = -numeric<CompDataType>::infinity();
                    });
                });
            }
            else
            {
                // Fully inside the mask, but the last tile may still contain padded tail
                // rows (row >= seqlen_q_end) that must be excluded from the dK/dV reduction.
                constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            pcomp_tile.get_tile_distribution(),
                            make_tuple(idx0, idx1),
                            partition_index);
                        const auto row    = seqlen_q_curr + tile_idx.at(number<0>{});
                        constexpr auto ij = make_tuple(idx0, idx1);
                        if(row >= seqlen_q_end)
                            pcomp_tile(ij) = -numeric<CompDataType>::infinity();
                    });
                });
            }

            // ---- P = exp(S - LSE[sq]) ----
            {
                constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx       = make_tuple(idx0);
                    const CompDataType lse_val = lse_tile[i_idx];
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto ij = make_tuple(idx0, idx1);
                        pcomp_tile(ij)    = f_exp(pcomp_tile[ij] - lse_val);
                    });
                });
            }

            // Build a per-element keep/drop mask WITHOUT disturbing P. Feeding a +1 sentinel tile
            // to BlockDropoutBwd::Run yields +1 (kept) / -1 (dropped). The mask is applied below as
            // drop_scale = rp_undrop (kept) / 0 (dropped) to P (-> dV) and to dP inside the softmax
            // dS jacobian (-> dK), matching reference_hstu_attention_bwd.hpp:
            //   dS = P * (drop_scale*dP - D),  P_drop = drop_scale * P   (P is the PURE softmax).
            // int8_t sentinel (1 byte/elem) keeps drop_mask cheap in VGPRs; Run only flips its
            // sign (+1 kept / -1 dropped), so 8 bits are enough.
            auto drop_mask =
                make_static_distributed_tensor<int8_t>(pcomp_tile.get_tile_distribution());
            if constexpr(kHasDropout)
            {
                __builtin_amdgcn_sched_barrier(0);

                tile_elementwise_inout([](auto& x) { x = type_convert<int8_t>(1); }, drop_mask);

                // BlockDropoutBwd::Run is M-major and needs no LDS scratch (no transpose).
                // Signature: Run<BlockGemm, RandValOutputDataType>(start_m0_idx, start_n0_idx,
                //                                                   p_compute,
                //                                                   randval_dram_window).
                dropout.template Run<Gemm0, uint8_t>(
                    seqlen_q_curr, i_n0, drop_mask, null_randval_window);

                move_tile_window(null_randval_window, {kM0, 0});

                __builtin_amdgcn_sched_barrier(0);
            }

            // ---- dS = P * (drop_scale*dP - D[sq]);  P_drop = drop_scale * P (for dV) ----
            // The dropout mask propagates onto dP (and onto P for dV); the softmax jacobian's outer
            // factor stays the PURE P. drop_scale = rp_undrop (kept) / 0 (dropped). Note the D term
            // is NOT scaled by drop_scale -- dropped positions still get dS = -P*D through the
            // softmax coupling (see reference derivation).
            {
                constexpr auto ds_spans = PcompBlockTileType::get_distributed_spans();
                sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx         = make_tuple(idx0);
                    const CompDataType delta_val = delta_tile[i_idx];
                    sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto ij         = make_tuple(idx0, idx1);
                        const CompDataType p_pure = pcomp_tile[ij];
                        CompDataType dp           = dscomp_tile[ij];
                        CompDataType p_drop       = p_pure;
                        if constexpr(kHasDropout)
                        {
                            if(drop_mask[ij] > 0)
                            {
                                dp     = dp * dropout.rp_undrop;
                                p_drop = p_pure * dropout.rp_undrop;
                            }
                            else
                            {
                                dp     = type_convert<CompDataType>(0.0f);
                                p_drop = type_convert<CompDataType>(0.0f);
                            }
                        }
                        // dK path: dS = P_pure * (drop_scale*dP - D)
                        dscomp_tile(ij) = p_pure * (dp - delta_val);
                        // dV path: P_drop = drop_scale * P
                        pcomp_tile(ij) = p_drop;
                    });
                });
            }

            auto p_gemm_tile = cast_tile<QKVDataType>(pcomp_tile);

            Policy::template PTFromGemm0CToGemm1A<Problem,
                                                  decltype(pt_tile),
                                                  decltype(p_gemm_tile)>(pt_tile, p_gemm_tile);

            block_sync_lds();

            // =======================================================================
            // Loop 3: Gemm1  dV += P^T @ dO^T
            // =======================================================================
            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                // Gemm1: dV += P^T_sub @ dO^T_sub
                auto pt_slice = get_slice_tile(
                    pt_tile, sequence<0, i_k1 * kK1>{}, sequence<kN0, (i_k1 + 1) * kK1>{});
                gemm_1(dv_acc, pt_slice, dot_lds_read_windows[i_k1]);
            });

            // Prefetch first round of Q tiles for the *next* outer do-while iteration
            static_for<0, NumQOGradPrefetches, 1>{}([&](auto i_buf) {
                q_tiles[i_buf] = load_tile(q_dram_window);
                move_tile_window(q_dram_window, {0, kK0});
            });

            auto ds_gemm_tile = cast_tile<QKVDataType>(dscomp_tile);

            Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                      decltype(dst_tile),
                                                      decltype(ds_gemm_tile)>(dst_tile,
                                                                              ds_gemm_tile);

            // =======================================================================
            // Loop 4: Gemm3  dK += dS^T @ Q^T
            // =======================================================================
            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                // Gemm3: dK += dS^T_sub @ Q^T_sub
                auto dst_slice = get_slice_tile(
                    dst_tile, sequence<0, i_k1 * kK1>{}, sequence<kN0, (i_k1 + 1) * kK1>{});
                gemm_3(dk_acc, dst_slice, qt_lds_read_windows[i_k1]);
            });

            seqlen_q_curr += kM0;

            // ensure LDS access finished in gemm_1 and gemm_3 before it being stored at next
            // iteration
            block_sync_lds();
        } while(seqlen_q_curr < seqlen_q_end);

        // Apply alpha scaling to accumulated dK
        tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, dk_acc);

        return make_tuple(dk_acc, dv_acc);
    }
};

} // namespace ck_tile
