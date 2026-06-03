// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdConvertQGrad
{
    using AccDataType   = remove_cvref_t<typename Problem::AccDataType>;
    using QGradDataType = remove_cvref_t<typename Problem::QGradDataType>;

    static constexpr index_t kM0 = Problem::kM0;
    static constexpr index_t kN0 = Problem::kN0;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;
    static constexpr index_t kQKHeaddim  = Problem::kQKHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ      = Problem::kPadSeqLenQ;
    static constexpr bool kPadHeadDimQ     = Problem::kPadHeadDimQ;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;

    static constexpr index_t kAlignmentQGradAcc =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentPostQGradAcc<Problem>();
    static constexpr index_t kAlignmentQGrad =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentPostQGrad<Problem>();

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    // Zero-fill only — used when the mask leaves every dq_acc split unwritten for
    // a given Q-tile (e.g., narrow SWA with a Q position outside the window).
    // dq = 0 is mathematically correct: no K positions contribute to this Q's gradient.
    //
    // We mirror the "Convert only" overload's pattern (create an AccDataType tile,
    // cast to QGradDataType, store) rather than directly creating a QGradDataType
    // tile via make_static_distributed_tensor + clear_tile + store_tile. The latter
    // empirically only stores half the kM0×kQKHeaddim tile rows for some
    // distribution configurations — a subtle internal-state issue. Going through
    // cast_tile from an AccDataType source produces a tile whose store_tile covers
    // the full row range, matching the simple convert path.
    template <typename QGradDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE void
    operator()(QGradDramBlockWindowTmp& dq_dram_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<QGradDataType,
                           remove_cvref_t<typename QGradDramBlockWindowTmp::DataType>>,
            "wrong!");
        static_assert(kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}], "wrong!");

        // Reuse the existing simple convert overload pattern: attach the canonical
        // MakePostQGradDramTileDistribution to a window over the dq output itself,
        // load_tile (gives us a properly-formed tile), clear in-place, then store.
        // The extra load reads garbage from dq (torch::empty_like(q)) — wasted but small.
        // We do this because make_static_distributed_tensor + clear_tile + store_tile
        // empirically only writes half the tile rows; load_tile produces a tile whose
        // store_tile covers the full row range.
        auto dq_dram_window_with_dist = make_tile_window(
            dq_dram_block_window_tmp.get_bottom_tensor_view(),
            dq_dram_block_window_tmp.get_window_lengths(),
            dq_dram_block_window_tmp.get_window_origin(),
            Policy::template MakePostQGradDramTileDistribution<Problem>());
        auto dq = load_tile(dq_dram_window_with_dist);
        clear_tile(dq);
        store_tile(dq_dram_block_window_tmp, dq);
    }

    // Convert only
    template <typename QGradAccDramBlockWindowTmp, typename QGradDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE void
    operator()(const QGradAccDramBlockWindowTmp& dq_acc_dram_block_window_tmp,
               QGradDramBlockWindowTmp& dq_dram_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<AccDataType,
                           remove_cvref_t<typename QGradAccDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QGradDataType,
                               remove_cvref_t<typename QGradDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}], "wrong!");

        auto dq_acc_dram_window =
            make_tile_window(dq_acc_dram_block_window_tmp.get_bottom_tensor_view(),
                             dq_acc_dram_block_window_tmp.get_window_lengths(),
                             dq_acc_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakePostQGradDramTileDistribution<Problem>());

        auto dq_acc   = load_tile(dq_acc_dram_window);
        const auto dq = cast_tile<QGradDataType>(dq_acc);

        store_tile(dq_dram_block_window_tmp, dq);
    }

    // Reduce + Convert
    template <typename QGradAccDramBlockWindowTmp, typename QGradDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE void
    operator()(const QGradAccDramBlockWindowTmp& dq_acc_dram_block_window_tmp,
               QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               index_t nsplits) const
    {
        static_assert(
            std::is_same_v<AccDataType,
                           remove_cvref_t<typename QGradAccDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QGradDataType,
                               remove_cvref_t<typename QGradDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}], "wrong!");

        auto dq_acc_dram_window =
            make_tile_window(dq_acc_dram_block_window_tmp.get_bottom_tensor_view(),
                             dq_acc_dram_block_window_tmp.get_window_lengths(),
                             dq_acc_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakePostQGradAccDramTileDistribution<Problem>());

        auto dq_acc = decltype(load_tile(dq_acc_dram_window)){};
        clear_tile(dq_acc);

        constexpr auto dq_acc_spans = decltype(dq_acc)::get_distributed_spans();
        index_t i_total_loops       = 0;
        auto dq_acc_buf             = load_tile(dq_acc_dram_window);
        move_tile_window(dq_acc_dram_window, {1, 0, 0});

        // Use while-loop (not do-while) so nsplits == 1 is correct.
        // For nsplits == 1: we prefetched split 0 above, the loop body is skipped,
        // and the tail accumulate below sums split 0 correctly with no OOB load.
        // For nsplits >= 2: pipelined pattern is preserved (prefetch next while
        // accumulating current).
        while(i_total_loops < (nsplits - 1))
        {
            sweep_tile_span(dq_acc_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(dq_acc_spans[number<1>{}], [&](auto idx1) {
                    sweep_tile_span(dq_acc_spans[number<2>{}], [&](auto idx2) {
                        constexpr auto n_i_j_idx = make_tuple(idx0, idx1, idx2);
                        dq_acc(n_i_j_idx) += dq_acc_buf(n_i_j_idx);
                    });
                });
            });

            dq_acc_buf = load_tile(dq_acc_dram_window);
            move_tile_window(dq_acc_dram_window, {1, 0, 0});

            i_total_loops += 1;
        }

        sweep_tile_span(dq_acc_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(dq_acc_spans[number<1>{}], [&](auto idx1) {
                sweep_tile_span(dq_acc_spans[number<2>{}], [&](auto idx2) {
                    constexpr auto n_i_j_idx = make_tuple(idx0, idx1, idx2);
                    dq_acc(n_i_j_idx) += dq_acc_buf(n_i_j_idx);
                });
            });
        });

        // declare dq
        constexpr auto dq_converted_dstr =
            Policy::template MakePostQGradAccDramTileDistribution<Problem>();
        auto dq_converted = make_static_distributed_tensor<QGradDataType>(dq_converted_dstr);

        sweep_tile_span(dq_acc_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(dq_acc_spans[number<1>{}], [&](auto idx1) {
                sweep_tile_span(dq_acc_spans[number<2>{}], [&](auto idx2) {
                    constexpr auto n_i_j_idx = make_tuple(idx0, idx1, idx2);
                    dq_converted(n_i_j_idx)  = type_convert<QGradDataType>(dq_acc[n_i_j_idx]);
                });
            });
        });

        constexpr auto dq_dstr = Policy::template MakePostQGradDramTileDistribution<Problem>();
        auto dq                = make_static_distributed_tensor<QGradDataType>(dq_dstr);
        dq.get_thread_buffer() = dq_converted.get_thread_buffer();

        store_tile(dq_dram_block_window_tmp, dq);
    }
};

} // namespace ck_tile
