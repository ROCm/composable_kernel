// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdOGradDotO
{
    using ODataType     = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
    using DDataType     = remove_cvref_t<typename Problem::DDataType>;
    using LSEDataType   = remove_cvref_t<typename Problem::LSEDataType>; // needed for sink gradient

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;
    static constexpr index_t kVHeaddim   = Problem::kVHeaddim;

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    template <typename ODramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp>
    // Computes D = diag(dO * O) and optionally accumulates the sink token gradient.
    // sink_value: log-space sink score; pass -inf and atomic_sink_grad_ptr=nullptr to skip sink.
    // atomic_sink_grad_ptr: per-head accumulator in global memory; nullptr disables sink path.
    CK_TILE_HOST_DEVICE void operator()(const ODramBlockWindowTmp& o_dram_block_window_tmp,
                                        const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
                                        const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
                                        DDramBlockWindowTmp& d_dram_block_window_tmp,
                                        const LSEDataType sink_value,
                                        float p_undrop,
                                        LSEDataType* atomic_sink_grad_ptr = nullptr) const
    {
        static_assert(
            std::is_same_v<ODataType, remove_cvref_t<typename ODramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
            "wrong!");
        // atomic_sink_grad_ptr is reinterpret_cast to float* in the sink path;
        // ensure LSEDataType is float so the cast is well-defined.
        static_assert(std::is_same_v<LSEDataType, float>,
                      "sink gradient atomicAdd requires LSEDataType == float");

        static_assert(kBlockSize == ODramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kBlockSize ==
                              OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kBlockSize == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");

        auto o_dram_window =
            make_tile_window(o_dram_block_window_tmp.get_bottom_tensor_view(),
                             o_dram_block_window_tmp.get_window_lengths(),
                             o_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakePreODramTileDistribution<Problem>());

        auto o = load_tile(o_dram_window);

        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             do_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakePreOGradDramTileDistribution<Problem>());

        auto do_ = load_tile(do_dram_window);

        // D[q] = sum_j(O[q,j] * dO[q,j]), used in softmax backward
        constexpr auto d_dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                o.get_tile_distribution().get_static_tile_distribution_encoding(), sequence<1>{}));

        auto d = make_static_distributed_tensor<DDataType>(d_dstr);
        clear_tile(d);

        constexpr auto o_spans = decltype(o)::get_distributed_spans();
        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                d(i_idx) +=
                    (type_convert<DDataType>(o[i_j_idx]) * type_convert<DDataType>(do_[i_j_idx]));
            });
        });

        // Scale by p_undrop (=1 when dropout is disabled)
        tile_elementwise_inout([&p_undrop](auto& x) { x = x * p_undrop; }, d);

        store_tile(d_dram_block_window_tmp, d);

        // Sink gradient path: skipped entirely when atomic_sink_grad_ptr is nullptr
        if(atomic_sink_grad_ptr != nullptr)
        {
            // Load LSE only on the sink path to avoid unnecessary global memory reads
            constexpr auto lse_dstr =
                make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                    o.get_tile_distribution().get_static_tile_distribution_encoding(),
                    sequence<1>{}));
            auto lse_dram_window =
                make_tile_window(lse_dram_block_window_tmp.get_bottom_tensor_view(),
                                 lse_dram_block_window_tmp.get_window_lengths(),
                                 lse_dram_block_window_tmp.get_window_origin(),
                                 lse_dstr);
            auto lse_ = load_tile(lse_dram_window);

            // Compute per-query contribution: -P_sink[q] * D[q]
            // where P_sink[q] = exp2(sink_value - log2e*lse[q])
            // sink_value has already been pre-multiplied by log2e at the kernel call site,
            // so exp2(sink_value - log2e*lse) == exp(raw_sink - lse).
            // exp2 maps directly to the v_exp_f32 hardware instruction on AMD GPUs.
            // Always accumulate in float regardless of DDataType to avoid precision loss
            // and to ensure atomicAdd works correctly on all architectures.
            auto sink_val_tensor = make_static_distributed_tensor<float>(d_dstr);
            tile_elementwise_inout(
                [&](auto& s_out, const auto& l_in, const auto& d_in) {
                    float p_sink = exp2(type_convert<float>(sink_value) -
                                        log2e_v<float> * type_convert<float>(l_in));
                    s_out        = -p_sink * type_convert<float>(d_in);
                },
                sink_val_tensor,
                lse_,
                d);

            // Reduce contributions held by this thread
            float thread_sum       = 0.f;
            constexpr auto s_spans = decltype(sink_val_tensor)::get_distributed_spans();
            sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                thread_sum += sink_val_tensor(i_idx);
            });

            // Warp-level reduction: fold thread_sum across lanes so only one
            // atomicAdd per warp is issued instead of one per thread.
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
            const index_t warp_sz = get_warp_size();
            for(index_t offset = warp_sz >> 1; offset > 0; offset >>= 1)
                thread_sum += warp_shuffle_down(thread_sum, offset);

            // Only lane 0 of each warp writes to global memory.
            // Note: this atomicAdd is non-deterministic across runs regardless of the
            // -deterministic flag, because d_sink is a single scalar per head accumulated
            // across all thread-blocks. The practical impact is negligible for this value.
            if(get_lane_id() == 0)
                atomicAdd(reinterpret_cast<float*>(atomic_sink_grad_ptr), thread_sum);
#endif
        }
    }
};

} // namespace ck_tile
