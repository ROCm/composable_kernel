// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BlockFmhaFwdSplitKVCombinePipelineDefaultPolicy>
struct BlockFmhaFwdSplitKVCombinePipeline
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using LSEDataType  = remove_cvref_t<typename Problem::LSEDataType>;
    using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType    = remove_cvref_t<typename Problem::ODataType>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kHeadDimV = Problem::kHeadDimV;
    static constexpr index_t kM0       = Problem::kM0;
    static constexpr index_t kN1       = Problem::kN1;

    static constexpr bool kIsGroupMode  = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ   = Problem::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV  = Problem::kPadHeadDimV;
    static constexpr bool kStoreLSE     = Problem::kStoreLSE;
    static constexpr index_t kMaxSplits = Problem::kMaxSplits;

    static constexpr index_t kAlignmentLSE = Policy::template GetAlignmentLSE<Problem>();

    static constexpr index_t kAlignmentOacc =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentOacc<Problem>();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kHeadDimV <= 32)
            {
                return 2;
            }
            else if constexpr(kHeadDimV <= 64)
            {
                return 3;
            }
            else if constexpr(kHeadDimV <= 128)
            {
                return 2;
            }
            else if constexpr(kHeadDimV <= 256)
            {
                return 1;
            }
        }
    }();

    static constexpr const char* name = "unused";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename LSEaccDramBlockWindowTmp,
              typename OaccDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename LSEElementFunction,
              typename OaccElementFunction>
    CK_TILE_HOST_DEVICE auto
    operator()(const LSEaccDramBlockWindowTmp& lse_acc_dram_block_window_tmp,
               const OaccDramBlockWindowTmp& o_acc_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp,
               const LSEElementFunction& lse_element_func,
               const OaccElementFunction& o_acc_element_func,
               index_t num_splits,
               index_t max_seqlen_q,
               void* smem_ptr) const
    {
        // LSEacc tile in LDS
        LSEDataType* lse_acc_lds_ptr =
            static_cast<LSEDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto lse_acc_lds_for_write = make_tensor_view<address_space_enum::lds>(
            lse_acc_lds_ptr, Policy::template MakeLSEaccLdsBlockDescriptor<Problem>());
        auto lse_acc_lds_write_window = make_tile_window(
            lse_acc_lds_for_write, make_tuple(number<kMaxSplits>{}, number<kM0>{}), {0, 0});
        auto lse_acc_lds_ms_m0_for_write = Policy::template MakeLSEaccLdsBlockDescriptor<Problem>();

        auto lse_acc_dist = Policy::template MakeLSEaccDramTileDistribution<Problem>();
        auto lse_acc_dram_window =
            make_tile_window(lse_acc_dram_block_window_tmp.get_bottom_tensor_view(),
                             lse_acc_dram_block_window_tmp.get_window_lengths(),
                             lse_acc_dram_block_window_tmp.get_window_origin(),
                             lse_acc_dist);

        // copy lse_acc to LDS
        auto lse_acc = load_tile(lse_acc_dram_window);
        store_tile(lse_acc_lds_write_window, lse_acc);
        block_sync_lds();

        auto lse_accum_dist = Policy::template MakeLSEaccTDramTileDistribution<Problem>();
        auto lse_accum      = make_static_distributed_tensor<LSEDataType>(lse_accum_dist);

        // copy LDS to lse_accum (transpose)
        {
            using DataType               = LSEDataType;
            using StaticTileDistribution = decltype(lse_accum_dist);
            constexpr auto out_spans =
                static_distributed_tensor<DataType,
                                          StaticTileDistribution>::get_distributed_spans();
            sweep_tile_span(out_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(out_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto distributed_indices = make_tuple(idx0, idx1);
                    const auto x_indices               = get_x_indices_from_distributed_indices(
                        StaticTileDistribution{}, distributed_indices);

                    const auto row = x_indices.at(number<0>{});
                    const auto col = x_indices.at(number<1>{});

                    auto offset =
                        lse_acc_lds_ms_m0_for_write.calculate_offset(make_tuple(col, row));
                    lse_accum(distributed_indices) = lse_acc_lds_ptr[offset];
                });
            });
        }

        // calculate row_max of lse_accum
        const auto f_max = [](auto e0, auto e1) { return ck_tile::max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        auto lse_max = block_tile_reduce<LSEDataType>(
            lse_accum, sequence<1>{}, f_max, -numeric<LSEDataType>::infinity());
        block_tile_reduce_sync(lse_max, f_max, bool_constant<false>{});

        static const auto get_validated_m = [](LSEDataType raw_m) {
            /// NOTICE: bias might be materialized mask including -inf values, need
            /// consideration
            return raw_m == -numeric<LSEDataType>::infinity() ? type_convert<LSEDataType>(0.f)
                                                              : raw_m;
        };

        auto p_compute = make_static_distributed_tensor<LSEDataType>(
            lse_accum.get_tile_distribution()); // Pcompute{j}
        clear_tile(p_compute);
        {
            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    const auto x_indices   = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), i_j_idx);

                    const auto row = x_indices.at(number<0>{});
                    const auto col = x_indices.at(number<1>{});

                    auto offset =
                        lse_acc_lds_ms_m0_for_write.calculate_offset(make_tuple(col, row));
                    if(col < num_splits)
                    {
                        // from shared memory
                        p_compute(i_j_idx) =
                            ck_tile::exp(lse_acc_lds_ptr[offset] - get_validated_m(lse_max(i_idx)));
                    }
                });
            });
        }
        __syncthreads();

        auto lse_sum = block_tile_reduce<LSEDataType>(
            p_compute, sequence<1>{}, f_sum, type_convert<LSEDataType>(0));
        block_tile_reduce_sync(lse_sum, f_sum, bool_constant<false>{});

        decltype(lse_max) lse_logsum;
        {
            constexpr auto out_spans = static_distributed_tensor<
                LSEDataType,
                decltype(lse_logsum.get_tile_distribution())>::get_distributed_spans();
            sweep_tile_span(out_spans[number<0>{}], [&](auto idx0) {
                constexpr auto distributed_indices = make_tuple(idx0);

                if(lse_sum(distributed_indices) == 0.f ||
                   lse_sum(distributed_indices) != lse_sum(distributed_indices))
                {
                    lse_logsum(distributed_indices) = numeric<LSEDataType>::infinity();
                }
                else
                {
                    lse_logsum(distributed_indices) = ck_tile::log(lse_sum(distributed_indices)) +
                                                      get_validated_m(lse_max(distributed_indices));
                }
            });
        }

        // write lse scales into LDS
        {
            constexpr auto out_spans =
                static_distributed_tensor<LSEDataType, decltype(lse_sum.get_tile_distribution())>::
                    get_distributed_spans();
            sweep_tile_span(out_spans[number<0>{}], [&](auto idx0) {
                constexpr auto distributed_indices = make_tuple(idx0);
                const auto x_indices               = get_x_indices_from_distributed_indices(
                    lse_sum.get_tile_distribution(), distributed_indices);

                const auto row = x_indices.at(number<0>{});

                for(index_t col = 0; col < num_splits; ++col)
                {
                    auto offset =
                        lse_acc_lds_ms_m0_for_write.calculate_offset(make_tuple(col, row));
                    lse_acc_lds_ptr[offset] =
                        ck_tile::exp(lse_acc_lds_ptr[offset] - lse_logsum(distributed_indices));
                }
            });
        }
        block_sync_lds();

        if constexpr(kStoreLSE)
        {
            constexpr auto out_spans = static_distributed_tensor<
                LSEDataType,
                decltype(lse_logsum.get_tile_distribution())>::get_distributed_spans();
            sweep_tile_span(out_spans[number<0>{}], [&](auto idx0) {
                constexpr auto distributed_indices = make_tuple(idx0);

                if(lse_logsum(distributed_indices) == numeric<LSEDataType>::infinity())
                {
                    lse_logsum(distributed_indices) = -numeric<LSEDataType>::infinity();
                }
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse_logsum));
        }

        auto o_acc_dist = Policy::template MakeOaccDramTileDistribution<Problem>();
        auto o_acc_dram_window =
            make_tile_window(o_acc_dram_block_window_tmp.get_bottom_tensor_view(),
                             o_acc_dram_block_window_tmp.get_window_lengths(),
                             o_acc_dram_block_window_tmp.get_window_origin(),
                             o_acc_dist);
        auto o_acc = make_static_distributed_tensor<OaccDataType>(o_acc_dist); // Pcompute{j}
        clear_tile(o_acc);

        const index_t new_max_seqlen_q = integer_divide_ceil(max_seqlen_q, kM0) * kM0;

        for(index_t i_split = 0; i_split < num_splits; ++i_split)
        {
            auto o_tile = load_tile(o_acc_dram_window);
            {
                using DataType = OaccDataType;
                constexpr auto out_spans =
                    static_distributed_tensor<DataType,
                                              decltype(o_acc_dist)>::get_distributed_spans();
                sweep_tile_span(out_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(out_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto distributed_indices = make_tuple(idx0, idx1);
                        const auto x_indices =
                            get_x_indices_from_distributed_indices(o_acc_dist, distributed_indices);

                        const auto row = x_indices.at(number<0>{});

                        auto offset =
                            lse_acc_lds_ms_m0_for_write.calculate_offset(make_tuple(i_split, row));

                        LSEDataType lse_scale = lse_acc_lds_ptr[offset];
                        o_acc(distributed_indices) += lse_scale * o_tile(distributed_indices);
                    });
                });
            }

            move_tile_window(o_acc_dram_window, {new_max_seqlen_q, 0});
        }

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename LSEaccDramBlockWindow,
              typename OaccDramBlockWindow,
              typename LSEDramBlockWindow>
    CK_TILE_HOST_DEVICE auto operator()(const LSEaccDramBlockWindow& lse_acc_dram_block_window,
                                        const OaccDramBlockWindow& o_acc_dram_block_window,
                                        LSEDramBlockWindow& lse_dram_block_window,
                                        index_t num_splits,
                                        index_t max_seqlen_q,
                                        void* smem_ptr) const
    {
        return operator()(lse_acc_dram_block_window,
                          o_acc_dram_block_window,
                          lse_dram_block_window,
                          identity{},
                          identity{},
                          num_splits,
                          max_seqlen_q,
                          smem_ptr);
    }
};

} // namespace ck_tile
