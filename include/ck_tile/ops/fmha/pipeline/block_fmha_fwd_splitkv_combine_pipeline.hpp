// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {
namespace detail {
template <index_t N>
struct log2;

template <>
struct log2<16> : std::integral_constant<index_t, 4>
{
};

template <>
struct log2<32> : std::integral_constant<index_t, 5>
{
};

template <>
struct log2<64> : std::integral_constant<index_t, 6>
{
};

template <>
struct log2<128> : std::integral_constant<index_t, 7>
{
};
} // namespace detail

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

    static constexpr index_t kAlignmentLSE =
        kPadSeqLenQ ? 1 : Policy::template GetAlignmentLSE<Problem>();
    static constexpr index_t kAlignmentLSEacc = kAlignmentLSE;

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
                constexpr std::array<int, 4> occupancy{3, 3, 3, 1};
                return occupancy[detail::log2<kMaxSplits>::value - 4];
            }
            else if constexpr(kHeadDimV <= 128)
            {
                constexpr std::array<int, 4> occupancy{3, 3, 2, 1};
                return occupancy[detail::log2<kMaxSplits>::value - 4];
            }
            else if constexpr(kHeadDimV <= 256)
            {
                constexpr std::array<int, 4> occupancy{2, 2, 2, 1};
                return occupancy[detail::log2<kMaxSplits>::value - 4];
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
        // lse_acc tile in LDS
        LSEDataType* lse_acc_lds_ptr =
            static_cast<LSEDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto lse_acc_lds = [=, lds_desc = Policy::template MakeLSEaccLdsBlockDescriptor<Problem>()](
                               index_t row, index_t col) -> LSEDataType& {
            return lse_acc_lds_ptr[lds_desc.calculate_offset(make_tuple(row, col))];
        };

        auto lse_acc_lds_write_window = [&]() {
            auto view = make_tensor_view<address_space_enum::lds>(
                lse_acc_lds_ptr, Policy::template MakeLSEaccLdsStoreBlockDescriptor<Problem>());
            return make_tile_window(view, make_tuple(number<kMaxSplits>{}, number<kM0>{}), {0, 0});
        }();

        auto lse_acc_dram_window =
            make_tile_window(lse_acc_dram_block_window_tmp.get_bottom_tensor_view(),
                             lse_acc_dram_block_window_tmp.get_window_lengths(),
                             lse_acc_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeLSEaccDramTileDistribution<Problem>());

        // copy lse_acc tile (shape=[kMaxSplits, kM0]) to LDS (shape=[kMaxSplits, kM0]).
        auto lse_acc_tile = load_tile(lse_acc_dram_window);
        store_tile(lse_acc_lds_write_window, lse_acc_tile);
        block_sync_lds();

        auto lse_accum = make_static_distributed_tensor<LSEDataType>(
            Policy::template MakeLSEaccRegTileDistribution<Problem>());

        // copy LDS (shape=[kM0, kMaxSplits]) to lse_accum (shape=[kM0, max(kMaxSplits, warp_size)])
        // this will extend the distributed tensor width so that each thread in wave have data to
        // reduce.
        {
            constexpr auto spans = decltype(lse_accum)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    const auto x_indices   = get_x_indices_from_distributed_indices(
                        lse_accum.get_tile_distribution(), i_j_idx);

                    const auto col = x_indices.at(number<1>{});
                    if(col < num_splits)
                    {
                        const auto row = x_indices.at(number<0>{});

                        lse_accum(i_j_idx) = lse_acc_lds(row, col);
                    }
                    else
                    {
                        lse_accum(i_j_idx) = -numeric<LSEDataType>::infinity();
                    }
                });
            });
        }

        // compute the logsumexp of the LSE along the split dimension.
        const auto f_max = [](auto e0, auto e1) { return ck_tile::max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        auto lse_max = block_tile_reduce<LSEDataType>(
            lse_accum, sequence<1>{}, f_max, -numeric<LSEDataType>::infinity());
        block_tile_reduce_sync(lse_max, f_max, bool_constant<false>{});

        static const auto get_validated_m = [](LSEDataType raw_m) {
            return raw_m == -numeric<LSEDataType>::infinity() ? type_convert<LSEDataType>(0.f)
                                                              : raw_m;
        };

        decltype(lse_accum) lse_exp;
        {
            constexpr auto spans = decltype(lse_exp)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    lse_exp(i_j_idx) =
                        ck_tile::exp(lse_accum(i_j_idx) - get_validated_m(lse_max(i_idx)));
                });
            });
        }

        auto lse_sum = block_tile_reduce<LSEDataType>(
            lse_exp, sequence<1>{}, f_sum, type_convert<LSEDataType>(0));
        block_tile_reduce_sync(lse_sum, f_sum, bool_constant<false>{});

        decltype(lse_max) lse_logsum;
        {
            constexpr auto spans = decltype(lse_logsum)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                if(lse_sum(i_idx) == 0.f || lse_sum(i_idx) != lse_sum(i_idx))
                {
                    lse_logsum(i_idx) = numeric<LSEDataType>::infinity();
                }
                else
                {
                    lse_logsum(i_idx) =
                        ck_tile::log(lse_sum(i_idx)) + get_validated_m(lse_max(i_idx));
                }
            });
        }

        // store the lse scales in shared memory.
        {
            constexpr auto spans = decltype(lse_accum)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    const auto x_indices = get_x_indices_from_distributed_indices(
                        lse_accum.get_tile_distribution(), i_j_idx);

                    const auto col = x_indices.at(number<1>{});
                    if(col < num_splits)
                    {
                        const auto row = x_indices.at(number<0>{});

                        lse_acc_lds(row, col) =
                            ck_tile::exp(lse_accum(i_j_idx) - lse_logsum(i_idx));
                    }
                });
            });
        }
        block_sync_lds();

        if constexpr(kStoreLSE)
        {
            constexpr auto spans = decltype(lse_logsum)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                if(lse_logsum(i_idx) == numeric<LSEDataType>::infinity())
                {
                    lse_logsum(i_idx) = -numeric<LSEDataType>::infinity();
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
        auto o_acc = make_static_distributed_tensor<OaccDataType>(o_acc_dist);
        clear_tile(o_acc);

        const index_t padded_max_seqlen_q = integer_divide_ceil(max_seqlen_q, kM0) * kM0;

        for(index_t i_split = 0; i_split < num_splits; ++i_split)
        {
            auto o_tile = load_tile(o_acc_dram_window);
            {
                constexpr auto spans = decltype(o_acc)::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        const auto x_indices   = get_x_indices_from_distributed_indices(
                            o_acc.get_tile_distribution(), i_j_idx);

                        const auto row = x_indices.at(number<0>{});

                        const LSEDataType lse_scale = lse_acc_lds(row, i_split);
                        o_acc(i_j_idx) += lse_scale * o_tile(i_j_idx);
                    });
                });
            }

            move_tile_window(o_acc_dram_window, {padded_max_seqlen_q, 0});
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
