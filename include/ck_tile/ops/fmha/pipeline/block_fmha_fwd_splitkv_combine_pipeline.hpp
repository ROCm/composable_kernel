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

    using QDataType    = remove_cvref_t<typename Problem::QDataType>;
    using LSEDataType  = remove_cvref_t<typename Problem::LSEDataType>;
    using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType    = remove_cvref_t<typename Problem::ODataType>;
    using FmhaMask     = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape                = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                       = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0            = BlockFmhaShape::kM0;
    static constexpr index_t kN0            = BlockFmhaShape::kN0;
    static constexpr index_t kK0            = BlockFmhaShape::kK0;
    static constexpr index_t kN1            = BlockFmhaShape::kN1;
    static constexpr index_t kK1            = BlockFmhaShape::kK1;
    static constexpr index_t kK0BlockLength = BlockFmhaShape::kK0BlockLength;

    static constexpr bool kIsGroupMode  = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ   = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ  = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV  = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum      = Problem::BiasEnum;
    static constexpr bool kStoreLSE     = Problem::kStoreLSE;
    static constexpr bool kHasDropout   = Problem::kHasDropout;
    static constexpr index_t kMaxSplits = Problem::kMaxSplits;

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kK0BlockLength <= 32)
            {
                return 2;
            }
            else if constexpr(kK0BlockLength <= 64)
            {
                return 3;
            }
            else if constexpr(kK0BlockLength <= 128)
            {
                return 2;
            }
            else if constexpr(kK0BlockLength <= 256)
            {
                return 1;
            }
        }
    }();

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        /// TODO: add padding to avoid bank conflict
        return (kM0 * kMaxSplits * sizeof(LSEDataType));
    }

#define MARKER(msg)                     \
    __builtin_amdgcn_sched_barrier(0);  \
    asm volatile("; [POYENC] " msg ::); \
    __builtin_amdgcn_sched_barrier(0)

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
               void* smem_ptr,
               index_t num_splits,
               index_t max_seqlen_q) const
    {
        LSEDataType* lse_acc_lds_ptr =
            static_cast<LSEDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto lse_acc_dist = Policy::template MakeLSEaccDramTileDistribution<Problem>();
        auto lse_acc_dram_window =
            make_tile_window(lse_acc_dram_block_window_tmp.get_bottom_tensor_view(),
                             lse_acc_dram_block_window_tmp.get_window_lengths(),
                             lse_acc_dram_block_window_tmp.get_window_origin(),
                             lse_acc_dist);

        auto lse_acc = load_tile(lse_acc_dram_window); // [kMaxSplits, kM0]

#if defined(ENABLE_DEBUG_STMTS)
#define DEBUG_STMTS if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == TID)
#else
#define DEBUG_STMTS if(false)
#endif
        // copy lse_acc to LDS
        {
            using DataType               = LSEDataType;
            using StaticTileDistribution = decltype(lse_acc_dist);

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

                    lse_acc_lds_ptr[row + col * kMaxSplits] = lse_acc(distributed_indices);
                });
            });
        }
        block_sync_lds();

        auto lse_accum_dist = Policy::template MakeLSEaccTDramTileDistribution<Problem>();
        auto lse_accum      = make_static_distributed_tensor<LSEDataType>(lse_accum_dist);

        // copy LDS to lse_accum
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

                    if(col < num_splits)
                    {
                        lse_accum(distributed_indices) = lse_acc_lds_ptr[col + row * kMaxSplits];
                    }
                    else
                    {
                        lse_accum(distributed_indices) = -numeric<LSEDataType>::infinity();
                    }

                    DEBUG_STMTS
                    {
                        printf("[POYENC][DEVICE] lse_accum[%2d,%2d]: %11.7f\n",
                               row,
                               col,
                               lse_accum(distributed_indices));
                    }
                });
            });
        }

        // calculate row_max of lse_accum
        const auto f_max = [](auto e0, auto e1) { return ck_tile::max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        auto lse_max = block_tile_reduce<LSEDataType>(
            lse_accum, sequence<1>{}, f_max, -numeric<LSEDataType>::infinity());
        block_tile_reduce_sync(lse_max, f_max, bool_constant<false>{});

#if defined(PRINT_LSE_MAX)
        DEBUG_STMTS
        {
            constexpr auto out_spans =
                static_distributed_tensor<LSEDataType, decltype(lse_max.get_tile_distribution())>::
                    get_distributed_spans();
            sweep_tile_span(out_spans[number<0>{}], [&](auto idx0) {
                constexpr auto distributed_indices = make_tuple(idx0);
                const auto x_indices               = get_x_indices_from_distributed_indices(
                    lse_max.get_tile_distribution(), distributed_indices);

                const auto row = x_indices.at(number<0>{});

                printf(
                    "[POYENC][DEVICE] lse_max[%2d]: %11.7f\n", row, lse_max(distributed_indices));
            });
        }
#endif

        static const auto get_validated_m = [](LSEDataType raw_m) {
            /// NOTICE: bias might be materialized mask including -inf values, need
            /// consideration
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         FmhaMask::IsMasking)
            {
                return raw_m == -numeric<LSEDataType>::infinity() ? type_convert<LSEDataType>(0.f)
                                                                  : raw_m;
            }
            else
            {
                return raw_m;
            }
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

#if 0
                    // from dist tensor
                    p_compute(i_j_idx) = ck_tile::exp(lse_accum(i_j_idx) - get_validated_m(lse_max(i_idx)));
#else
                    if (col < num_splits) {
                        // from shared memory
                        p_compute(i_j_idx) = ck_tile::exp(lse_acc_lds_ptr[col + row * kMaxSplits] -
                                                          get_validated_m(lse_max(i_idx)));
                    }
#endif
#if 0
                    DEBUG_STMTS
                    {
                        printf("[POYENC][DEVICE] p_compute[%2d,%2d]: %11.7f\n",
                               row,
                               col,
                               p_compute(i_j_idx));
                    }
#endif
                });
            });
        }
        __syncthreads();

        auto lse_sum = block_tile_reduce<LSEDataType>(
            p_compute, sequence<1>{}, f_sum, type_convert<LSEDataType>(0));
        block_tile_reduce_sync(lse_sum, f_sum, bool_constant<false>{});

#if defined(PRINT_LSE_SUM)
        DEBUG_STMTS
        {
            constexpr auto out_spans =
                static_distributed_tensor<LSEDataType, decltype(lse_sum.get_tile_distribution())>::
                    get_distributed_spans();
            sweep_tile_span(out_spans[number<0>{}], [&](auto idx0) {
                constexpr auto distributed_indices = make_tuple(idx0);
                const auto x_indices               = get_x_indices_from_distributed_indices(
                    lse_sum.get_tile_distribution(), distributed_indices);

                const auto row = x_indices.at(number<0>{});

                printf(
                    "[POYENC][DEVICE] lse_sum[%2d]: %11.7f\n", row, lse_sum(distributed_indices));
            });
        }
#endif

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
                    lse_logsum(distributed_indices) =
                        ck_tile::log(lse_sum(distributed_indices)) + lse_max(distributed_indices);
                }

#if defined(PRINT_LSE_LOGSUM)
                DEBUG_STMTS
                {
                    const auto x_indices = get_x_indices_from_distributed_indices(
                        lse_logsum.get_tile_distribution(), distributed_indices);

                    const auto row = x_indices.at(number<0>{});
                    printf("[POYENC][DEVICE] lse_logsum[%d]: %11.7f\n",
                           row,
                           lse_logsum(distributed_indices));
                }
#endif
            });
        }

#if defined(PRINT_LSE_ACCUM)
        DEBUG_STMTS
        {
            for(index_t row = 0; row < kM0; ++row)
            {
                printf("[POYENC][DEVICE] lse_accum[%d] = ", row);
                for(index_t col = 0; col < num_splits; ++col)
                {
                    printf("%11.7f", lse_acc_lds_ptr[col + row * kMaxSplits]);
                }
                printf("\n");
            }
        }
#endif

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
                    lse_acc_lds_ptr[col + row * kMaxSplits] = ck_tile::exp(
                        lse_acc_lds_ptr[col + row * kMaxSplits] - lse_logsum(distributed_indices));
                }
            });
        }
        block_sync_lds();

#if defined(PRINT_LSE_SCALE)
        DEBUG_STMTS
        {
            for(index_t row = 0; row < 32; ++row)
            {
                printf("[POYENC][DEVICE] lse_scale[%2d] = ", row);
                for(index_t col = 0; col < num_splits; ++col)
                {
                    printf("%11.7f", lse_acc_lds_ptr[col + row * kMaxSplits]);
                }
                printf("\n");
            }
        }
#endif

        if constexpr(kStoreLSE)
        {
            static_assert(kBlockSize == 256);
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

        // [POYENC] added
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

                        LSEDataType lse_scale = lse_acc_lds_ptr[i_split + row * kMaxSplits];
                        o_acc(distributed_indices) += lse_scale * o_tile(distributed_indices);
                    });
                });
            }

            move_tile_window(o_acc_dram_window, {max_seqlen_q, 0});
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
                                        void* smem_ptr,
                                        index_t num_splits,
                                        index_t max_seqlen_q) const
    {
        return operator()(lse_acc_dram_block_window,
                          o_acc_dram_block_window,
                          lse_dram_block_window,
                          identity{},
                          identity{},
                          smem_ptr,
                          num_splits,
                          max_seqlen_q);
    }
};

} // namespace ck_tile
