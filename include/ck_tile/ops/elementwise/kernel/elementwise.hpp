// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_>
struct ElementWiseKernel
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType            = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType      = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType            = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using ElementWiseOperation = ck_tile::remove_cvref_t<typename Problem::ElementWiseOperation>;

    CK_TILE_DEVICE void operator()(
        const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, const index_t M0) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_a, make_tuple(M0), make_tuple(1), number<S::Vector_M>{});

        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_b, make_tuple(M0), make_tuple(1), number<S::Vector_M>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  amd_buffer_coherence_enum::slc>(
            p_y, make_tuple(M0), make_tuple(1), number<S::Vector_M>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_window_a = make_tile_window(x_m_n_a,
                                           make_tuple(number<S::Block_M>{}),
                                           {iM},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto x_window_b = make_tile_window(x_m_n_b,
                                           make_tuple(number<S::Block_M>{}),
                                           {iM},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(number<S::Block_M>{}),
                                         {iM},
                                         Policy::template MakeXBlockTileDistribution<Problem>());
        
        index_t num_m_tile_iteration = 
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(M0, S::Block_M));

        for(int i = __builtin_amdgcn_readfirstlane(0); i < num_m_tile_iteration; ++i)
        {
            // Load tile data
            const auto xa = load_tile(x_window_a);
            const auto xb = load_tile(x_window_b);
            auto y_compute = load_tile(y_window);

            // Process the vector add
            constexpr auto spans = decltype(xa)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx) {
            
                const auto tile_idx = make_tuple(idx);
                const auto a_val = type_convert<ComputeDataType>(xa[tile_idx]);
                const auto b_val = type_convert<ComputeDataType>(xb[tile_idx]);
                ElementWiseOperation{}(y_compute(tile_idx), a_val, b_val);
            });

            // Store results
            store_tile(y_window, cast_tile<YDataType>(y_compute));
            
            // Move windows to next block (corrected)
            // For 1D operation, we only need to move along the M dimension by Block_M elements
            move_tile_window(x_window_a, {S::Block_M});
            move_tile_window(x_window_b, {S::Block_M});
            move_tile_window(y_window, {S::Block_M});
        }
    }


    CK_TILE_DEVICE void operator()(
        const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_a, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_b, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  amd_buffer_coherence_enum::slc>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_window_a = make_tile_window(x_m_n_a,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto x_window_b = make_tile_window(x_m_n_b,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto xa  = load_tile(x_window_a);
            const auto xb  = load_tile(x_window_b);
            auto y_compute = load_tile(y_window);

            constexpr auto spans = decltype(xa)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x           = ck_tile::type_convert<ComputeDataType>(xa[i_j_idx]);
                    const auto y           = ck_tile::type_convert<ComputeDataType>(xb[i_j_idx]);
                    ElementWiseOperation{}(y_compute(i_j_idx), x, y);
                });
            });

            store_tile(y_window, cast_tile<YDataType>(y_compute));
            move_tile_window(x_window_a, {0, S::Block_N});
            move_tile_window(x_window_b, {0, S::Block_N});
            move_tile_window(y_window, {0, S::Block_N});
        }
    }

    template <typename Dims>
    CK_TILE_DEVICE void operator()(
        const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, Dims lens, Dims strides) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_a, lens, strides, number<S::Vector_N>{}, number<1>{});

        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_b, lens, strides, number<S::Vector_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  amd_buffer_coherence_enum::slc>(
            p_y, lens, strides, number<S::Vector_N>{}, number<1>{});


        const auto x_m_n_a_t = Policy::template MakeXTransformation<Problem>(x_m_n_a);
        const auto x_m_n_b_t = Policy::template MakeXTransformation<Problem>(x_m_n_b);
        const auto y_m_n_t = Policy::template MakeXTransformation<Problem>(y_m_n);

        const auto iM = get_block_id() * S::Block_M;

        auto x_window_a = make_tile_window(x_m_n_a_t,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto x_window_b = make_tile_window(x_m_n_b_t,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n_t,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(lens.template at<lens.size() - 1>(), S::Block_N));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto xa  = load_tile(x_window_a);
            const auto xb  = load_tile(x_window_b);
            auto y_compute = load_tile(y_window);

            constexpr auto spans = decltype(xa)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x           = ck_tile::type_convert<ComputeDataType>(xa[i_j_idx]);
                    const auto y           = ck_tile::type_convert<ComputeDataType>(xb[i_j_idx]);
                    ElementWiseOperation{}(y_compute(i_j_idx), x, y);
                });
            });

            store_tile(y_window, cast_tile<YDataType>(y_compute));
            move_tile_window(x_window_a, {0, S::Block_N});
            move_tile_window(x_window_b, {0, S::Block_N});
            move_tile_window(y_window, {0, S::Block_N});
        }
    }
};

} // namespace ck_tile
