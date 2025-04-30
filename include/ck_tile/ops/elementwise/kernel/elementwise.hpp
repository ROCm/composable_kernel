// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = ElementWiseDefaultPolicy>
struct ElementWiseKernel
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType            = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType      = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType            = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using ElementWiseOperation = ck_tile::remove_cvref_t<typename Problem::ElementWiseOperation>;

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
};

} // namespace ck_tile
