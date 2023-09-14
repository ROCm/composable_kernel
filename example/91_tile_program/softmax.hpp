// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"

template <typename ADataType,
          typename AccDataType,
          typename BDataType,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock>
struct Softmax
{
#if 0
     __device__ static constexpr auto MakeABlockTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        // 2x2 wave
        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<2, 2, 4, 2, 4>, Sequence<2, 2, 32>>,
                                           Tuple<Sequence<1, 2>, Sequence<1, 2>>,
                                           Tuple<Sequence<1, 1>, Sequence<3, 2>>,
                                           Sequence<1, 2, 1, 1>,
                                           Sequence<0, 0, 2, 4>>{});
    }
#elif 0
    __device__ static constexpr auto MakeABlockTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        // 2x2 wave
        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<2, 2, 32>, Sequence<2, 2, 4, 2, 4>>,
                                           Tuple<Sequence<2, 1>, Sequence<2, 1>>,
                                           Tuple<Sequence<1, 1>, Sequence<3, 2>>,
                                           Sequence<2, 1, 2, 2>,
                                           Sequence<0, 0, 2, 4>>{});
    }
#elif 1
    __device__ static constexpr auto MakeABlockTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        // 4x1 wave
        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<1, 4, 4, 2, 4>, Sequence<4, 1, 32>>,
                                           Tuple<Sequence<1, 2>, Sequence<1, 2>>,
                                           Tuple<Sequence<1, 1>, Sequence<3, 2>>,
                                           Sequence<1, 2, 1, 1>,
                                           Sequence<0, 0, 2, 4>>{});
    }
#endif

    __device__ void
    operator()(const ADataType* p_a, BDataType* p_b, ck::index_t M, ck::index_t N) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        const auto a_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        const auto iM = get_block_id() * kMPerBlock;

        // A window
        auto a_block_window =
            make_tile_window(a_m_n,
                             make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}),
                             {iM, 0},
                             MakeABlockTileDistribution());

        constexpr auto reduce_dims = Sequence<1>{};

        const auto f_max = [](auto v0, auto v1) { return max(v0, v1); };

        const ADataType max_reduce_init_value = NumericLimits<ADataType>::Lowest();

        // max = max(a)
        auto max_block_tensor = decltype(block_tile_reduce<AccDataType>(
            load_tile(a_block_window), reduce_dims, f_max, max_reduce_init_value)){};

        tile_elementwise_inout(
            [&](auto& max) { max = type_convert<AccDataType>(max_reduce_init_value); },
            max_block_tensor);

        index_t iN = 0;

        do
        {
            const auto a_block_tensor = load_tile(a_block_window);

            block_tile_reduce(max_block_tensor, a_block_tensor, reduce_dims, f_max);

            move_tile_window(a_block_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);

        // cross lane reduce: max
        block_tile_reduce_sync(max_block_tensor, f_max);

        // exp_sum = sum(exp(a - a_max))
        auto exp_sum_block_tensor =
            make_static_distributed_tensor<AccDataType>(max_block_tensor.GetTileDistribution());

        tile_elementwise_inout([&](auto& exp_sum) { exp_sum = 0; }, exp_sum_block_tensor);

        // reset window location
        iN = 0;
        move_tile_window(a_block_window, {0, -N});

        do
        {
            const auto a_block_tensor = load_tile(a_block_window);

            constexpr auto a_spans = decltype(a_block_tensor)::GetDistributedSpans();

            //
            sweep_tile_span(a_spans[I0], [&](auto idx0) {
                constexpr auto m_idx = make_tuple(idx0);

                const auto v_max = max_block_tensor.GetElementFromTileDistributedIndices(m_idx);

                AccDataType v_exp_sum =
                    exp_sum_block_tensor.GetElementFromTileDistributedIndices(m_idx);

                sweep_tile_span(a_spans[I1], [&](auto idx1) {
                    constexpr auto m_n_idx = make_tuple(idx0, idx1);

                    const auto v_a = a_block_tensor.GetElementFromTileDistributedIndices(m_n_idx);

                    (void)v_max;

                    // exp and sum
                    v_exp_sum += math::exp(v_a - v_max);
                });

                exp_sum_block_tensor.SetElementFromTileDistributedIndices(m_idx, v_exp_sum);
            });

            move_tile_window(a_block_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);

        // cross lane reduce: sum
        block_tile_reduce_sync(exp_sum_block_tensor, [](auto v0, auto v1) { return v0 + v1; });

        // B
        const auto b_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        // B window
        auto b_block_window = make_tile_window(
            b_m_n, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0});

        // reset window location
        iN = 0;
        move_tile_window(a_block_window, {0, -N});

        do
        {
            const auto a_block_tensor = load_tile(a_block_window);

            constexpr auto a_spans = decltype(a_block_tensor)::GetDistributedSpans();

            auto b_block_tensor =
                make_static_distributed_tensor<BDataType>(a_block_tensor.GetTileDistribution());

            //
            sweep_tile_span(a_spans[I0], [&](auto idx0) {
                constexpr auto m_idx = make_tuple(idx0);

                const auto v_max = max_block_tensor.GetElementFromTileDistributedIndices(m_idx);

                const auto v_exp_sum =
                    exp_sum_block_tensor.GetElementFromTileDistributedIndices(m_idx);

                sweep_tile_span(a_spans[I1], [&](auto idx1) {
                    constexpr auto m_n_idx = make_tuple(idx0, idx1);

                    const auto v_a = a_block_tensor.GetElementFromTileDistributedIndices(m_n_idx);

                    // exp
                    const BDataType v_b =
                        type_convert<BDataType>(math::exp(v_a - v_max) / v_exp_sum);

                    b_block_tensor.SetElementFromTileDistributedIndices(m_n_idx, v_b);
                });
            });

            // store B tile
            store_tile(b_block_window, b_block_tensor);

            move_tile_window(a_block_window, {0, kNPerBlock});
            move_tile_window(b_block_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);
    }
};
