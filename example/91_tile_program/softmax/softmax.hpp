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

    __device__ void
    MultiPassSoftmax(const ADataType* p_a, BDataType* p_b, ck::index_t M, ck::index_t N) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // A DRAM tensor view
        const auto a_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        const auto iM = get_block_id() * kMPerBlock;

        // A DRAM window
        auto a_dram_window =
            make_tile_window(a_dram,
                             make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}),
                             {iM, 0},
                             MakeABlockTileDistribution());

        // m = rowmax(A)
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };

        auto m = decltype(block_tile_reduce<AccDataType>(
            load_tile(a_dram_window), Sequence<1>{}, f_max, ADataType{})){};

        tile_elementwise_inout(
            [&](auto& e) { e = type_convert<AccDataType>(NumericLimits<ADataType>::Lowest()); }, m);

        index_t iN = 0;

        do
        {
            // load A tile from DRAM
            const auto a = load_tile(a_dram_window);

            // m = rowmax(A)
            block_tile_reduce(m, a, Sequence<1>{}, f_max);

            move_tile_window(a_dram_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);

        // cross lane reduce: max
        block_tile_reduce_sync(m, f_max);

        // reset window location
        iN = 0;
        move_tile_window(a_dram_window, {0, -N});

        // l = rowsum(exp(A - m))
        auto l = make_static_distributed_tensor<AccDataType>(m.GetTileDistribution());

        tile_elementwise_inout([&](auto& e) { e = 0; }, l);

        do
        {
            // load A tile from DRAM
            const auto a = load_tile(a_dram_window);

            constexpr auto a_spans = decltype(a)::GetDistributedSpans();

            sweep_tile_span(a_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                sweep_tile_span(a_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    // l = rowsum(exp(A - m))
                    l(i_idx) += math::exp(a[i_j_idx] - m[i_idx]);
                });
            });

            move_tile_window(a_dram_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);

        // cross lane reduce: sum
        block_tile_reduce_sync(l, [](auto e0, auto e1) { return e0 + e1; });

        // reset window location
        iN = 0;
        move_tile_window(a_dram_window, {0, -N});

        // B DRAM tensor view
        const auto b_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        // B DRAM window
        auto b_dram_window = make_tile_window(
            b_dram, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0});

        // B = exp(A - m) / l
        do
        {
            // load A tile from DRAM
            const auto a = load_tile(a_dram_window);

            constexpr auto a_spans = decltype(a)::GetDistributedSpans();

            auto b = make_static_distributed_tensor<BDataType>(a.GetTileDistribution());

            sweep_tile_span(a_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                sweep_tile_span(a_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    // B = exp(A - m) / l
                    b(i_j_idx) =
                        type_convert<BDataType>(math::exp(a[i_j_idx] - m[i_idx]) / l[i_idx]);
                });
            });

            // store B tile
            store_tile(b_dram_window, b);

            move_tile_window(a_dram_window, {0, kNPerBlock});
            move_tile_window(b_dram_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);
    }

    __device__ void
    SinglePassSoftmax(const ADataType* p_a, BDataType* p_b, ck::index_t M, ck::index_t N) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // A DRAM tensor view
        const auto a_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        const auto iM = get_block_id() * kMPerBlock;

        // A DRAM window
        auto a_dram_window =
            make_tile_window(a_dram,
                             make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}),
                             {iM, 0},
                             MakeABlockTileDistribution());

        // f_max
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };

        // m = rowmax(A)
        auto m = decltype(block_tile_reduce<AccDataType>(
            load_tile(a_dram_window), Sequence<1>{}, f_max, ADataType{})){};

        tile_elementwise_inout(
            [&](auto& e) { e = type_convert<AccDataType>(NumericLimits<ADataType>::Lowest()); }, m);

        // l = rowsum(exp(A - m))
        auto l = make_static_distributed_tensor<AccDataType>(m.GetTileDistribution());

        tile_elementwise_inout([&](auto& e) { e = 0; }, l);

        // load A tile from DRAM
        const auto a = load_tile(a_dram_window);

        constexpr auto a_spans = decltype(a)::GetDistributedSpans();

        // m = rowmax(A)
        block_tile_reduce(m, a, Sequence<1>{}, f_max);

        // cross lane reduce: max
        block_tile_reduce_sync(m, f_max);

        // l = rowsum(exp(A - m))
        sweep_tile_span(a_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            sweep_tile_span(a_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                l(i_idx) += math::exp(a[i_j_idx] - m[i_idx]);
            });
        });

        // cross lane reduce: sum
        block_tile_reduce_sync(l, [](auto e0, auto e1) { return e0 + e1; });

        auto b = make_static_distributed_tensor<BDataType>(a.GetTileDistribution());

        // B = exp(A - m) / l
        sweep_tile_span(a_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            sweep_tile_span(a_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                b(i_j_idx) = type_convert<BDataType>(math::exp(a[i_j_idx] - m[i_idx]) / l[i_idx]);
            });
        });

        // B DRAM tensor view
        const auto b_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        // B DRAM window
        auto b_dram_window = make_tile_window(
            b_dram, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, 0});

        // store B tile
        store_tile(b_dram_window, b);
    }

    __device__ void
    operator()(const ADataType* p_a, BDataType* p_b, ck::index_t M, ck::index_t N) const
    {
        if(N > kNPerBlock)
        {
            MultiPassSoftmax(p_a, p_b, M, N);
        }
        else
        {
            SinglePassSoftmax(p_a, p_b, M, N);
        }
    }
};
