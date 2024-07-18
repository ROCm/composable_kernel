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
struct Reduce
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

        const auto a_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        const auto iM = get_block_id() * kMPerBlock;

        // A window
        auto a_block_window =
            make_tile_window(a_m_n,
                             make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}),
                             {iM, 0},
                             MakeABlockTileDistribution());

        const auto f_reduce = [](const auto& v0, const auto& v1) { return v0 + v1; };

        const ADataType reduce_init_value = 0;

        constexpr auto reduce_dims = Sequence<1>{};

        // Acc tile
        // FIXME: support cross warp reduction
        auto acc_block_tensor = decltype(block_tile_reduce<AccDataType>(
            load_tile(a_block_window), reduce_dims, f_reduce, reduce_init_value)){};

        // init Acc tile
        tile_elementwise_inout(
            [&](auto& acc) { acc = type_convert<AccDataType>(reduce_init_value); },
            acc_block_tensor);

        // loop
        index_t iN = 0;

        do
        {
            const auto a_block_tensor = load_tile(a_block_window);

            // FIXME: support cross warp reduction
            block_tile_reduce(acc_block_tensor, a_block_tensor, reduce_dims, f_reduce);

            move_tile_window(a_block_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);

        // FIXME: support cross warp reduction
        block_tile_reduce_sync(acc_block_tensor, f_reduce);

        // convert acc_block_tensor to b_block_tensor
        const auto b_block_tensor = tile_elementwise_in(
            [](const auto& acc) { return type_convert<BDataType>(acc); }, acc_block_tensor);

        // B
        const auto b_m = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
            p_b, make_tuple(M), Number<32>{});

        // B window
        auto b_block_window = make_tile_window(b_m, make_tuple(Number<kMPerBlock>{}), {iM});

        // store B tile
        store_tile(b_block_window, b_block_tensor);
    }
};
