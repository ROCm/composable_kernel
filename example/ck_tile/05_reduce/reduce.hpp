// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename ADataType,
          typename AccDataType,
          typename BDataType,
          index_t kBlockSize,
          typename BlockWarps, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WarpTile,   // warp size, seq<M, N>
          typename ThreadTile> // contiguous pixels(vector size) along seq<M, N>
struct Reduce
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile::at(number<1>{});

    static constexpr index_t Thread_M = ThreadTile::at(number<0>{});
    static constexpr index_t Thread_N = ThreadTile::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = BlockWarps::at(number<1>{});

    static constexpr index_t ThreadPerWarp_M = Warp_M / Thread_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / Thread_N;

    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);

    __device__ static constexpr auto MakeABlockTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Repeat_M, WarpPerBlock_M, ThreadPerWarp_M, Thread_M>,
                      sequence<Repeat_N, WarpPerBlock_N, ThreadPerWarp_N, Thread_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    __device__ void operator()(const ADataType* p_a, BDataType* p_b, index_t M, index_t N) const
    {
        const auto a_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_a, make_tuple(M, N), make_tuple(N, 1), number<Thread_N>{}, number<1>{});

        const auto iM = get_block_id() * Block_M;

        // A window
        auto a_block_window = make_tile_window(a_m_n,
                                               make_tuple(number<Block_M>{}, number<Block_N>{}),
                                               {iM, 0},
                                               MakeABlockTileDistribution());

        const auto f_reduce = [](const auto& v0, const auto& v1) { return v0 + v1; };

        const ADataType reduce_init_value = 0;

        constexpr auto reduce_dims = sequence<1>{};

        // Acc tile
        // TODO: support cross warp reduction
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

            move_tile_window(a_block_window, {0, Block_N});

            iN += Block_N;

        } while(iN < N);

        // FIXME: support cross warp reduction
        block_tile_reduce_sync(acc_block_tensor, f_reduce);

        // convert acc_block_tensor to b_block_tensor
        const auto b_block_tensor = tile_elementwise_in(
            [](const auto& acc) { return type_convert<BDataType>(acc); }, acc_block_tensor);

        // B
        const auto b_m = make_naive_tensor_view_packed<address_space_enum::global>(
            p_b, make_tuple(M), number<32>{});

        // B window
        auto b_block_window = make_tile_window(b_m, make_tuple(number<Block_M>{}), {iM});

        // store B tile
        store_tile(b_block_window, b_block_tensor);
    }
};

} // namespace ck_tile
