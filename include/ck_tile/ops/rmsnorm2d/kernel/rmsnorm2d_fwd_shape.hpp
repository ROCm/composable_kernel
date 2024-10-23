// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
/*
// clang-format off

4-level descriptor: BlockTile-> WarpPerBlock-> WarpTile-> Vector

                         Block_N (Warp_N * WarpPerBlock_N * Repeat_N )
        +<----------------------< Repeat_N(2)>--------------------->+
        |                                                           |
        +<--    <WarpPerBlock_N(2)>  -->+
            Warp_N
        +--------------+--------------+--------------+--------------+----+----------------+
 Warp_M | wrap_0       | wrap_1       |                             |    ^                ^
        +--------------+--------------+                             |   <WarpPerBlock_M(2)> |
        | wrap_2       | wrap_3       |                             |    v
        +--------------+--------------+--------------+--------------+----+           Block_M
        |                             |                             |
        +                             +                             |
        |                             |                             |                     v
        +--------------+--------------+--------------+--------------+                     +

        each Warp-tile (e.g 16 thrd per row)

         Vector_N (contiguous pixels each thrd holds along N, or vector size)
        +-----------+-----------+-----------+-----------+-----------+
        | thrd_0    | thrd_1    | thrd_2    | thrd_3    | ...         Vector_M
        +-----------+-----------+-----------+-----------+-----------+
        | thrd_16   | thrd_17   | thrd_18   | thrd_19   | ...
        +-----------+-----------+-----------+-----------+-----------+
// clang-format on
*/
template <typename BlockTile_,    // block size, seq<M, N>
          typename WarpPerBlock_, // num warps along seq<M, N>
          typename WarpTile_,     // warp size, seq<M, N>
          typename Vector_,       // contiguous pixels(vector size) along seq<M, N>
          index_t BlockSize_ =
              warpSize* reduce_on_sequence(WarpPerBlock_{}, multiplies{}, number<1>{})>
struct Rmsnorm2dShape
{
    // block size
    static constexpr index_t Block_M = BlockTile_::at(number<0>{});
    static constexpr index_t Block_N = BlockTile_::at(number<1>{});

    // num warps along seq<M, N>, within each block
    static constexpr index_t WarpPerBlock_M = WarpPerBlock_::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = WarpPerBlock_::at(number<1>{});

    // warp size
    static constexpr index_t Warp_M = WarpTile_::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile_::at(number<1>{});

    static_assert(Block_M % (WarpPerBlock_M * Warp_M) == 0);
    static_assert(Block_N % (WarpPerBlock_N * Warp_N) == 0);
    // repeat of each thread along seq<M, N>
    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);

    // vector size along seq<M, N>
    static constexpr index_t Vector_M = Vector_::at(number<0>{});
    static constexpr index_t Vector_N = Vector_::at(number<1>{});

    static_assert(Warp_M % Vector_M == 0);
    static_assert(Warp_N % Vector_N == 0);
    // num of threads along seq<M, N>, within each warp
    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / Vector_N;

    static constexpr index_t BlockSize = BlockSize_;
};

} // namespace ck_tile
