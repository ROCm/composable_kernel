// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

namespace ck_tile {

// template <WarpPerBlock_M,
//           WarpPerBlock_N,
//           ThreadPerWarp_M,
//           ThreadPerWarp_N,
//           ThreadTile_M,
//           ThreadTile_N,
//           Repeat_M,
//           Repeat_N>
// struct SinkHornKnoppShape
// {
//     static constexpr index_t Block_M         = WarpPerBlock_M;
//     static constexpr index_t Block_N         = WarpPerBlock_N;
//     static constexpr index_t ThreadPerWarp_M = ThreadPerWarp_M;
//     static constexpr index_t ThreadPerWarp_N = ThreadPerWarp_N;
//     static constexpr index_t ThreadTile_M    = ThreadTile_M;
//     static constexpr index_t ThreadTile_N    = ThreadTile_N;
//     static constexpr index_t Repeat_M        = Repeat_M;
//     static constexpr index_t Repeat_N        = Repeat_N;
// };


template <typename BlockWarps, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WarpTile,   // warp size, seq<M, N>
          typename ThreadTile> // contiguous pixels(vector size) along seq<M, N>
struct SinkhornKnoppShape
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile::at(number<1>{});

    static constexpr index_t ThreadTile_M = ThreadTile::at(number<0>{});
    static constexpr index_t ThreadTile_N = ThreadTile::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = BlockWarps::at(number<1>{});

    static constexpr index_t RepeatInWarp =
        Warp_M * Warp_N / ThreadTile_M / ThreadTile_N / ck_tile::get_warp_size();
    static constexpr index_t RepeatInWarp_M =
        (Warp_M / ThreadTile_M > Warp_N / ThreadTile_N) ? RepeatInWarp : 1;
    static constexpr index_t RepeatInWarp_N =
        (Warp_M / ThreadTile_M > Warp_N / ThreadTile_N) ? 1 : RepeatInWarp;

    static constexpr index_t ThreadPerWarp_M = Warp_M / ThreadTile_M / RepeatInWarp_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / ThreadTile_N / RepeatInWarp_N;

    static constexpr index_t Repeat_M = Block_M * RepeatInWarp_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N * RepeatInWarp_N / (WarpPerBlock_N * Warp_N);

    // static constexpr index_t BlockSize = ck_tile::get_warp_size();
    static constexpr index_t BlockSize = 1; // TODO
};

template <typename _XDataType,
          typename _YDataType,
          typename _BlockShape,
          typename _ComputeDataType = float>
struct SinkhornKnoppProblem
{
    using XDataType       = remove_cvref_t<_XDataType>;
    using ComputeDataType = remove_cvref_t<_ComputeDataType>;
    using YDataType       = remove_cvref_t<_YDataType>;

    using BlockShape = remove_cvref_t<_BlockShape>;
};

} // namespace ck_tile
