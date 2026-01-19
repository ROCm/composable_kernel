// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

namespace ck_tile {

template <WarpPerBlock_M,
          WarpPerBlock_N,
          ThreadPerWarp_M,
          ThreadPerWarp_N,
          ThreadTile_M,
          ThreadTile_N,
          Repeat_M,
          Repeat_N>
struct SinkHornKnoppShape
{
    static constexpr index_t Block_M         = WarpPerBlock_M;
    static constexpr index_t Block_N         = WarpPerBlock_N;
    static constexpr index_t ThreadPerWarp_M = ThreadPerWarp_M;
    static constexpr index_t ThreadPerWarp_N = ThreadPerWarp_N;
    static constexpr index_t ThreadTile_M    = ThreadTile_M;
    static constexpr index_t ThreadTile_N    = ThreadTile_N;
    static constexpr index_t Repeat_M        = Repeat_M;
    static constexpr index_t Repeat_N        = Repeat_N;
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
