// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockTile_, typename WavePerBlock_, typename WaveTile_>
struct FlatmmShape
{
    using BlockTile    = remove_cvref_t<BlockTile_>;
    using WavePerBlock = remove_cvref_t<WavePerBlock_>;
    using WaveTile     = remove_cvref_t<WaveTile_>;

    static constexpr index_t NumWaves =
        reduce_on_sequence(WavePerBlock_{}, multiplies{}, number<1>{});
    static constexpr index_t BlockSize = warpSize * NumWaves;

    static constexpr index_t Block_M        = BlockTile::at(number<0>{});
    static constexpr index_t Block_N        = BlockTile::at(number<1>{});
    static constexpr index_t Block_K        = BlockTile::at(number<2>{});
    static constexpr index_t WavePerBlock_M = WavePerBlock::at(number<0>{});
    static constexpr index_t WavePerBlock_N = WavePerBlock::at(number<1>{});
    static constexpr index_t WavePerBlock_K = WavePerBlock::at(number<2>{});
    static constexpr index_t Wave_M         = WaveTile::at(number<0>{});
    static constexpr index_t Wave_N         = WaveTile::at(number<1>{});
    static constexpr index_t Wave_K         = WaveTile::at(number<2>{});

    static constexpr index_t ThreadPerBlock_M = Wave_M * WavePerBlock_M;
    static constexpr index_t ThreadPerBlock_N = Wave_N * WavePerBlock_N;
    static constexpr index_t ThreadPerBlock_K = Wave_K * WavePerBlock_K;
    static constexpr index_t Repeat_M         = Block_M / ThreadPerBlock_M;
    static constexpr index_t Repeat_N         = Block_N / ThreadPerBlock_N;
    static constexpr index_t Repeat_K         = Block_K / ThreadPerBlock_K;
    static constexpr index_t Block_W          = Wave_N * Wave_K;
    static constexpr index_t Block_Mr         = Block_M / Wave_M;
    static constexpr index_t Block_Nr         = Block_N / Wave_N;
    static constexpr index_t Block_Kr         = Block_K / Wave_K;

    static_assert(Block_M % ThreadPerBlock_M == 0);
    static_assert(Block_N % ThreadPerBlock_N == 0);
    static_assert(Block_K % ThreadPerBlock_K == 0);
};
} // namespace ck_tile
