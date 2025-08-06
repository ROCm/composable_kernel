// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Problem: defines the nature of the data and the function to apply to the result
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_,
          typename CElementFunction_>
struct PracticeGemmProblem
{
    using ADataType        = ADataType_;
    using BDataType        = BDataType_;
    using CDataType        = CDataType_;
    using AccDataType      = AccDataType_;
    using CElementFunction = CElementFunction_;

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "practice_gemm_problem",
                      concat('x', ADataType::GetName(), BDataType::GetName(), CDataType::GetName(), AccDataType::GetName()));
        // clang-format on
    }
};

template <typename BlockTile_, typename WaveTile_>
struct PracticeGemmShape
{
    using BlockTile = remove_cvref_t<BlockTile_>;
    using WaveTile  = remove_cvref_t<WaveTile_>;

    static constexpr index_t BlockTile_M = BlockTile::at(number<0>{});
    static constexpr index_t BlockTile_N = BlockTile::at(number<1>{});
    static constexpr index_t BlockTile_K = BlockTile::at(number<2>{});

    static constexpr index_t WaveTile_M = WaveTile::at(number<0>{});
    static constexpr index_t WaveTile_N = WaveTile::at(number<1>{});
    static constexpr index_t WaveTile_K = WaveTile::at(number<2>{});

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "practice_gemm_shape",
                      concat('x', BlockTile_M, BlockTile_N, BlockTile_K),
                      concat('x', WaveTile_M, WaveTile_N, WaveTile_K));
        // clang-format on
    }
};

} // namespace ck_tile
