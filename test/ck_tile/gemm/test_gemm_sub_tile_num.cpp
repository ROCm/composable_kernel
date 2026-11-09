// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "gtest/gtest.h"

#include <array>

namespace {

using ck_tile::index_t;

// GetDividingSubTileNum uses nothing from the derived policy.
struct SubTilePolicy : ck_tile::UniversalGemmBasePolicy<SubTilePolicy>
{
};

struct SubTileCase
{
    index_t requested;       // the register estimate's request: 1, 2 or 4
    index_t k_iter_per_warp; // warp GEMMs in the K tile
    index_t expected;
};

// The TDM and async default policies round their K sub-tile count with this helper, so that the
// count divides the warp GEMMs of the K tile. Host only: no device allocation or kernel launch.
TEST(GemmSubTileNum, RoundsTheRequestUpToADivisorOfTheWarpGemms)
{
    constexpr std::array<SubTileCase, 14> cases = {{
        {1, 3, 1}, // one sub-tile always divides
        {2, 3, 3}, // a 384-deep FP8 K tile of 128-deep warp tiles
        {4, 3, 3}, // capped at the warp GEMM count
        {2, 1, 1}, // a single warp GEMM
        {4, 2, 2}, // capped
        {2, 4, 2}, // the request divides: unchanged
        {2, 8, 2},
        {4, 8, 4},
        {2, 6, 2},
        {4, 6, 6}, // 4 and 5 do not divide 6
        {4, 5, 5}, // a prime warp GEMM count
        {2, 9, 3},
        {4, 9, 9}, // 4 to 8 do not divide 9
        {0, 3, 1}, // a request below 1 counts as 1
    }};
    for(const auto& c : cases)
    {
        EXPECT_EQ(SubTilePolicy::GetDividingSubTileNum(c.requested, c.k_iter_per_warp), c.expected)
            << "requested " << c.requested << ", warp GEMMs per K tile " << c.k_iter_per_warp;
    }
}

} // namespace
