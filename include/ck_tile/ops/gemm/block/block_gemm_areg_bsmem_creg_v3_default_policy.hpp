// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

// Default policy for BlockGemmARegBSmemCRegV3
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmARegBSmemCRegNonTransposedV3DefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{}, 4, 1);
    }
};
struct BlockGemmARegBSmemCRegTransposedV3Policy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemmMfmaF16F16F32M32N32K8{}, 4, 1);
    }
};

} // namespace ck_tile
