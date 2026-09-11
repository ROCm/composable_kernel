// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

// Default policy for BlockGemmARegBSmemCRegV2
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmARegBSmemCRegV2DefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {

        return make_tuple(WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{}, 4, 1);
    }
};

} // namespace ck_tile
