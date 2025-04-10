// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

namespace ck_tile {

struct BlockGemmARegBSmemCRegV1K8Policy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution{}, 4, 1);
    }
};

} // namespace ck_tile
