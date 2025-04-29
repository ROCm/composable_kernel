// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

namespace ck_tile {

struct BlockGemmARegBSmemCRegV1K8Policy
{
    template <typename Problem, index_t kM0>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        if constexpr(kM0 == 64)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution{}, 4, 1);
        }
        else if constexpr(kM0 == 32)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution{}, 2, 1);
        }
        else if constexpr(kM0 == 128)
        {
#if !defined(TOY_FA_FWD_QK_SWIZZLE)
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution{}, 4, 1);
#else
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{}, 4, 1);
#endif
        }
        else
        {
            static_assert(false, "Unsupported configuration for warp execution.");
        }
    }
};

} // namespace ck_tile
