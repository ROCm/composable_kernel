// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct ElementwiseUnaryPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeInputDistribution()
    {
        // TODO: Y dim must have one dim that is not reduced
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Problem::IssuesPerRow,
                                                      Problem::WarpsPerBlock,
                                                      Problem::LanesPerRow,
                                                      Problem::VectorSize>>,
                                       tuple<sequence<1>, sequence<1>>,
                                       tuple<sequence<1>, sequence<2>>,
                                       sequence<1, 1>,
                                       sequence<0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOutputDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Problem::IssuesPerRow,
                                                      Problem::WarpsPerBlock,
                                                      Problem::LanesPerRow,
                                                      Problem::VectorSize>>,
                                       tuple<sequence<1>, sequence<1>>,
                                       tuple<sequence<1>, sequence<2>>,
                                       sequence<1, 1>,
                                       sequence<0, 3>>{});
    }
};
} // namespace ck_tile
