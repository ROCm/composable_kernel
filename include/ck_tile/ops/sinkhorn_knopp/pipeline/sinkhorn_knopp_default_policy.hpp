// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"

namespace ck_tile {

struct SinkhornKnoppDefaultPolicy : public Reduce2dDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeInputBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<S::Block_M, 1>, sequence<1, S::Block_N>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<0, 0>>,
                                       sequence<2, 1>,
                                       sequence<1, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeTransposedInputBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<1, S::Block_N>, sequence<S::Block_M, 1>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<0, 0>>,
                                       sequence<2, 1>,
                                       sequence<1, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeLogUBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(tile_distribution_encoding<sequence<>,
                                                                        tuple<sequence<S::Block_M>>,
                                                                        tuple<sequence<1>>,
                                                                        tuple<sequence<0>>,
                                                                        sequence<1>,
                                                                        sequence<0>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeLogVBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(tile_distribution_encoding<sequence<>,
                                                                        tuple<sequence<S::Block_N>>,
                                                                        tuple<sequence<1>>,
                                                                        tuple<sequence<0>>,
                                                                        sequence<1>,
                                                                        sequence<0>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // For the LSE implementation, both log_u and log_v are stored in LDS,
        // requiring M + N elements
        using S = typename Problem::BlockShape;

        return (S::Block_M + S::Block_N) * sizeof(typename Problem::ComputeDataType);
    }
};

} // namespace ck_tile
