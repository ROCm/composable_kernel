// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/norm_reduce/block/block_norm_reduce_problem.hpp"
#include "ck_tile/ops/norm_reduce/block/block_norm_reduce.hpp"

namespace ck_tile {
struct ElementWiseDefaultPolicy1D
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>, // Replicate
                                       tuple<sequence<S::Repeat_M,
                                                      S::WarpPerBlock_M,
                                                      S::ThreadPerWarp_M,
                                                      S::Vector_M>>,    // Hierarchical
                                       tuple<sequence<1>, sequence<1>>, // Parallel
                                       tuple<sequence<1>, sequence<2>>, // Parallel
                                       sequence<1, 1>,                  // Yield
                                       sequence<0, 3>>{}                // Yield
        );
    }
};

struct ElementWiseDefaultPolicy2D
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }


    template <typename Problem, typename TensorView>
    CK_TILE_DEVICE static constexpr auto MakeXTransformation(TensorView view)
    {
        // using S = typename Problem::BlockShape;
        return transform_tensor_view(view,
            make_tuple(make_merge_transform(make_tuple(number<8>{}, number<4096>{})),
                       make_pass_through_transform(number<4096>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }
};
}
