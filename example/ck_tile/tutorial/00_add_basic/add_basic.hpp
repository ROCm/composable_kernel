// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

// struct that holds the tile size of the block, warp, and vector
// and the number of warps per block
// and the number of threads per warp
// and the number of times the warp tile is repeated in the block tile
// and the block size
template <typename BlockWarps, typename BlockTile, typename WarpTile, typename Vector>
struct AddVectorShape
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});

    static constexpr index_t Vector_M = Vector::at(number<0>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});

    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;

    static constexpr index_t Repeat_M =
        Block_M /
        (WarpPerBlock_M * Warp_M); // Number of times the warp tile is repeated in the block tile

    static constexpr index_t BlockSize =
        warpSize * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
};

template <typename XDataType_, typename ComputeDataType_, typename YDataType_, typename BlockShape_>
struct AddVectorProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
};

// data mapping beween threads and memory
struct AddDefaultPolicy
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

template <typename Problem_, typename Policy_ = AddDefaultPolicy>
struct AddVectorKernel
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    // body of the kernel
    CK_TILE_DEVICE void
    operator()(const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, index_t M) const
    {
        using S = typename Problem::BlockShape;

        // create tensor view for the input and output data, this defines how the data is laid out
        // in memory
        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global>(
            p_x_a,
            make_tuple(M),
            make_tuple(1),
            number<S::Vector_M>{}); // raw pointer, shape of the tensor, stride of the tensor, and
                                    // lastGarunteedVectorLength

        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global>(
            p_x_b, make_tuple(M), make_tuple(1), number<S::Vector_M>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M), make_tuple(1), number<S::Vector_M>{});

        // origin of the block tile
        const auto iM = get_block_id() * S::Block_M;

        // creating tile windows for the input and output data
        auto x_window_a = make_tile_window(x_m_n_a,
                                           make_tuple(number<S::Block_M>{}),
                                           {iM},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto x_window_b = make_tile_window(x_m_n_b,
                                           make_tuple(number<S::Block_M>{}),
                                           {iM},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(number<S::Block_M>{}),
                                         {iM},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        // Load tile data
        const auto xa =
            load_tile(x_window_a); // load tile data from global tensor view, load from where? what?
                                   // how many? logical memory layout? all are defined in x_window_a
        const auto xb  = load_tile(x_window_b);
        auto y_compute = load_tile(y_window);

        // Process the vector add
        constexpr auto spans = decltype(xa)::get_distributed_spans(); // shape of the tile
        sweep_tile_span(spans[number<0>{}], [&](auto idx) {           // iterate over the tile
            const auto tile_idx = make_tuple(idx);
            const auto a_val    = type_convert<ComputeDataType>(xa[tile_idx]);
            const auto b_val    = type_convert<ComputeDataType>(xb[tile_idx]);
            y_compute(tile_idx) = a_val + b_val;

        });

        // Store results
        store_tile(y_window,
                   cast_tile<YDataType>(y_compute)); // store the result back to global tensor view
    }
};

} // namespace ck_tile
