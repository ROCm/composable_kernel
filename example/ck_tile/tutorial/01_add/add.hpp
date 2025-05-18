// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

template <typename BlockWarps, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WarpTile,   // warp size, seq<M, N>
          typename Vector>     // contiguous pixels(vector size) along seq<M, N>
struct AddShape
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{}); // elements along M in one Block
    static constexpr index_t Block_N = BlockTile::at(number<1>{}); // elements along N in one Block

    static constexpr index_t Warp_M = WarpTile::at(number<0>{}); // elements along M in one Warp
    static constexpr index_t Warp_N = WarpTile::at(number<1>{}); // elements along N in one Warp

    static constexpr index_t Vector_M = Vector::at(number<0>{}); // elements along M in one Vector
    static constexpr index_t Vector_N = Vector::at(number<1>{}); // elements along N in one Vector

    static constexpr index_t WarpPerBlock_M =
        BlockWarps::at(number<0>{}); // num concurrent warps along M
    static constexpr index_t WarpPerBlock_N =
        BlockWarps::at(number<1>{}); // num concurrent warps along N

    static constexpr index_t ThreadPerWarp_M =
        Warp_M /
        Vector_M; // num threads along M in one Warp (ThreadPerWarp_M * ThreadPerWarp_N must be 64)
    static constexpr index_t ThreadPerWarp_N =
        Warp_N /
        Vector_N; // num threads along N in one Warp (ThreadPerWarp_M * ThreadPerWarp_N must be 64)

    static constexpr index_t Repeat_M =
        Block_M /
        (WarpPerBlock_M *
         Warp_M); // num of time a warp iterates along M to ensure the entire block is covered
    static constexpr index_t Repeat_N =
        Block_N /
        (WarpPerBlock_N *
         Warp_N); // num of time a warp iterates along N to ensure the entire block is covered

    static constexpr index_t BlockSize =
        warpSize *
        reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{}); // num of threads in one block
};

template <typename XDataType_, typename ComputeDataType_, typename YDataType_, typename BlockShape_>
struct AddProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;       // data type of input tensor
    using ComputeDataType = remove_cvref_t<ComputeDataType_>; // data type of compute tensor
    using YDataType       = remove_cvref_t<YDataType_>;       // data type of output tensor
    using BlockShape      = remove_cvref_t<BlockShape_>;      // block shapes and sizes
};

struct AddDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M,
                               S::WarpPerBlock_M,
                               S::ThreadPerWarp_M,
                               S::Vector_M>, // how many sub division is a block divided in
                      sequence<S::Repeat_N,
                               S::WarpPerBlock_N,
                               S::ThreadPerWarp_N,
                               S::Vector_N>>, // how many sub division is a block divided in
                tuple<sequence<1, 2>, sequence<1, 2>>, // What are the shapes of those sub divisions
                tuple<sequence<1, 1>, sequence<2, 2>>, // What are the shapes of those sub divisions
                sequence<1, 1, 2, 2>, // How much data does a thread work on and how many iterations
                                      // of warps are there
                sequence<0, 3, 0, 3>>{}); // How much data does a thread work on and how many
                                          // iterations of warps are there
    }
};

template <typename Problem_, typename Policy_ = AddDefaultPolicy>
struct Add
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    CK_TILE_DEVICE void operator()(
        const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_a,
            make_tuple(M, N),
            make_tuple(N, 1),
            number<S::Vector_N>{},
            number<1>{}); // raw data, shape of tensor, stride of tensor, lastGarunteedVectorLength,
                          // lastGarunteedVectorStride

        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_b, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  amd_buffer_coherence_enum::slc>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto iM = get_block_id() * S::Block_M; // origin of the block along

        auto x_window_a = make_tile_window(x_m_n_a,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto x_window_b = make_tile_window(x_m_n_b,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto xa  = load_tile(x_window_a);
            const auto xb  = load_tile(x_window_b);
            auto y_compute = load_tile(y_window);

            constexpr auto spans = decltype(xa)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x           = ck_tile::type_convert<ComputeDataType>(xa[i_j_idx]);
                    const auto y           = ck_tile::type_convert<ComputeDataType>(xb[i_j_idx]);
                    y_compute(i_j_idx)     = x + y;
                });
            });

            store_tile(y_window, cast_tile<YDataType>(y_compute));
            move_tile_window(x_window_a, {0, S::Block_N});
            move_tile_window(x_window_b, {0, S::Block_N});
            move_tile_window(y_window, {0, S::Block_N});
        }
    }
};

} // namespace ck_tile
