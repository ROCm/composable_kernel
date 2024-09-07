// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

struct TopkSoftmaxHostArgs
{
    const void* p_input;
    void* p_output;
    void* p_indices;
    index_t num_rows;
    index_t num_experts;
    index_t topk;
};

template <typename Pipeline_>
struct TopkSoftmaxKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = remove_cvref_t<typename Pipeline::Problem>;

    using InputType  = typename Problem::InputType;
    using WeightType = typename Problem::WeightType;
    using IndexType  = typename Problem::IndexType;

    struct TopkSoftmaxKargs
    {
        const void* p_input;
        void* p_output;
        void* p_indices;
        index_t num_rows;
        index_t num_experts;
        index_t topk;
    };

    using Kargs = TopkSoftmaxKargs;
    using Hargs = TopkSoftmaxHostArgs;

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& h)
    {
        const int num_warps  = (h.num_rows + Problem::RowsPerWarp - 1) / Problem::RowsPerWarp;
        const int num_blocks = (num_warps + Problem::WarpsPerBlock - 1) / Problem::WarpsPerBlock;

        return dim3(num_blocks);
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_input     = h.p_input;
        k.p_output    = h.p_output;
        k.p_indices   = h.p_indices;
        k.num_rows    = h.num_rows;
        k.num_experts = h.num_experts;
        k.topk        = h.topk;
        return k;
    }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::BlockSize; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        index_t block_row_id    = static_cast<index_t>(blockIdx.x * Problem::RowsPerBlock);
        const auto input_window = [&]() {
            const InputType* p_input = reinterpret_cast<const InputType*>(kargs.p_input) +
                                       block_row_id * kargs.num_experts;

            auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                p_input,
                make_tuple(kargs.num_rows, kargs.num_experts),
                number<Problem::VectorSize>{});
            auto view = pad_tensor_view(
                tmp,
                make_tuple(number<Problem::RowsPerBlock>{}, number<Problem::Experts>{}),
                sequence<1, 1>{});

            return make_tile_window(
                view,
                make_tuple(number<Problem::RowsPerBlock>{}, number<Problem::Experts>{}),
                {0, 0});
        }();

        auto output_window = [&]() {
            WeightType* p_output =
                reinterpret_cast<WeightType*>(kargs.p_output) + block_row_id * kargs.topk;
            auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                p_output, make_tuple(kargs.num_rows, kargs.topk), number<Problem::VectorSize>{});
            auto view = pad_tensor_view(
                tmp, make_tuple(number<Problem::RowsPerBlock>{}, number<1>{}), sequence<1, 0>{});

            return make_tile_window(
                view, make_tuple(number<Problem::RowsPerBlock>{}, number<1>{}), {0, 0});
        }();

        auto indices_window = [&]() {
            IndexType* p_indices =
                reinterpret_cast<IndexType*>(kargs.p_indices) + block_row_id * kargs.topk;
            auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                p_indices, make_tuple(kargs.num_rows, kargs.topk), number<Problem::VectorSize>{});
            auto view = pad_tensor_view(
                tmp, make_tuple(number<Problem::RowsPerBlock>{}, number<1>{}), sequence<1, 0>{});
            return make_tile_window(
                view, make_tuple(number<Problem::RowsPerBlock>{}, number<1>{}), {0, 0});
        }();

        Pipeline{}(input_window, output_window, indices_window, kargs.topk, kargs.num_experts);
    }
};
} // namespace ck_tile
