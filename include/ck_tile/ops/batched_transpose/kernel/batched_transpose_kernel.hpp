// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

struct BatchedTransposeHostArgs
{
    const void* p_input;
    void* p_output;
    index_t batch;
    index_t height;
    index_t width;
    index_t dim_stride;
    index_t dim_total;
    index_t magic_h;
    index_t shift_h;
    index_t magic_w;
    index_t shift_w;
    index_t dim_h;
    index_t dim_w;
};

template <typename Pipeline_>
struct BatchedTransposeKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = remove_cvref_t<typename Pipeline::Problem>;

    using Type = typename Problem::InputType;

    struct BatchedTransposeKargs
    {
        const void* p_input;
        void* p_output;
        index_t batch;
        index_t height;
        index_t width;
        index_t dim_stride;
        index_t dim_total;
        index_t magic_h;
        index_t shift_h;
        index_t magic_w;
        index_t shift_w;
        index_t dim_h;
        index_t dim_w;
    };

    using Kargs = BatchedTransposeKargs;
    using Hargs = BatchedTransposeHostArgs;

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& h)
    {
        size_t grid_size = h.batch * h.dim_h * h.dim_w;
        return dim3(grid_size);
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_input    = h.p_input;
        k.p_output   = h.p_output;
        k.batch      = h.batch;
        k.height     = h.height;
        k.width      = h.width;
        k.dim_stride = h.dim_stride;
        k.dim_total  = h.dim_total;
        k.magic_h    = h.magic_h;
        k.shift_h    = h.shift_h;
        k.magic_w    = h.magic_w;
        k.shift_w    = h.shift_w;
        k.dim_h      = h.dim_h;
        k.dim_w      = h.dim_w;
        return k;
    }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::kBlockSize; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        printf("in kernel. k.height:%d k.width:%d\n", kargs.height, kargs.width);

        static constexpr ck_tile::index_t kMPerBlock = Problem::kMPerBlock;
        static constexpr ck_tile::index_t kNPerBlock = Problem::kNPerBlock;
        static constexpr bool kPadM                  = Problem::kPadM;
        static constexpr bool kPadN                  = Problem::kPadN;

        static constexpr ck_tile::index_t kMPerThread = Problem::kMPerThread;
        static constexpr ck_tile::index_t kNPerThread = Problem::kNPerThread;

        const auto x_m_n = [&]() {
            const auto x_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const Type*>(kargs.p_input),
                make_tuple(kargs.height, kargs.width),
                make_tuple(kargs.width, 1),
                number<kNPerThread>{}, // TODO thread load value
                number<1>{});

            return pad_tensor_view(x_dram_naive,
                                   make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                   sequence<kPadM, kPadN>{});
        }();

        const auto iM = get_block_id() / kargs.dim_w;
        const auto iN = get_block_id() % kargs.dim_w;

        const auto y_n_m = [&]() {
            const auto y_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<Type*>(kargs.p_output),
                make_tuple(kargs.width, kargs.height),
                make_tuple(kargs.height, 1),
                number<kMPerThread>{},
                number<1>{});

            return pad_tensor_view(y_dram_naive,
                                   make_tuple(number<kNPerBlock>{}, number<kMPerBlock>{}),
                                   sequence<kPadN, kPadM>{});
        }();

        auto x_block_window =
            make_tile_window(x_m_n,
                             make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                             {iM * kMPerBlock, iN * kNPerBlock});
        auto y_block_window =
            make_tile_window(y_n_m,
                             make_tuple(number<kNPerBlock>{}, number<kMPerBlock>{}),
                             {iN * kNPerBlock, iM * kMPerBlock});

        Pipeline{}(x_block_window, y_block_window, kargs.batch, kargs.height, kargs.width, iM, iN);
    }
};
} // namespace ck_tile
