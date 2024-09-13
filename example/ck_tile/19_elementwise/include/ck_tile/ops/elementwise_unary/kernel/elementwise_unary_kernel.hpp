// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

struct ElementwiseUnaryHostArgs
{
    const void* p_input;
    void* p_output;
    uint64_t num_pixels;
};

template <typename Pipeline_>
struct ElementwiseUnaryKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = remove_cvref_t<typename Pipeline::Problem>;

    using InputType  = typename Problem::InputType;
    using OutputType = typename Problem::OutputType;

    struct ElementwiseUnaryKargs
    {
        const void* p_input;
        void* p_output;
        uint64_t num_pixels;
    };

    using Kargs = ElementwiseUnaryKargs;
    using Hargs = ElementwiseUnaryHostArgs;

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& h)
    {
        constexpr index_t issues_per_block =
            Problem::BlockSize * Problem::Chunks * Problem::VectorSize;
        index_t grids =
            static_cast<index_t>((h.num_pixels + issues_per_block - 1) / issues_per_block);
        return dim3(grids);
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_input    = h.p_input;
        k.p_output   = h.p_output;
        k.num_pixels = h.num_pixels;
        return k;
    }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::BlockSize; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        uint64_t block_base =
            static_cast<uint64_t>(blockIdx.x) * Problem::BlockSize * Problem::VectorSize;
        uint64_t pixels_rem = kargs.num_pixels - block_base;

        const auto input_window = [&]() {
            const InputType* p_input =
                reinterpret_cast<const InputType*>(kargs.p_input) + block_base;

            auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                p_input,
                make_tuple(static_cast<index_t>(pixels_rem)),
                number<Problem::VectorSize>{});

            return make_tile_window(
                tmp, make_tuple(number<Problem::BlockSize * Problem::VectorSize>{}), {0});
        }();

        auto output_window = [&]() {
            OutputType* p_output =
                reinterpret_cast<OutputType*>(kargs.p_output) + block_base;

            auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                p_output,
                make_tuple(static_cast<index_t>(pixels_rem)),
                number<Problem::VectorSize>{});

            return make_tile_window(
                tmp, make_tuple(number<Problem::BlockSize * Problem::VectorSize>{}), {0});
        }();

        index_t loop_stride =
            __builtin_amdgcn_readfirstlane(gridDim.x * Problem::BlockSize * Problem::VectorSize);

        Pipeline{}(input_window, output_window, loop_stride);
    }
};
} // namespace ck_tile
