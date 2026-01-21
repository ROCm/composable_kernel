// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// template <typename XDataType, typename YDataType>
// struct SinkhornKnoppArgs
// {
//     YDataType* out;
//     const XDataType* p_x;
//     const index_t input_m;
//     int max_iterations;
// };

struct SinkhornKnoppArgs
{
    void* out;
    const void* p_x;
    const index_t input_m;
    int max_iterations;
};

struct SinkhornKnoppKernelReduce
{
    template <typename Problem>
    CK_TILE_DEVICE void operator()([[maybe_unused]] const SinkhornKnoppArgs& args) const
    {
        // // Creating tensor descriptors, views and windows for inputs and outputs

        // // Create the reduce ops
        // // * Reduce Op ADD for row and column sums
        // // * Elementwise Op EXP for exponentiation

        // using ExponentiationOp = ElementwiseOp<ExponentiationOperation>;
        // using AddOp            = ElementwiseOp<AddOperation>;
        // using DivideOp         = ElementwiseOp<DivideOperation>;

        // using ReduceOp = ReduceOp<AddOp, AddOp>;
        // // Run the first steps iteration of the Sinkhorn-Knopp algorithm
        // // Exponentiate the matrix x
        // auto x = load_tile(...);

        // // Hot loop for Sinkhorn-Knopp iterations from 1 to max_iterations
        // // Use BlockReduce2D for row and column sums
        // for(int i = 0; i <= args.max_iterations; i++)
        // {
        //     // 0. LOAD x
        //     // 1. Compute row sums (REDUCE)
        //     // 2. Divide values by row sums (SWEEP)
        //     // 3. STORE the result of the division (in transposed format)
        //     // 4. LOAD transposed x
        //     // 5. Compute column sums (REDUCE)
        //     // 6. Divide values by column sums (SWEEP)
        //     // 7. STORE the result of the division (in transposed format)
        // }
    }
};

template <typename Problem, typename Policy>
struct SinkhornKnoppKernelDummyNonStochastic
{

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    template <typename XDistributedTensor_>
    CK_TILE_DEVICE static auto MakeComputeBlockTile()
    {
        constexpr auto dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                XDistributedTensor_::get_tile_distribution()
                    .get_static_tile_distribution_encoding(),
                sequence<0>{}));

        auto tensor = make_static_distributed_tensor<typename Problem::YDataType>(dstr);

        return tensor;
    }

    // template <typename XDistributedTensor_>
    // CK_TILE_DEVICE static auto MakeYBlockTile()
    // {
    //     constexpr auto dstr =
    //         make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
    //             XDistributedTensor_::get_tile_distribution()
    //                 .get_static_tile_distribution_encoding(),
    //             sequence<0>{}));

    //     auto tensor = make_static_distributed_tensor<typename Problem::YDataType>(dstr);

    //     return tensor;
    // }

    CK_TILE_DEVICE void operator()([[maybe_unused]] const SinkhornKnoppArgs& args) const
    {
        using S         = Problem::BlockShape;
        using XDataType = typename Problem::XDataType;
        using YDataType = typename Problem::YDataType;

        static_assert(S::Block_M == S::Block_N, "Input must be a square matrix!");

        auto* p_x        = static_cast<const Problem::XDataType*>(args.p_x);
        auto* p_y        = static_cast<Problem::YDataType*>(args.out);
        auto reduce_func = ck_tile::ReduceOp::Add{};

        const XDataType custom_padding_value = type_convert<XDataType>(
            reduce_func.GetIdentityValue<typename Problem::ComputeDataType>());

        const auto x_desc = make_naive_tensor_descriptor(make_tuple(args.input_m, args.input_m),
                                                         make_tuple(args.input_m, 1),
                                                         number<4>{}, // TODO: Hardcoded
                                                         // vectorization, //we should calculate it!
                                                         number<1>{});

        auto buffer_view = make_buffer_view<address_space_enum::global>(
            p_x, x_desc.get_element_space_size(), custom_padding_value);

        const auto x_tensor =
            tensor_view<decltype(buffer_view), decltype(x_desc)>{buffer_view, x_desc};

        [[maybe_unused]] auto x_window =
            make_tile_window(x_tensor,
                             make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                             {0, 0},
                             Policy::template MakeXBlockTileDistribution<Problem>());

        auto out_buffer_view = make_buffer_view<address_space_enum::global>(
            p_y, x_desc.get_element_space_size(), type_convert<YDataType>(custom_padding_value));

        auto y_tensor =
            tensor_view<decltype(out_buffer_view), decltype(x_desc)>{out_buffer_view, x_desc};

        [[maybe_unused]] auto y_window =
            make_tile_window(y_tensor,
                             make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                             {0, 0},
                             Policy::template MakeXBlockTileDistribution<Problem>());
        // Dummy copy from input to output

        [[maybe_unused]] auto x_tile = load_tile(x_window);

        // auto y_tile = MakeYBlockTile<decltype(x_window)>();
        auto y_tile = make_static_distributed_tensor<YDataType>(
            Policy::template MakeXBlockTileDistribution<Problem>());

        // Set all output elements to the custom padding value.
        // // Simple solution to set the whole tile to a constant //
        // set_tile(y_tile, custom_padding_value);
        // store_tile(y_window, y_tile);

        constexpr auto y_spans = y_tile.get_distributed_spans();
        sweep_tile_span(y_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(y_spans[number<1>{}], [&](auto idx1) {
                constexpr auto distributed_indices = make_tuple(idx0, idx1);
                y_tile(distributed_indices)        = type_convert<YDataType>(custom_padding_value);
            });
        });

        store_tile(y_window, y_tile);
    }
};

} // namespace ck_tile
