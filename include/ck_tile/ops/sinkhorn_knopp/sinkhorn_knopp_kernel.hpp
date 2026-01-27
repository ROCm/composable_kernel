// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct SinkhornKnoppArgs
{
    void* p_out;
    const void* p_in;
    const index_t input_m;
    int max_iterations;
};

template <typename Problem, typename Policy>
struct SinkhornKnoppKernelReduce
{
    CK_TILE_DEVICE void operator()([[maybe_unused]] const SinkhornKnoppArgs& args) const
    {
        // // Creating tensor descriptors, views and windows for inputs and outputs

        using S           = Problem::BlockShape;
        using InDataType  = typename Problem::OutDataType;
        using OutDataType = typename Problem::OutDataType;

        static_assert(S::Block_M == S::Block_N, "Input must be a square matrix!");

        auto* p_in  = static_cast<const Problem::InDataType*>(args.p_in);
        auto* p_out = static_cast<Problem::OutDataType*>(args.p_out);

        [[maybe_unused]] auto exp_op = ck_tile::element_wise::Exp{};
        [[maybe_unused]] auto acc_op = ck_tile::ReduceOp::Add{};
        [[maybe_unused]] auto div_op = ck_tile::element_wise::UnaryDivide{};

        // We require exp(input) > 0, and exp(padding) == 0
        const InDataType x_padding_value = -ck_tile::numeric<InDataType>::infinity();

        const auto in_desc =
            make_naive_tensor_descriptor(make_tuple(args.input_m, args.input_m),
                                         make_tuple(args.input_m, 1),
                                         number<4>{}, // TODO: Hardcoded
                                         // vectorization, //we should calculate it!
                                         number<1>{});

        auto buffer_view = make_buffer_view<address_space_enum::global>(
            p_in, in_desc.get_element_space_size(), x_padding_value);

        const auto x_tensor =
            tensor_view<decltype(buffer_view), decltype(in_desc)>{buffer_view, in_desc};

        [[maybe_unused]] auto x_window =
            make_tile_window(x_tensor,
                             make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                             {0, 0},
                             Policy::template MakeXBlockTileDistribution<Problem>());

        const OutDataType y_padding_value = acc_op.template GetIdentityValue<OutDataType>();
        auto out_buffer_view              = make_buffer_view<address_space_enum::global>(
            p_out, in_desc.get_element_space_size(), y_padding_value);

        auto y_tensor =
            tensor_view<decltype(out_buffer_view), decltype(in_desc)>{out_buffer_view, in_desc};

        [[maybe_unused]] auto y_window =
            make_tile_window(y_tensor,
                             make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                             {0, 0},
                             Policy::template MakeXBlockTileDistribution<Problem>());

        [[maybe_unused]] auto c_window =
            make_null_tile_window(make_tuple(number<S::Block_M>{}, number<S::Block_N>{}));
        [[maybe_unused]] auto x_tile = load_tile(x_window);

        // // Run the first steps iteration of the Sinkhorn-Knopp algorithm
        // // Exponentiate the matrix x
        // elementwise()

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

        auto tensor = make_static_distributed_tensor<typename Problem::OutDataType>(dstr);

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

    //     auto tensor = make_static_distributed_tensor<typename Problem::OutDataType>(dstr);

    //     return tensor;
    // }

    CK_TILE_DEVICE void operator()([[maybe_unused]] const SinkhornKnoppArgs& args) const
    {
        using S           = Problem::BlockShape;
        using InDataType  = typename Problem::InDataType;
        using OutDataType = typename Problem::OutDataType;

        static_assert(S::Block_M == S::Block_N, "Input must be a square matrix!");

        auto* p_in       = static_cast<const Problem::InDataType*>(args.p_in);
        auto* p_out      = static_cast<Problem::OutDataType*>(args.p_out);
        auto reduce_func = ck_tile::ReduceOp::Add{};

        const InDataType custom_padding_value = type_convert<InDataType>(
            reduce_func.GetIdentityValue<typename Problem::ComputeDataType>());

        const auto in_desc =
            make_naive_tensor_descriptor(make_tuple(args.input_m, args.input_m),
                                         make_tuple(args.input_m, 1),
                                         number<4>{}, // TODO: Hardcoded
                                         // vectorization, //we should calculate it!
                                         number<1>{});

        auto buffer_view = make_buffer_view<address_space_enum::global>(
            p_in, in_desc.get_element_space_size(), custom_padding_value);

        const auto input_tensor =
            tensor_view<decltype(buffer_view), decltype(in_desc)>{buffer_view, in_desc};

        [[maybe_unused]] auto input_window =
            make_tile_window(input_tensor,
                             make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                             {0, 0},
                             Policy::template MakeXBlockTileDistribution<Problem>());

        auto out_buffer_view = make_buffer_view<address_space_enum::global>(
            p_out,
            in_desc.get_element_space_size(),
            type_convert<OutDataType>(custom_padding_value));

        auto y_tensor =
            tensor_view<decltype(out_buffer_view), decltype(in_desc)>{out_buffer_view, in_desc};

        [[maybe_unused]] auto y_window =
            make_tile_window(y_tensor,
                             make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                             {0, 0},
                             Policy::template MakeXBlockTileDistribution<Problem>());
        // Dummy copy from input to output

        [[maybe_unused]] auto input_tile = load_tile(input_window);

        // auto y_tile = MakeYBlockTile<decltype(input_window)>();
        auto y_tile = make_static_distributed_tensor<OutDataType>(
            Policy::template MakeXBlockTileDistribution<Problem>());

        // Set all output elements to the custom padding value.
        // // Simple solution to set the whole tile to a constant //
        // set_tile(y_tile, custom_padding_value);
        // store_tile(y_window, y_tile);

        constexpr auto y_spans = y_tile.get_distributed_spans();
        sweep_tile_span(y_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(y_spans[number<1>{}], [&](auto idx1) {
                constexpr auto distributed_indices = make_tuple(idx0, idx1);
                y_tile(distributed_indices) = type_convert<OutDataType>(custom_padding_value);
            });
        });

        store_tile(y_window, y_tile);
    }
};

} // namespace ck_tile
