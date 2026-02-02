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
    int iterations;
};

template <typename Problem, typename Policy>
struct SinkhornKnoppKernelReduce
{
    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    CK_TILE_DEVICE void operator()(const SinkhornKnoppArgs& args) const
    {
        // Creating tensor descriptors, views and windows for inputs and outputs
        using S               = Problem::BlockShape;
        using InDataType      = typename Problem::InDataType;
        using ComputeDataType = typename Problem::ComputeDataType;
        using OutDataType     = typename Problem::OutDataType;

        static_assert(S::Block_M == S::Block_N, "Input must be a square matrix!");

        auto* p_in  = static_cast<const Problem::InDataType*>(args.p_in);
        auto* p_out = static_cast<Problem::OutDataType*>(args.p_out);

        auto acc_op = ck_tile::ReduceOp::Add{};

        const auto in_out_desc =
            make_naive_tensor_descriptor(make_tuple(args.input_m, args.input_m),
                                         make_tuple(args.input_m, 1),
                                         number<4>{}, // TODO: Hardcoded
                                         // vectorization, //we should calculate it!
                                         number<1>{});

        const auto input_window = [&]() {
            // We require exp(input) > 0, and exp(padding) == 0
            const InDataType input_padding_value = -ck_tile::numeric<InDataType>::infinity();

            auto buffer_view = make_buffer_view<address_space_enum::global>(
                p_in, in_out_desc.get_element_space_size(), input_padding_value);

            const auto in_tensor =
                tensor_view<decltype(buffer_view), decltype(in_out_desc)>{buffer_view, in_out_desc};

            return make_tile_window(in_tensor,
                                    make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                    {0, 0},
                                    Policy::template MakeInputBlockTileDistribution<Problem>());
        }();

        auto out_window = [&]() {
            const OutDataType out_padding_value = acc_op.template GetIdentityValue<OutDataType>();
            auto out_buffer_view                = make_buffer_view<address_space_enum::global>(
                p_out, in_out_desc.get_element_space_size(), out_padding_value);

            auto out_tensor = tensor_view<decltype(out_buffer_view), decltype(in_out_desc)>{
                out_buffer_view, in_out_desc};

            return make_tile_window(out_tensor,
                                    make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                    {0, 0},
                                    Policy::template MakeInputBlockTileDistribution<Problem>());
        }();

        auto input_tile = load_tile(input_window);

        // Run the first steps iteration of the Sinkhorn-Knopp algorithm
        // Exponentiate the input to make it strictly positive
        // auto exp_func = [](ComputeDataType x) -> ComputeDataType { return ck_tile::exp(x); };
        // auto exp_func = []([[maybe_unused]] ComputeDataType x) {
        //     return static_cast<ComputeDataType>(1.0);
        // };
        auto exp_func = [](InDataType x) -> ComputeDataType {
            return static_cast<ComputeDataType>(x);
        };

        auto compute_tile = tile_elementwise_in(exp_func, input_tile);

        // Create a transposed tile
        [[maybe_unused]] auto compute_tile_t = make_static_distributed_tensor<ComputeDataType>(
            Policy::template MakeTransposedInputBlockTileDistribution<Problem>());

        // Hot loop for Sinkhorn-Knopp iterations from 1 to iterations
        // Use BlockReduce2D for row and column sums
        auto row_sum = [&](const auto& c_tile) {
            // TODO: Handle case where the input doesn't fit in a single tile
            using br_problem = BlockReduce2dProblem<typename Problem::ComputeDataType,
                                                    typename Problem::ComputeDataType,
                                                    typename Problem::BlockShape>;

            auto block_reduce2d      = Policy::template GetBlockReduce2d<br_problem>();
            auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<br_problem>();
            // TODO: Deduce/allow specifying a separate type for the accumulators?
            // NOTE: MakeYBlockTile defaults to reducing 2nd dimension
            auto acc_tile = block_reduce2d.template MakeYBlockTile<decltype(c_tile)>();
            set_tile(acc_tile, acc_op.template GetIdentityValue<ComputeDataType>());

            block_reduce2d(c_tile, acc_tile, acc_op);
            block_reduce2d_sync(acc_tile, acc_op);

            return acc_tile;
        };

        for(int i = 0; i < 1; i++)
        {
            // 1. Compute row sums (REDUCE)
            // FIXME: Uses overload that is hardcoded to reduce 2nd dimension, be explicit instead
            auto row_acc_tile = row_sum(compute_tile);

            // 2. Divide values by row sums (SWEEP)
            constexpr auto c_spans = compute_tile.get_distributed_spans();
            sweep_tile_span(c_spans[number<0>{}], [&](const auto idx0) {
                sweep_tile_span(c_spans[number<1>{}], [&](const auto idx1) {
                    constexpr auto c_idx       = make_tuple(idx0, idx1);
                    constexpr auto row_acc_idx = make_tuple(idx0);
                    if(threadIdx.x == 0)
                    {
                        print(row_acc_idx);
                        print(":");
                        print(row_acc_tile(row_acc_idx));
                        print("\n");
                    }
                    compute_tile(c_idx) = compute_tile(c_idx) / row_acc_tile(row_acc_idx);
                });
            });

            if(threadIdx.x == 0)
            {
                print("compute tile after rows summed and divided\n");
                [&](auto tile) {
                    constexpr auto spans = tile.get_distributed_spans();
                    sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
                        sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
                            constexpr auto idx = make_tuple(idx0, idx1);
                            print(idx);
                            print(tile(idx));
                            print("\n");
                        });
                    });
                }(compute_tile);
            }

            transpose_tile2d(compute_tile_t, compute_tile);

            if(threadIdx.x == 0)
            {
                print("compute tile transposed\n");
                [&](auto tile) {
                    constexpr auto spans = tile.get_distributed_spans();
                    sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
                        sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
                            constexpr auto idx = make_tuple(idx0, idx1);
                            print(idx);
                            print(tile(idx));
                            print("\n");
                        });
                    });
                }(compute_tile_t);
            }

            // Row sum is column sum for transposed c_tile
            auto col_acc_tile = row_sum(compute_tile_t);

            constexpr auto c_t_spans = compute_tile_t.get_distributed_spans();
            sweep_tile_span(c_t_spans[number<0>{}], [&](const auto idx0) {
                sweep_tile_span(c_t_spans[number<1>{}], [&](const auto idx1) {
                    constexpr auto c_t_idx     = make_tuple(idx0, idx1);
                    constexpr auto col_acc_idx = make_tuple(idx0);
                    compute_tile_t(c_t_idx) = compute_tile_t(c_t_idx) / col_acc_tile(col_acc_idx);
                });
            });

            transpose_tile2d(compute_tile, compute_tile_t);
        }

        // Copy the final values to the output
        store_tile(out_window, compute_tile);
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

        auto out_tensor =
            tensor_view<decltype(out_buffer_view), decltype(in_desc)>{out_buffer_view, in_desc};

        [[maybe_unused]] auto out_window =
            make_tile_window(out_tensor,
                             make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                             {0, 0},
                             Policy::template MakeXBlockTileDistribution<Problem>());
        // Dummy copy from input to output

        [[maybe_unused]] auto input_tile = load_tile(input_window);

        // auto out_tile = MakeYBlockTile<decltype(input_window)>();
        auto out_tile = make_static_distributed_tensor<OutDataType>(
            Policy::template MakeXBlockTileDistribution<Problem>());

        // Set all output elements to the custom padding value.
        // // Simple solution to set the whole tile to a constant //
        // set_tile(out_tile, custom_padding_value);
        // store_tile(out_window, out_tile);

        constexpr auto y_spans = out_tile.get_distributed_spans();
        sweep_tile_span(y_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(y_spans[number<1>{}], [&](auto idx1) {
                constexpr auto distributed_indices = make_tuple(idx0, idx1);
                out_tile(distributed_indices) = type_convert<OutDataType>(custom_padding_value);
            });
        });

        store_tile(out_window, out_tile);
    }
};

} // namespace ck_tile
