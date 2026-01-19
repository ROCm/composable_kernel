#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct SinkhornKnoppArgs
{
    void* out;
    const void* p_x;
    int max_iterations;
};

struct SinkhornKnoppKernelReduce
{
    template <typename Problem>
    CK_TILE_DEVICE void operator()(const SinkhornKnoppArgs& args) const {
        // Creating tensor descriptors, views and windows for inputs and outputs

        // Create the reduce ops
            // * Reduce Op ADD for row and column sums
            // * Elementwise Op EXP for exponentiation

        using ExponentiationOp = ElementwiseOp<ExponentiationOperation>;
        using AddOp = ElementwiseOp<AddOperation>;
        using DivideOp = ElementwiseOp<DivideOperation>;

        using ReduceOp = ReduceOp<AddOp, AddOp>;
        // Run the first steps iteration of the Sinkhorn-Knopp algorithm
        // Exponentiate the matrix x
        auto x = load_tile(...);

        // Hot loop for Sinkhorn-Knopp iterations from 1 to max_iterations
        // Use BlockReduce2D for row and column sums
        for (int i = 0; i <= args.max_iterations; i++) {
            // 0. LOAD x
            // 1. Compute row sums (REDUCE)
            // 2. Divide values by row sums (SWEEP)
            // 3. STORE the result of the division (in transposed format)
            // 4. LOAD transposed x
            // 5. Compute column sums (REDUCE)
            // 6. Divide values by column sums (SWEEP)
            // 7. STORE the result of the division (in transposed format)
        }
    }
};

struct SinkhornKnoppKernelDummyNonStochastic
{
    template <typename Problem>
    CK_TILE_DEVICE void operator()(const SinkhornKnoppArgs& args) const {
        using S = Problem::BlockShape;
        
        auto desc = make_naive_tensor_descriptor();

        auto buffer_view = make_buffer_view<address_space_enum::global>(
            args.p_x, desc.get_element_space_size(), number<0>{});

        const auto x_tensor =
            tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc};

        auto x_window =
            make_tile_window(x_tensor,
                                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                {0, 0},
                                Policy::template MakeXBlockTileDistribution<Problem>());

    }
};

} // namespace ck_tile