#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct SinkhornKnoppArgs
{
    void* out;
    const void* p_x;
    const index_t n;
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

} // namespace ck_tile