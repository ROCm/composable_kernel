#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct SinkhornKnoppArgs
{
    const void* p_x;
    const index_t n;
    int max_iterations;
};

struct SinkhornKnoppKernel
{
    template <typename Problem>
    CK_TILE_DEVICE void operator()(const SinkhornKnoppArgs& args) const {
        // Creating tensor descriptors, views and windows for inputs and outputs

        // Create the reduce ops
            // * Reduce Op ADD for row and column sums
            // * Elementwise Op EXP for exponentiation

        // Run the first steps iteration of the Sinkhorn-Knopp algorithm
        // Using the exponentiation as the elementwise operation

        // Hot loop for Sinkhorn-Knopp iterations from max_iterations=1
        //
    }
};
} // namespace ck_tile