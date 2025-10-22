// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
enum StreamKReductionStrategy : uint32_t
{
    Atomic    = 0u,
    Reduction = 1u
};

/**
 * @brief Estimates the number of Stream-K workgroups per macro tile in the C tensor.
 *
 * @param sk_ctas           Number of Stream-K workgroups.
 * @param iters_per_sk_cta  Number of iterations per Stream-K workgroup.
 * @param iters_per_tile    Number of iterations per tile (i.e., the number of macro tiles in the K
 * dimension).
 * @return ck_tile::index_t An estimate of the number of workgroups per macro tile in the C tensor.
 * @note It is assumed that `iters_per_sk_cta` > 0.
 */
template <ck_tile::StreamKReductionStrategy ReductionStrategy>
ck_tile::index_t
estimate_num_wgs_per_tile(index_t sk_ctas, index_t iters_per_sk_cta, index_t iters_per_tile)
{
    // In the case of non-atomic reduction or data-parallel only, there will always be 1 workgroup
    // writing final results to a given macro tile in C.
    int num_wgs_per_tile = 1;

    // Otherwise, for atomics, multiple workgroups may be writing to the same macro tile in C.
    if(sk_ctas > 0 && ReductionStrategy == ck_tile::StreamKReductionStrategy::Atomic)
    {
        // Estimate the number of workgroups per macro tile.
        num_wgs_per_tile =
            (iters_per_tile / iters_per_sk_cta) + ((iters_per_tile % iters_per_sk_cta) != 0);
    }

    return std::max(num_wgs_per_tile, 1);
}
} // namespace ck_tile
