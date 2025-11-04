// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdint>

namespace ck_tile {
namespace dispatcher {

/// Problem: Runtime parameters for kernel invocation
/// Captures problem dimensions and resource constraints that vary between invocations
/// even when using the same kernel
struct Problem {
    // Problem dimensions
    std::int64_t M;  // Number of rows in A and C
    std::int64_t N;  // Number of columns in B and C
    std::int64_t K;  // Shared dimension (columns of A, rows of B)
    
    // Batch configuration
    std::int32_t k_batch;  // Number of K-dimension splits for split-K GEMM
    
    // Resource preferences
    std::int32_t smem_budget;      // Shared memory budget in bytes (0 = no constraint)
    bool prefer_persistent;         // Prefer persistent kernel variants
    
    // Validation control
    bool enable_validation;  // Enable output validation against reference
    
    /// Default constructor with sensible defaults
    Problem()
        : M(0)
        , N(0)
        , K(0)
        , k_batch(1)
        , smem_budget(0)
        , prefer_persistent(false)
        , enable_validation(false)
    {}
    
    /// Constructor with problem dimensions
    Problem(std::int64_t m, std::int64_t n, std::int64_t k)
        : M(m)
        , N(n)
        , K(k)
        , k_batch(1)
        , smem_budget(0)
        , prefer_persistent(false)
        , enable_validation(false)
    {}
    
    /// Check if problem dimensions are valid
    [[nodiscard]] bool is_valid() const
    {
        return M > 0 && N > 0 && K > 0 && k_batch > 0;
    }
    
    /// Get total number of operations (for performance metrics)
    [[nodiscard]] std::int64_t num_ops() const
    {
        return 2 * M * N * K;  // Multiply-add counts as 2 ops
    }
};

} // namespace dispatcher
} // namespace ck_tile

