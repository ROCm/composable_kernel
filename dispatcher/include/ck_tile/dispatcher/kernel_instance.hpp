// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include <memory>
#include <string>

namespace ck_tile {
namespace dispatcher {

/// KernelInstance: Uniform interface for kernel execution
/// Abstracts away implementation details (CK Library vs CK Tile vs future JIT)
/// Enables type-erased storage in registry while backends perform type-safe casts
class KernelInstance
{
    public:
    virtual ~KernelInstance() = default;

    /// Get the kernel's configuration metadata
    [[nodiscard]] virtual const KernelKey& get_key() const = 0;

    /// Check if this kernel supports the given problem
    /// Returns false if problem dimensions don't meet kernel requirements
    /// (e.g., divisibility constraints, resource limits)
    [[nodiscard]] virtual bool supports(const Problem& problem) const = 0;

    /// Get human-readable kernel name for logging and debugging
    [[nodiscard]] virtual std::string get_name() const = 0;

    /// Execute the kernel with given problem and data pointers
    /// @param a_ptr Pointer to matrix A (device memory)
    /// @param b_ptr Pointer to matrix B (device memory)
    /// @param c_ptr Pointer to matrix C (device memory, input/output)
    /// @param d_ptrs Array of pointers to additional D tensors for fusion (device memory)
    /// @param problem Problem configuration
    /// @param stream HIP stream for kernel launch (nullptr = default stream)
    /// @return Kernel execution time in milliseconds (0 if timing not available)
    [[nodiscard]] virtual float run(const void* a_ptr,
                                    const void* b_ptr,
                                    void* c_ptr,
                                    const void** d_ptrs,
                                    const Problem& problem,
                                    void* stream = nullptr) const = 0;

    /// Validate kernel output against reference implementation
    /// @param a_ptr Pointer to matrix A (device memory)
    /// @param b_ptr Pointer to matrix B (device memory)
    /// @param c_ptr Pointer to matrix C (device memory, kernel output)
    /// @param d_ptrs Array of pointers to additional D tensors (device memory)
    /// @param problem Problem configuration
    /// @param tolerance Relative error tolerance for validation
    /// @return true if validation passes, false otherwise
    [[nodiscard]] virtual bool validate(const void* a_ptr,
                                        const void* b_ptr,
                                        const void* c_ptr,
                                        const void** d_ptrs,
                                        const Problem& problem,
                                        float tolerance = 1e-3f) const = 0;
};

/// Shared pointer type for kernel instances
using KernelInstancePtr = std::shared_ptr<KernelInstance>;

} // namespace dispatcher
} // namespace ck_tile
