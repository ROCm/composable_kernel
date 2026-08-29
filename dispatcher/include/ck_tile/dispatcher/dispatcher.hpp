// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Dispatcher - Main Kernel Selection and Execution Engine
 *
 * The Dispatcher provides unified interface for selecting and executing
 * CK Tile GEMM kernels based on problem specifications.
 *
 * Features:
 * - Multiple selection strategies (FirstFit, Heuristic)
 * - Custom heuristic functions
 * - Thread-safe registry integration
 * - Real GPU execution with timing
 *
 * Usage:
 *   Dispatcher dispatcher;
 *   Problem problem(M, N, K);
 *   float time = dispatcher.run(a_dev, b_dev, c_dev, problem);
 *
 * Status: Production ready - 319 TFLOPS validated
 */

#pragma once

#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ck_tile {
namespace dispatcher {

/// Heuristic function type: maps Problem to ordered list of kernel identifiers
/// Returns kernel identifiers ranked by expected performance (best first)
using HeuristicFunction = std::function<std::vector<std::string>(const Problem&)>;

/// Dispatcher: Top-level orchestration for kernel selection and execution
/// Provides unified interface for kernel dispatch across different backends
///
/// Concurrency contract: a Dispatcher instance is NOT safe for concurrent use
/// from multiple threads / HIP streams. It owns a single reduction workspace for
/// Stream-K linear/tree kernels (see workspace_ below), which would be corrupted
/// by two overlapping dispatches. Callers that need concurrency should create one
/// Dispatcher per stream/thread (the object is a lightweight handle -- just a
/// Registry* + arch string + heuristic), exactly as one would use per-stream
/// library handles. This mirrors how the workspace is zeroed on the caller's
/// stream in run() (hipMemsetAsync), so a per-stream Dispatcher stays correctly
/// ordered without any cross-stream synchronization.
class Dispatcher
{
    public:
    /// Selection strategy for kernel choice
    enum class SelectionStrategy
    {
        FirstFit, // Use first kernel that supports the problem
        Heuristic // Use heuristic function to guide selection
    };

    /// Constructor
    /// @param registry Registry instance to use (default: global singleton)
    /// @param gfx_arch Target GPU architecture (e.g. "gfx950")
    explicit Dispatcher(Registry* registry = nullptr, const std::string& gfx_arch = "");

    /// Frees the dispatcher-owned Stream-K reduction workspace, if any.
    ~Dispatcher();

    /// The Dispatcher owns a raw HIP reduction workspace that it frees in the
    /// destructor, so it must not be copied (a copy would double-free the buffer)
    /// nor moved (no use-case, and consistent with the single-stream contract
    /// above). Non-copyable, non-movable.
    Dispatcher(const Dispatcher&)            = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;
    Dispatcher(Dispatcher&&)                 = delete;
    Dispatcher& operator=(Dispatcher&&)      = delete;

    void set_arch(const std::string& arch) { gfx_arch_ = arch; }
    [[nodiscard]] const std::string& arch() const { return gfx_arch_; }

    /// Register a heuristic function for kernel selection
    /// @param heuristic Function that maps problems to ranked kernel identifiers
    void set_heuristic(HeuristicFunction heuristic);

    /// Set selection strategy
    /// @param strategy Strategy to use for kernel selection
    void set_strategy(SelectionStrategy strategy);

    /// Select a kernel for the given problem
    /// @param problem Problem configuration
    /// @return Selected kernel instance, or nullptr if no suitable kernel found
    [[nodiscard]] KernelInstancePtr select_kernel(const Problem& problem) const;

    /// Execute GEMM operation with automatic kernel selection
    /// @param a_ptr Pointer to matrix A (device memory)
    /// @param b_ptr Pointer to matrix B (device memory)
    /// @param c_ptr Pointer to matrix C (device memory, input/output)
    /// @param problem Problem configuration
    /// @param stream HIP stream for kernel launch (nullptr = default stream)
    /// @return Kernel execution time in milliseconds
    /// @throws NoKernelFound if no suitable kernel found
    [[nodiscard]] float run(const void* a_ptr,
                            const void* b_ptr,
                            void* c_ptr,
                            const Problem& problem,
                            void* stream = nullptr) const;

    /// Execute GEMM operation with fusion (multi-D)
    /// @param a_ptr Pointer to matrix A (device memory)
    /// @param b_ptr Pointer to matrix B (device memory)
    /// @param c_ptr Pointer to matrix C (device memory, input/output)
    /// @param d_ptrs Array of pointers to additional D tensors (device memory)
    /// @param problem Problem configuration
    /// @param stream HIP stream for kernel launch (nullptr = default stream)
    /// @return Kernel execution time in milliseconds
    /// @throws NoKernelFound if no suitable kernel found
    [[nodiscard]] float run_fused(const void* a_ptr,
                                  const void* b_ptr,
                                  void* c_ptr,
                                  const void** d_ptrs,
                                  const Problem& problem,
                                  void* stream = nullptr) const;

    /// Execute with explicit kernel selection
    /// @param kernel_id Kernel identifier string
    /// @param a_ptr Pointer to matrix A (device memory)
    /// @param b_ptr Pointer to matrix B (device memory)
    /// @param c_ptr Pointer to matrix C (device memory, input/output)
    /// @param d_ptrs Array of pointers to additional D tensors (device memory)
    /// @param problem Problem configuration
    /// @param stream HIP stream for kernel launch (nullptr = default stream)
    /// @return Kernel execution time in milliseconds
    /// @throws NoKernelFound if the kernel identifier is not registered
    /// @throws UnsupportedProblem if the selected kernel does not support the problem
    [[nodiscard]] float run_explicit(const std::string& kernel_id,
                                     const void* a_ptr,
                                     const void* b_ptr,
                                     void* c_ptr,
                                     const void** d_ptrs,
                                     const Problem& problem,
                                     void* stream = nullptr) const;

    /// Validate kernel output
    /// @param a_ptr Pointer to matrix A (device memory)
    /// @param b_ptr Pointer to matrix B (device memory)
    /// @param c_ptr Pointer to matrix C (device memory, kernel output)
    /// @param d_ptrs Array of pointers to additional D tensors (device memory)
    /// @param problem Problem configuration
    /// @param tolerance Relative error tolerance
    /// @return true if validation passes, false otherwise
    [[nodiscard]] bool validate(const void* a_ptr,
                                const void* b_ptr,
                                const void* c_ptr,
                                const void** d_ptrs,
                                const Problem& problem,
                                float tolerance = 1e-3f) const;

    /// Enable or disable GPU benchmarking (timing) on all kernels.
    /// When disabled, kernels execute once with no timing overhead
    /// (one-shot mode for production plugins).
    void set_benchmarking(bool enable) { benchmarking_ = enable; }
    [[nodiscard]] bool benchmarking_enabled() const { return benchmarking_; }

    private:
    Registry* registry_;
    HeuristicFunction heuristic_;
    SelectionStrategy strategy_;
    std::string gfx_arch_;
    bool benchmarking_ = true;

    // Dispatcher-owned, grow-on-demand reduction workspace for Stream-K kernels
    // (linear/tree). Sized via KernelInstance::get_workspace_size() and reused
    // across calls so we don't hipMalloc/hipFree on the hot path. Held as a raw
    // pointer to keep HIP/ck_tile out of this public header.
    mutable void* workspace_             = nullptr;
    mutable std::size_t workspace_bytes_ = 0;

    /// Ensure the owned workspace holds at least `bytes`, growing it if needed,
    /// and zero the first `bytes` on `stream` (hipMemsetAsync). Not thread-safe --
    /// see the Dispatcher concurrency contract above (one Dispatcher per stream).
    /// `stream` is a hipStream_t held as void* to keep HIP out of this header.
    void ensure_workspace(std::size_t bytes, void* stream) const;

    /// Select kernel using first-fit strategy
    [[nodiscard]] KernelInstancePtr select_first_fit(const Problem& problem) const;

    /// Select kernel using heuristic strategy
    [[nodiscard]] KernelInstancePtr select_heuristic(const Problem& problem) const;
};

} // namespace dispatcher
} // namespace ck_tile
