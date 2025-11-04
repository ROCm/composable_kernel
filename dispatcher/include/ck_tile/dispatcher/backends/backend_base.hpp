// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ck_tile {
namespace dispatcher {
namespace backends {

/// Backend type enumeration
enum class BackendType
{
    Tile,     ///< CK Tile generated kernels
    Library,  ///< CK Library pre-compiled kernels
    JIT,      ///< JIT compiled kernels (future)
    Unknown
};

/// Abstract base class for kernel instances
class KernelInstance
{
public:
    virtual ~KernelInstance() = default;

    /// Get kernel key
    virtual const KernelKey& get_key() const = 0;

    /// Check if kernel supports the given problem
    virtual bool supports(const Problem& problem) const = 0;

    /// Get kernel name
    virtual std::string get_name() const = 0;

    /// Execute kernel
    /// @param a_ptr Input tensor A device pointer
    /// @param b_ptr Input tensor B device pointer
    /// @param c_ptr Output tensor C device pointer
    /// @param problem Problem specification
    /// @param stream HIP stream
    /// @return Execution time in milliseconds
    virtual float run(const void* a_ptr,
                     const void* b_ptr,
                     void* c_ptr,
                     const Problem& problem,
                     hipStream_t stream = nullptr) = 0;

    /// Validate kernel output (optional)
    /// @param a_ptr Input tensor A device pointer
    /// @param b_ptr Input tensor B device pointer
    /// @param c_ptr Output tensor C device pointer
    /// @param problem Problem specification
    /// @param rtol Relative tolerance
    /// @param atol Absolute tolerance
    /// @return True if validation passes
    virtual bool validate(const void* a_ptr,
                         const void* b_ptr,
                         const void* c_ptr,
                         const Problem& problem,
                         float rtol = 1e-3f,
                         float atol = 1e-5f) const
    {
        (void)a_ptr;
        (void)b_ptr;
        (void)c_ptr;
        (void)problem;
        (void)rtol;
        (void)atol;
        return true; // Default: assume correct
    }

    /// Get backend type
    virtual BackendType get_backend_type() const = 0;

    /// Get kernel metadata
    virtual std::string get_metadata() const
    {
        return "backend=" + backend_type_to_string(get_backend_type()) +
               ",name=" + get_name();
    }

    /// Convert backend type to string
    static std::string backend_type_to_string(BackendType type)
    {
        switch(type)
        {
        case BackendType::Tile: return "tile";
        case BackendType::Library: return "library";
        case BackendType::JIT: return "jit";
        default: return "unknown";
        }
    }
};

/// Abstract base class for backend implementations
class BackendBase
{
public:
    virtual ~BackendBase() = default;

    /// Discover available kernels
    /// @param search_path Path to search for kernels
    /// @return List of kernel instances
    virtual std::vector<std::shared_ptr<KernelInstance>>
    discover_kernels(const std::string& search_path) = 0;

    /// Create kernel instance from configuration
    /// @param kernel_config Kernel configuration
    /// @return Kernel instance
    virtual std::shared_ptr<KernelInstance>
    create_kernel_instance(const KernelKey& kernel_key) = 0;

    /// Get backend type
    virtual BackendType get_backend_type() const = 0;

    /// Initialize backend (optional)
    virtual void initialize() {}

    /// Cleanup backend resources (optional)
    virtual void cleanup() {}
};

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile

