// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Registry - Thread-Safe Kernel Storage
 * 
 * Central registry for all available kernel instances with priority-based
 * ordering and efficient lookup.
 * 
 * Features:
 * - Thread-safe registration and lookup
 * - Priority-based ordering (High, Normal, Low)
 * - Lookup by name or KernelKey
 * - Filter by problem compatibility
 * - Singleton pattern for global access
 * 
 * Usage:
 *   auto& registry = Registry::instance();
 *   registry.register_kernel(kernel, Priority::High);
 *   auto kernel = registry.lookup("kernel_name");
 * 
 * Status: Production ready, thread-safe
 */

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ck_tile {
namespace dispatcher {

/// Registry: Central mapping from kernel configurations to executable instances
/// Thread-safe kernel registration and lookup
class Registry {
public:
    /// Priority levels for conflict resolution when multiple kernels have same key
    enum class Priority {
        Low = 0,
        Normal = 1,
        High = 2
    };
    
    /// Register a kernel instance with the registry
    /// @param instance Kernel instance to register
    /// @param priority Priority level for conflict resolution (default: Normal)
    /// @return true if registered successfully, false if duplicate with higher priority exists
    bool register_kernel(KernelInstancePtr instance, Priority priority = Priority::Normal);
    
    /// Lookup a kernel by its string identifier
    /// @param identifier Kernel identifier string
    /// @return Kernel instance if found, nullptr otherwise
    [[nodiscard]] KernelInstancePtr lookup(const std::string& identifier) const;
    
    /// Lookup a kernel by its KernelKey
    /// @param key Kernel configuration key
    /// @return Kernel instance if found, nullptr otherwise
    [[nodiscard]] KernelInstancePtr lookup(const KernelKey& key) const;
    
    /// Get all registered kernels
    /// @return Vector of all kernel instances
    [[nodiscard]] std::vector<KernelInstancePtr> get_all() const;
    
    /// Get all kernels matching a predicate
    /// @param predicate Function to filter kernels
    /// @return Vector of matching kernel instances
    [[nodiscard]] std::vector<KernelInstancePtr> filter(
        std::function<bool(const KernelInstance&)> predicate) const;
    
    /// Get number of registered kernels
    [[nodiscard]] std::size_t size() const;
    
    /// Clear all registered kernels
    void clear();
    
    /// Get singleton instance of the registry
    static Registry& instance();

private:
    Registry() = default;
    ~Registry() = default;
    
    // Prevent copying
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
    
    struct RegistryEntry {
        KernelInstancePtr instance;
        Priority priority;
    };
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, RegistryEntry> kernels_;
};

} // namespace dispatcher
} // namespace ck_tile

