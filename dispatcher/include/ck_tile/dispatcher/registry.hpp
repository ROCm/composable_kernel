// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
 * - Supports both singleton and multiple instance patterns
 *
 * Usage (Singleton - backward compatible):
 *   auto& registry = Registry::instance();
 *   registry.register_kernel(kernel, Priority::High);
 *   auto kernel = registry.lookup("kernel_name");
 *
 * Usage (Multiple registries):
 *   Registry fp16_registry;
 *   Registry bf16_registry;
 *   fp16_registry.register_kernel(fp16_kernel, Priority::High);
 *   bf16_registry.register_kernel(bf16_kernel, Priority::High);
 *
 *   Dispatcher fp16_dispatcher(&fp16_registry);
 *   Dispatcher bf16_dispatcher(&bf16_registry);
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
#include <memory>

namespace ck_tile {
namespace dispatcher {

/// Registry: Central mapping from kernel configurations to executable instances
/// Thread-safe kernel registration and lookup
/// Supports both singleton pattern and multiple independent instances
class Registry
{
    public:
    /// Priority levels for conflict resolution when multiple kernels have same key
    enum class Priority
    {
        Low    = 0,
        Normal = 1,
        High   = 2
    };

    /// Default constructor - creates an empty registry instance
    /// Use this to create independent registries for different kernel sets
    Registry();

    /// Destructor - triggers auto-export if enabled
    ~Registry();

    /// Move constructor
    Registry(Registry&& other) noexcept;

    /// Move assignment
    Registry& operator=(Registry&& other) noexcept;

    // Prevent copying (registries contain shared_ptrs that shouldn't be duplicated)
    Registry(const Registry&)            = delete;
    Registry& operator=(const Registry&) = delete;

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
    [[nodiscard]] std::vector<KernelInstancePtr>
    filter(std::function<bool(const KernelInstance&)> predicate) const;

    /// Get number of registered kernels
    [[nodiscard]] std::size_t size() const;

    /// Check if registry is empty
    [[nodiscard]] bool empty() const;

    /// Clear all registered kernels
    void clear();

    /// Get registry name (for logging/debugging)
    [[nodiscard]] const std::string& get_name() const;

    /// Set registry name (for logging/debugging)
    void set_name(const std::string& name);

    /// Export registry to JSON string
    /// @param include_statistics Whether to include kernel statistics breakdown
    /// @return JSON string with all kernel metadata
    [[nodiscard]] std::string export_json(bool include_statistics = true) const;

    /// Export registry to JSON file
    /// @param filename Output filename
    /// @param include_statistics Whether to include kernel statistics breakdown
    /// @return true if export succeeded, false otherwise
    bool export_json_to_file(const std::string& filename, bool include_statistics = true) const;

    /// Enable automatic JSON export on kernel registration
    /// @param filename Output filename for auto-export
    /// @param include_statistics Whether to include statistics in auto-export
    /// @param export_on_every_registration If true, exports after every registration (default).
    ///                                      If false, only exports on destruction.
    void enable_auto_export(const std::string& filename,
                            bool include_statistics           = true,
                            bool export_on_every_registration = true);

    /// Disable automatic JSON export
    void disable_auto_export();

    /// Check if auto-export is enabled
    [[nodiscard]] bool is_auto_export_enabled() const;

    /// Merge kernels from another registry into this one
    /// @param other Registry to merge from
    /// @param priority Priority for merged kernels (default: Normal)
    /// @return Number of kernels successfully merged
    std::size_t merge_from(const Registry& other, Priority priority = Priority::Normal);

    /// Filter kernels in-place by architecture
    /// @param gpu_arch Target GPU architecture string (e.g., "gfx942")
    /// @return Number of kernels removed
    std::size_t filter_by_arch(const std::string& gpu_arch);

    /// Get singleton instance of the global registry (backward compatible)
    /// This is the default registry used when no specific registry is provided
    static Registry& instance();

    private:
    struct RegistryEntry
    {
        KernelInstancePtr instance;
        Priority priority;
    };

    /// Perform auto-export if enabled
    void perform_auto_export();

    mutable std::mutex mutex_;
    std::unordered_map<std::string, RegistryEntry> kernels_;
    std::string name_;

    // Auto-export configuration
    bool auto_export_enabled_ = false;
    std::string auto_export_filename_;
    bool auto_export_include_statistics_    = true;
    bool auto_export_on_every_registration_ = true;
};

/// Shared pointer type for registries (useful for managing lifetime)
using RegistryPtr = std::shared_ptr<Registry>;

/// Create a new registry instance (factory function)
inline RegistryPtr make_registry(const std::string& name = "")
{
    auto reg = std::make_shared<Registry>();
    if(!name.empty())
    {
        reg->set_name(name);
    }
    return reg;
}

} // namespace dispatcher
} // namespace ck_tile
