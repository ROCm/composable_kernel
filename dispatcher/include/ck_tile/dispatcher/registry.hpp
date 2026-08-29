// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Registry - Thread-Safe Kernel Storage
 *
 * Central registry for all available kernel instances with priority-based
 * ordering and efficient lookup.
 *
 * Derives from BaseRegistry for shared logic (thread safety, naming, priority,
 * merge) while keeping GEMM-specific APIs (lookup by KernelKey, filter_by_arch,
 * JSON export, auto-export).
 *
 * Status: Production ready, thread-safe
 */

#pragma once

#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include <functional>
#include <string>
#include <vector>
#include <memory>

namespace ck_tile {
namespace dispatcher {

/// Registry: Central mapping from kernel configurations to executable instances
/// Thread-safe kernel registration and lookup
/// Derives from BaseRegistry<Registry, std::string, KernelInstance> for shared functionality
class Registry : public BaseRegistry<Registry, std::string, KernelInstance>
{
    using Base = BaseRegistry<Registry, std::string, KernelInstance>;

    public:
    // Re-export Priority from the shared enum for backward compatibility
    using Priority = ck_tile::dispatcher::Priority;

    /// Default constructor - creates an empty registry instance
    Registry();

    /// Destructor - triggers auto-export if enabled
    ~Registry();

    /// Move constructor
    Registry(Registry&& other) noexcept;

    /// Move assignment
    Registry& operator=(Registry&& other) noexcept;

    // Prevent copying
    Registry(const Registry&)            = delete;
    Registry& operator=(const Registry&) = delete;

    /// Register a kernel instance with the registry
    bool register_kernel(KernelInstancePtr instance, Priority priority = Priority::Normal);

    /// Lookup a kernel by its string identifier
    [[nodiscard]] KernelInstancePtr lookup(const std::string& identifier) const;

    /// Lookup a kernel by its KernelKey
    [[nodiscard]] KernelInstancePtr lookup(const KernelKey& key) const;

    /// Get all registered kernels
    [[nodiscard]] std::vector<KernelInstancePtr> get_all() const;

    /// Get all kernels matching a predicate
    [[nodiscard]] std::vector<KernelInstancePtr>
    filter(std::function<bool(const KernelInstance&)> predicate) const;

    // size(), empty(), clear(), get_name(), set_name(), merge_from() inherited from Base

    /// Export registry to JSON string
    [[nodiscard]] std::string export_json(bool include_statistics = true) const;

    /// Export registry to JSON file
    bool export_json_to_file(const std::string& filename, bool include_statistics = true) const;

    void enable_auto_export(const std::string& filename,
                            bool include_statistics           = true,
                            bool export_on_every_registration = true);

    void disable_auto_export();

    [[nodiscard]] bool is_auto_export_enabled() const;

    /// Filter kernels in-place by architecture
    std::size_t filter_by_arch(const std::string& gpu_arch);

    /// Get singleton instance
    static Registry& instance();

    private:
    void perform_auto_export();

    // Auto-export configuration
    bool auto_export_enabled_ = false;
    std::string auto_export_filename_;
    bool auto_export_include_statistics_    = true;
    bool auto_export_on_every_registration_ = true;
};

/// Shared pointer type for registries
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
