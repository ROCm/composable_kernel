// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ck_tile {
namespace dispatcher {

/// Shared priority enum used by all registry types
enum class Priority
{
    Low    = 0,
    Normal = 1,
    High   = 2
};

/// BaseRegistry: Thread-safe, priority-aware kernel storage shared by GEMM and Conv registries.
///
/// Template Parameters:
///   Derived      - CRTP derived class (e.g., Registry, ConvRegistry)
///   KeyType      - primary key type (std::string for GEMM, ConvKernelKey for Conv)
///   InstanceType - kernel instance type (KernelInstance, ConvKernelInstance)
///   KeyHash      - hash functor for KeyType (defaults to std::hash<KeyType>)
template <typename Derived,
          typename KeyType,
          typename InstanceType,
          typename KeyHash = std::hash<KeyType>>
class BaseRegistry
{
    public:
    using InstancePtr = std::shared_ptr<InstanceType>;

    struct Entry
    {
        InstancePtr instance;
        Priority priority;
    };

    BaseRegistry()          = default;
    virtual ~BaseRegistry() = default;

    BaseRegistry(BaseRegistry&& other) noexcept
    {
        std::lock_guard<std::mutex> lock(other.mutex_);
        entries_ = std::move(other.entries_);
        name_    = std::move(other.name_);
    }

    BaseRegistry& operator=(BaseRegistry&& other) noexcept
    {
        if(this != &other)
        {
            std::scoped_lock lock(mutex_, other.mutex_);
            entries_ = std::move(other.entries_);
            name_    = std::move(other.name_);
        }
        return *this;
    }

    BaseRegistry(const BaseRegistry&)            = delete;
    BaseRegistry& operator=(const BaseRegistry&) = delete;

    /// Register a kernel. If the key already exists, the new entry replaces it
    /// only when its priority is strictly higher than the existing entry's
    /// priority. Same-priority registration is rejected (first-writer-wins).
    bool
    register_kernel(const KeyType& key, InstancePtr instance, Priority priority = Priority::Normal)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if(it != entries_.end() && it->second.priority >= priority)
        {
            return false;
        }
        entries_[key] = Entry{std::move(instance), priority};
        return true;
    }

    [[nodiscard]] std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    [[nodiscard]] bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.empty();
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
    }

    [[nodiscard]] std::string get_name() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return name_; // return by value to avoid dangling reference
    }

    void set_name(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        name_ = name;
    }

    [[nodiscard]] std::vector<InstancePtr> get_all_instances() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<InstancePtr> result;
        result.reserve(entries_.size());
        for(const auto& [key, entry] : entries_)
        {
            result.push_back(entry.instance);
        }
        return result;
    }

    std::size_t merge_from(const BaseRegistry& other, Priority priority = Priority::Normal)
    {
        std::scoped_lock lock(mutex_, other.mutex_);
        std::size_t merged = 0;
        for(const auto& [key, entry] : other.entries_)
        {
            auto it = entries_.find(key);
            if(it == entries_.end() || it->second.priority <= priority)
            {
                entries_[key] = Entry{entry.instance, priority};
                ++merged;
            }
        }
        return merged;
    }

    /// Enable automatic JSON export after every kernel registration.
    /// Requires the derived class to implement export_json_to_file(path, stats).
    void enable_auto_export(const std::string& path,
                            bool include_statistics           = true,
                            bool export_on_every_registration = true)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto_export_path_        = path;
        auto_export_stats_       = include_statistics;
        auto_export_on_register_ = export_on_every_registration;
        auto_export_enabled_.store(true, std::memory_order_release);
    }

    void disable_auto_export() { auto_export_enabled_.store(false, std::memory_order_release); }

    [[nodiscard]] bool is_auto_export_enabled() const
    {
        return auto_export_enabled_.load(std::memory_order_acquire);
    }

    /// Call after registration to trigger auto-export if enabled.
    void perform_auto_export()
    {
        if(!auto_export_enabled_.load(std::memory_order_acquire))
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        if(auto_export_on_register_)
        {
            static_cast<Derived*>(this)->export_json_to_file(auto_export_path_, auto_export_stats_);
        }
    }

    protected:
    [[nodiscard]] const std::unordered_map<KeyType, Entry, KeyHash>& entries() const
    {
        return entries_;
    }

    [[nodiscard]] std::unordered_map<KeyType, Entry, KeyHash>& entries_mut() { return entries_; }

    std::mutex& mutex() const { return mutex_; }

    private:
    mutable std::mutex mutex_;
    std::unordered_map<KeyType, Entry, KeyHash> entries_;
    std::string name_ = "default";

    std::atomic<bool> auto_export_enabled_{false};
    bool auto_export_on_register_ = true;
    bool auto_export_stats_       = true;
    std::string auto_export_path_;
};

} // namespace dispatcher
} // namespace ck_tile
