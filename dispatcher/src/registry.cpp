// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/json_export.hpp"
#include "ck_tile/dispatcher/arch_filter.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>

namespace ck_tile {
namespace dispatcher {

Registry::Registry() = default;

Registry::~Registry()
{
    if(auto_export_enabled_)
    {
        perform_auto_export();
    }
}

Registry::Registry(Registry&& other) noexcept : Base(std::move(other))
{
    // Base move constructor already locked+released other.mutex_.
    // Re-acquire to safely read the remaining fields.
    std::lock_guard<std::mutex> lock(other.mutex());
    auto_export_enabled_               = other.auto_export_enabled_;
    auto_export_filename_              = std::move(other.auto_export_filename_);
    auto_export_include_statistics_    = other.auto_export_include_statistics_;
    auto_export_on_every_registration_ = other.auto_export_on_every_registration_;

    other.auto_export_enabled_ = false;
}

Registry& Registry::operator=(Registry&& other) noexcept
{
    if(this != &other)
    {
        Base::operator=(std::move(other));
        auto_export_enabled_               = other.auto_export_enabled_;
        auto_export_filename_              = std::move(other.auto_export_filename_);
        auto_export_include_statistics_    = other.auto_export_include_statistics_;
        auto_export_on_every_registration_ = other.auto_export_on_every_registration_;

        // Disable auto-export on the moved-from object
        other.auto_export_enabled_ = false;
    }
    return *this;
}

bool Registry::register_kernel(KernelInstancePtr instance, Priority priority)
{
    if(!instance)
        return false;

    // Store under the encoded identifier so Registry::lookup(KernelKey) finds it.
    // Previously stored under instance->get_name(), but lookup(KernelKey) queries by
    // key.encode_identifier() — those keys never matched, breaking key-based lookup.
    if(Base::register_kernel(instance->get_key().encode_identifier(), instance, priority))
    {
        if(auto_export_enabled_ && auto_export_on_every_registration_)
        {
            perform_auto_export();
        }
        return true;
    }
    return false;
}

KernelInstancePtr Registry::lookup(const std::string& identifier) const
{
    std::lock_guard<std::mutex> lock(mutex());
    auto it = entries().find(identifier);
    if(it != entries().end())
    {
        return it->second.instance;
    }
    return nullptr;
}

KernelInstancePtr Registry::lookup(const KernelKey& key) const
{
    return lookup(key.encode_identifier());
}

std::vector<KernelInstancePtr> Registry::get_all() const { return Base::get_all_instances(); }

std::vector<KernelInstancePtr>
Registry::filter(std::function<bool(const KernelInstance&)> predicate) const
{
    std::lock_guard<std::mutex> lock(mutex());
    std::vector<KernelInstancePtr> result;
    for(const auto& [name, entry] : entries())
    {
        if(predicate(*(entry.instance)))
        {
            result.push_back(entry.instance);
        }
    }
    return result;
}

std::string Registry::export_json(bool include_statistics) const
{
    return export_registry_json(*this, include_statistics);
}

bool Registry::export_json_to_file(const std::string& filename, bool include_statistics) const
{
    return export_registry_json_to_file(*this, filename, include_statistics);
}

void Registry::enable_auto_export(const std::string& filename,
                                  bool include_statistics,
                                  bool export_on_every_registration)
{
    std::lock_guard<std::mutex> lock(mutex());
    auto_export_enabled_               = true;
    auto_export_filename_              = filename;
    auto_export_include_statistics_    = include_statistics;
    auto_export_on_every_registration_ = export_on_every_registration;
}

void Registry::disable_auto_export()
{
    std::lock_guard<std::mutex> lock(mutex());
    auto_export_enabled_ = false;
}

bool Registry::is_auto_export_enabled() const
{
    std::lock_guard<std::mutex> lock(mutex());
    return auto_export_enabled_;
}

void Registry::perform_auto_export()
{
    // Don't hold the lock during file I/O
    std::string filename;
    bool include_stats;

    {
        std::lock_guard<std::mutex> lock(mutex());
        if(!auto_export_enabled_)
        {
            return;
        }
        filename      = auto_export_filename_;
        include_stats = auto_export_include_statistics_;
    }

    // Export without holding the lock
    export_json_to_file(filename, include_stats);
}

std::size_t Registry::filter_by_arch(const std::string& gpu_arch)
{
    ArchFilter filter(gpu_arch);
    std::vector<std::string> to_remove;

    {
        std::lock_guard<std::mutex> lock(mutex());

        for(const auto& pair : entries())
        {
            if(!filter.is_valid(pair.second.instance->get_key()))
            {
                to_remove.push_back(pair.first);
            }
        }

        for(const auto& key : to_remove)
        {
            entries_mut().erase(key);
        }
    }

    return to_remove.size();
}

Registry& Registry::instance()
{
    static Registry global_registry;
    return global_registry;
}

} // namespace dispatcher
} // namespace ck_tile