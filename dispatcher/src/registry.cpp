// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/json_export.hpp"
#include "ck_tile/dispatcher/arch_filter.hpp"
#include <algorithm>

namespace ck_tile {
namespace dispatcher {

Registry::Registry()
    : name_("default"),
      auto_export_enabled_(false),
      auto_export_include_statistics_(true),
      auto_export_on_every_registration_(true)
{
}

Registry::~Registry()
{
    // Perform auto-export on destruction if enabled (regardless of export_on_every_registration
    // setting)
    if(auto_export_enabled_)
    {
        perform_auto_export();
    }
}

Registry::Registry(Registry&& other) noexcept
    : mutex_() // mutex is not movable, create new one
      ,
      kernels_(std::move(other.kernels_)),
      name_(std::move(other.name_)),
      auto_export_enabled_(other.auto_export_enabled_),
      auto_export_filename_(std::move(other.auto_export_filename_)),
      auto_export_include_statistics_(other.auto_export_include_statistics_),
      auto_export_on_every_registration_(other.auto_export_on_every_registration_)
{
    // Disable auto-export on the moved-from object to prevent double export
    other.auto_export_enabled_ = false;
}

Registry& Registry::operator=(Registry&& other) noexcept
{
    if(this != &other)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> other_lock(other.mutex_);

        kernels_                           = std::move(other.kernels_);
        name_                              = std::move(other.name_);
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
    {
        return false;
    }

    const std::string identifier = instance->get_key().encode_identifier();

    bool registered = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = kernels_.find(identifier);
        if(it != kernels_.end())
        {
            // Kernel with this identifier already exists
            // Only replace if new priority is higher
            if(priority > it->second.priority)
            {
                it->second.instance = instance;
                it->second.priority = priority;
                registered          = true;
            }
        }
        else
        {
            // New kernel, insert it
            kernels_[identifier] = RegistryEntry{instance, priority};
            registered           = true;
        }
    }

    // Perform auto-export if enabled and configured to export on every registration
    if(registered && auto_export_enabled_ && auto_export_on_every_registration_)
    {
        perform_auto_export();
    }

    return registered;
}

KernelInstancePtr Registry::lookup(const std::string& identifier) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(identifier);
    if(it != kernels_.end())
    {
        return it->second.instance;
    }

    return nullptr;
}

KernelInstancePtr Registry::lookup(const KernelKey& key) const
{
    return lookup(key.encode_identifier());
}

std::vector<KernelInstancePtr> Registry::get_all() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<KernelInstancePtr> result;
    result.reserve(kernels_.size());

    for(const auto& pair : kernels_)
    {
        result.push_back(pair.second.instance);
    }

    return result;
}

std::vector<KernelInstancePtr>
Registry::filter(std::function<bool(const KernelInstance&)> predicate) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<KernelInstancePtr> result;

    for(const auto& pair : kernels_)
    {
        if(predicate(*pair.second.instance))
        {
            result.push_back(pair.second.instance);
        }
    }

    return result;
}

std::size_t Registry::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return kernels_.size();
}

bool Registry::empty() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return kernels_.empty();
}

void Registry::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    kernels_.clear();
}

const std::string& Registry::get_name() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return name_;
}

void Registry::set_name(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    name_ = name;
}

Registry& Registry::instance()
{
    static Registry global_registry;
    return global_registry;
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
    std::lock_guard<std::mutex> lock(mutex_);
    auto_export_enabled_               = true;
    auto_export_filename_              = filename;
    auto_export_include_statistics_    = include_statistics;
    auto_export_on_every_registration_ = export_on_every_registration;
}

void Registry::disable_auto_export()
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto_export_enabled_ = false;
}

bool Registry::is_auto_export_enabled() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return auto_export_enabled_;
}

void Registry::perform_auto_export()
{
    // Don't hold the lock during file I/O
    std::string filename;
    bool include_stats;

    {
        std::lock_guard<std::mutex> lock(mutex_);
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

std::size_t Registry::merge_from(const Registry& other, Priority priority)
{
    auto other_kernels       = other.get_all();
    std::size_t merged_count = 0;

    for(const auto& kernel : other_kernels)
    {
        if(register_kernel(kernel, priority))
        {
            merged_count++;
        }
    }

    return merged_count;
}

std::size_t Registry::filter_by_arch(const std::string& gpu_arch)
{
    ArchFilter filter(gpu_arch);
    std::vector<std::string> to_remove;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        for(const auto& pair : kernels_)
        {
            if(!filter.is_valid(pair.second.instance->get_key()))
            {
                to_remove.push_back(pair.first);
            }
        }

        for(const auto& key : to_remove)
        {
            kernels_.erase(key);
        }
    }

    return to_remove.size();
}

} // namespace dispatcher
} // namespace ck_tile
