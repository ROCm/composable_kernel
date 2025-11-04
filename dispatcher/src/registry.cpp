// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/dispatcher/registry.hpp"
#include <algorithm>

namespace ck_tile {
namespace dispatcher {

bool Registry::register_kernel(KernelInstancePtr instance, Priority priority)
{
    if (!instance) {
        return false;
    }
    
    const std::string identifier = instance->get_key().encode_identifier();
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = kernels_.find(identifier);
    if (it != kernels_.end()) {
        // Kernel with this identifier already exists
        // Only replace if new priority is higher
        if (priority > it->second.priority) {
            it->second.instance = instance;
            it->second.priority = priority;
            return true;
        }
        return false;  // Existing kernel has higher or equal priority
    }
    
    // New kernel, insert it
    kernels_[identifier] = RegistryEntry{instance, priority};
    return true;
}

KernelInstancePtr Registry::lookup(const std::string& identifier) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = kernels_.find(identifier);
    if (it != kernels_.end()) {
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
    
    for (const auto& pair : kernels_) {
        result.push_back(pair.second.instance);
    }
    
    return result;
}

std::vector<KernelInstancePtr> Registry::filter(
    std::function<bool(const KernelInstance&)> predicate) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<KernelInstancePtr> result;
    
    for (const auto& pair : kernels_) {
        if (predicate(*pair.second.instance)) {
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

void Registry::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    kernels_.clear();
}

Registry& Registry::instance()
{
    static Registry registry;
    return registry;
}

} // namespace dispatcher
} // namespace ck_tile

