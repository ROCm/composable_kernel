// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "conv_signature_types.hpp"

// Export macro for dynamic library
#define CKB_EXPORT __attribute__((visibility("default")))

namespace ck_tile::builder::registry {

using namespace ck_tile::builder::test;

// Registry entry structure
struct InstanceEntry {
    //std::string signature_hash;
    std::string type_string;
    std::function<void*()> create_invoker;
    
    // Metadata for selection
    struct Metadata {
        //ConvSignature signature;
        std::string algorithm_name;
    } metadata;
};

// Main registry class
class ConvInstanceRegistry {
private:
    std::unordered_map<std::string, InstanceEntry> entries_;
    
public:
    // Register an instance
    void register_instance(const std::string& id, InstanceEntry entry) {
        entries_[id] = std::move(entry);
    }
    
    // Get instance by ID
    const InstanceEntry* get_instance(const std::string& id) const {
        auto it = entries_.find(id);
        return (it != entries_.end()) ? &it->second : nullptr;
    }
    
    // Get all registered instance IDs
    std::vector<std::string> get_all_instance_ids() const {
        std::vector<std::string> ids;
        ids.reserve(entries_.size());
        for (const auto& [id, entry] : entries_) {
            ids.push_back(id);
        }
        return ids;
    }
    
    // Get registry statistics
    struct Stats {
        size_t total_instances;
        std::unordered_map<std::string, size_t> by_data_type;
        std::unordered_map<std::string, size_t> by_layout;
    };
    
    Stats get_stats() const {
        Stats stats;
        stats.total_instances = entries_.size();
        // Implement counting logic here
        return stats;
    }

private:
    bool signatures_compatible(const ConvSignature& registered, const ConvSignature& target) const {
        return registered.spatial_dim == target.spatial_dim &&
               registered.direction == target.direction &&
               //registered.layout == target.layout &&
               registered.data_type == target.data_type;
    }
};

// Global registry instance
static ConvInstanceRegistry& get_global_registry() {
    static ConvInstanceRegistry registry;
    return registry;
}

// Auto-registration helper
template<typename Builder>
struct AutoRegister {
    AutoRegister(const std::string& id) {
        using Instance = typename Builder::Instance;
        
        // Get the signature first to use in initialization
        // TODO: Get this from builder.
        //ConvSignature builder_signature{};
        
        // Initialize InstanceEntry with proper metadata initialization
        InstanceEntry entry{
            //.signature_hash = compute_signature_hash(builder_signature),
            .type_string = Instance{}.GetInstanceString(),
            .create_invoker = []() -> void* {
                return Instance{}.MakeInvokerPointer().release();
            },
            .metadata = {
                //.signature = builder_signature,
                .algorithm_name = Instance{}.GetInstanceString()
            }
        };
        
        get_global_registry().register_instance(id, std::move(entry));
    }
};

} // namespace ck_tile::builder::registry

// C API for dynamic library usage
extern "C" {
    // Get the registry instance
    CKB_EXPORT void* ckb_get_registry() {
        return &ck_tile::builder::registry::get_global_registry();
    }
    
    // Get instance count
    CKB_EXPORT size_t ckb_get_instance_count() {
        return ck_tile::builder::registry::get_global_registry().get_all_instance_ids().size();
    }
    
    // Get all instance IDs
    CKB_EXPORT const char** ckb_get_all_instance_ids(size_t* count) {
        static std::vector<std::string> ids;
        static std::vector<const char*> c_strs;
        
        ids = ck_tile::builder::registry::get_global_registry().get_all_instance_ids();
        c_strs.clear();
        c_strs.reserve(ids.size());
        
        for (const auto& id : ids) {
            c_strs.push_back(id.c_str());
        }
        
        *count = c_strs.size();
        return c_strs.data();
    }
    
    // Create invoker by ID
    CKB_EXPORT void* ckb_create_invoker(const char* instance_id) {
        auto* entry = ck_tile::builder::registry::get_global_registry().get_instance(instance_id);
        return entry ? entry->create_invoker() : nullptr;
    }
    
    // Get type string by ID
    CKB_EXPORT const char* ckb_get_type_string(const char* instance_id) {
        auto* entry = ck_tile::builder::registry::get_global_registry().get_instance(instance_id);
        return entry ? entry->type_string.c_str() : nullptr;
    }
}
