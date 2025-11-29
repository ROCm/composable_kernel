// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file conv_registry.hpp
 * @brief Convolution kernel registry and dispatcher
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>

#include "ck_tile/dispatcher/conv_problem.hpp"
#include "ck_tile/dispatcher/conv_kernel_decl.hpp"

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// ConvKernelKey - Unique identifier for a convolution kernel
// =============================================================================

struct ConvKernelKey
{
    std::string dtype_in;
    std::string dtype_wei;
    std::string dtype_out;
    std::string layout; // e.g., "nhwgc_gkyxc_nhwgk"
    int ndim_spatial;   // 1, 2, or 3
    ConvOp op;

    // Tile configuration
    int tile_m;
    int tile_n;
    int tile_k;

    // Pipeline
    std::string pipeline;
    std::string scheduler;

    bool operator==(const ConvKernelKey& other) const
    {
        return dtype_in == other.dtype_in && dtype_wei == other.dtype_wei &&
               dtype_out == other.dtype_out && layout == other.layout &&
               ndim_spatial == other.ndim_spatial && op == other.op && tile_m == other.tile_m &&
               tile_n == other.tile_n && tile_k == other.tile_k && pipeline == other.pipeline &&
               scheduler == other.scheduler;
    }

    std::string to_string() const
    {
        std::string op_str;
        switch(op)
        {
        case ConvOp::Forward: op_str = "fwd"; break;
        case ConvOp::BackwardData: op_str = "bwdd"; break;
        case ConvOp::BackwardWeight: op_str = "bwdw"; break;
        }
        return "conv_" + op_str + "_" + dtype_in + "_" + std::to_string(ndim_spatial) + "d_" +
               std::to_string(tile_m) + "x" + std::to_string(tile_n) + "x" + std::to_string(tile_k);
    }
};

struct ConvKernelKeyHash
{
    std::size_t operator()(const ConvKernelKey& key) const
    {
        std::size_t h = std::hash<std::string>{}(key.dtype_in);
        h ^= std::hash<std::string>{}(key.layout) << 1;
        h ^= std::hash<int>{}(key.ndim_spatial) << 2;
        h ^= std::hash<int>{}(static_cast<int>(key.op)) << 3;
        h ^= std::hash<int>{}(key.tile_m) << 4;
        h ^= std::hash<int>{}(key.tile_n) << 5;
        h ^= std::hash<int>{}(key.tile_k) << 6;
        return h;
    }
};

// =============================================================================
// ConvKernelInstance - Runtime representation of a kernel
// =============================================================================

class ConvKernelInstance
{
    public:
    using RunFn = std::function<float(const ConvProblem&, void*)>;

    ConvKernelInstance(const ConvKernelKey& key, const std::string& name, RunFn run_fn)
        : key_(key), name_(name), run_fn_(std::move(run_fn))
    {
    }

    const ConvKernelKey& key() const { return key_; }
    const std::string& name() const { return name_; }

    float run(const ConvProblem& problem, void* stream = nullptr) const
    {
        return run_fn_(problem, stream);
    }

    bool matches(const ConvProblem& problem) const
    {
        // Check if this kernel can handle the problem
        return problem.op == key_.op;
    }

    private:
    ConvKernelKey key_;
    std::string name_;
    RunFn run_fn_;
};

// =============================================================================
// ConvRegistry - Stores and manages convolution kernels
// =============================================================================

class ConvRegistry
{
    public:
    enum class Priority
    {
        Low    = 0,
        Normal = 1,
        High   = 2
    };

    ConvRegistry() = default;

    void set_name(const std::string& name) { name_ = name; }
    const std::string& name() const { return name_; }

    /// Register a kernel instance
    bool register_kernel(std::shared_ptr<ConvKernelInstance> kernel,
                         Priority priority = Priority::Normal)
    {
        const auto& key  = kernel->key();
        kernels_[key]    = kernel;
        priorities_[key] = priority;
        return true;
    }

    /// Register kernels from a ConvKernelSet
    bool register_set(const ConvKernelSet& kernel_set, Priority priority = Priority::Normal)
    {
        for(const auto& decl : kernel_set.declarations())
        {
            // Create kernel instance from declaration
            ConvKernelKey key;
            key.dtype_in     = decl.signature.dtype_in_;
            key.dtype_wei    = decl.signature.dtype_wei_;
            key.dtype_out    = decl.signature.dtype_out_;
            key.layout       = decl.signature.layout_;
            key.ndim_spatial = decl.signature.num_dims_;
            key.op           = (decl.signature.conv_op_ == "forward")    ? ConvOp::Forward
                               : (decl.signature.conv_op_ == "bwd_data") ? ConvOp::BackwardData
                                                                         : ConvOp::BackwardWeight;
            key.tile_m       = 128; // Default, would come from algorithm
            key.tile_n       = decl.algorithm.tile_k_;
            key.tile_k       = decl.algorithm.tile_c_;
            key.pipeline     = decl.algorithm.pipeline_;
            key.scheduler    = decl.algorithm.scheduler_;

            auto instance = std::make_shared<ConvKernelInstance>(
                key,
                decl.name(),
                [](const ConvProblem&, void*) -> float { return 0.0f; } // Placeholder
            );
            register_kernel(instance, priority);
        }
        return true;
    }

    /// Find the best kernel for a problem
    const ConvKernelInstance* find(const ConvProblem& problem) const
    {
        const ConvKernelInstance* best = nullptr;
        Priority best_priority         = Priority::Low;

        for(const auto& [key, kernel] : kernels_)
        {
            if(kernel->matches(problem))
            {
                auto it           = priorities_.find(key);
                Priority priority = (it != priorities_.end()) ? it->second : Priority::Normal;
                if(!best || priority > best_priority)
                {
                    best          = kernel.get();
                    best_priority = priority;
                }
            }
        }

        return best;
    }

    /// Get all registered kernels
    std::vector<const ConvKernelInstance*> all_kernels() const
    {
        std::vector<const ConvKernelInstance*> result;
        for(const auto& [key, kernel] : kernels_)
        {
            result.push_back(kernel.get());
        }
        return result;
    }

    size_t size() const { return kernels_.size(); }
    bool empty() const { return kernels_.empty(); }

    void clear()
    {
        kernels_.clear();
        priorities_.clear();
    }

    private:
    std::string name_ = "default";
    std::unordered_map<ConvKernelKey, std::shared_ptr<ConvKernelInstance>, ConvKernelKeyHash>
        kernels_;
    std::unordered_map<ConvKernelKey, Priority, ConvKernelKeyHash> priorities_;
};

// =============================================================================
// ConvDispatcher - Selects and runs the best kernel for a problem
// =============================================================================

class ConvDispatcher
{
    public:
    explicit ConvDispatcher(ConvRegistry* registry) : registry_(registry) {}

    /// Run convolution with automatic kernel selection
    float run(const ConvProblem& problem, void* stream = nullptr)
    {
        const auto* kernel = registry_->find(problem);
        if(!kernel)
        {
            throw std::runtime_error("No suitable convolution kernel found for problem: " +
                                     problem.to_string());
        }
        return kernel->run(problem, stream);
    }

    /// Get the kernel that would be selected for a problem
    const ConvKernelInstance* select(const ConvProblem& problem) const
    {
        return registry_->find(problem);
    }

    private:
    ConvRegistry* registry_;
};

} // namespace dispatcher
} // namespace ck_tile
