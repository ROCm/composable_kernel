// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file grouped_conv_registry.hpp
 * @brief Grouped Convolution kernel registry and dispatcher
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>

#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_kernel_decl.hpp"

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// Thread-local buffer context for GroupedConvDispatcher::run()
// The generated conv backend RunFn reads these to get buffer pointers.
// =============================================================================

struct ConvDispatchBuffers
{
    const void* input_ptr  = nullptr;
    const void* weight_ptr = nullptr;
    void* output_ptr       = nullptr;
    int warmup             = 3;
    int repeat             = 10;
    bool benchmarking      = true;
    int split_k            = 1;
};

inline thread_local ConvDispatchBuffers g_conv_dispatch_buffers;

// =============================================================================
// GroupedConvKernelKey - Unique identifier for a grouped convolution kernel
// =============================================================================

struct GroupedConvKernelKey
{
    // Signature fields
    std::string dtype_in;
    std::string dtype_wei;
    std::string dtype_out;
    std::string layout;   // e.g., "nhwgc"
    int ndim_spatial = 2; // 1, 2, or 3
    GroupedConvOp op = GroupedConvOp::Forward;

    // Tile configuration
    int tile_m = 1;
    int tile_n = 128;
    int tile_k = 128;

    // Wave/warp configuration
    int wave_m = 2;
    int wave_n = 2;
    int wave_k = 1;
    int warp_m = 32;
    int warp_n = 32;
    int warp_k = 16;

    // Pipeline
    std::string pipeline  = "compv3";
    std::string scheduler = "intrawave";
    std::string epilogue  = "cshuffle";

    // ConvConfigBase parity fields
    int vector_size_a       = 4;
    int vector_size_b       = 8;
    int vector_size_c       = 8;
    int block_per_cu        = 1;
    int num_wave_groups     = 1;
    int num_groups_to_merge = 1;

    // GPU architecture (for filter_by_arch)
    std::string arch = "gfx942";

    bool operator==(const GroupedConvKernelKey& other) const
    {
        return dtype_in == other.dtype_in && dtype_wei == other.dtype_wei &&
               dtype_out == other.dtype_out && layout == other.layout &&
               ndim_spatial == other.ndim_spatial && op == other.op && tile_m == other.tile_m &&
               tile_n == other.tile_n && tile_k == other.tile_k && wave_m == other.wave_m &&
               wave_n == other.wave_n && wave_k == other.wave_k && warp_m == other.warp_m &&
               warp_n == other.warp_n && warp_k == other.warp_k && pipeline == other.pipeline &&
               scheduler == other.scheduler && epilogue == other.epilogue &&
               vector_size_a == other.vector_size_a && vector_size_b == other.vector_size_b &&
               vector_size_c == other.vector_size_c && block_per_cu == other.block_per_cu &&
               num_wave_groups == other.num_wave_groups &&
               num_groups_to_merge == other.num_groups_to_merge && arch == other.arch;
    }

    std::string to_string() const
    {
        std::string op_str;
        switch(op)
        {
        case GroupedConvOp::Forward: op_str = "fwd"; break;
        case GroupedConvOp::BackwardData: op_str = "bwd_data"; break;
        case GroupedConvOp::BackwardWeight: op_str = "bwd_weight"; break;
        }
        return "grouped_conv_" + op_str + "_" + dtype_in + "_" + std::to_string(ndim_spatial) +
               "d_" + std::to_string(tile_m) + "x" + std::to_string(tile_n) + "x" +
               std::to_string(tile_k) + "_" + std::to_string(wave_m) + "x" +
               std::to_string(wave_n) + "x" + std::to_string(wave_k) + "_" +
               std::to_string(warp_m) + "x" + std::to_string(warp_n) + "x" +
               std::to_string(warp_k) + "_" + pipeline;
    }
};

struct GroupedConvKernelKeyHash
{
    std::size_t operator()(const GroupedConvKernelKey& key) const
    {
        std::size_t h = std::hash<std::string>{}(key.dtype_in);
        h ^= std::hash<std::string>{}(key.layout) << 1;
        h ^= std::hash<int>{}(key.ndim_spatial) << 2;
        h ^= std::hash<int>{}(static_cast<int>(key.op)) << 3;
        h ^= std::hash<int>{}(key.tile_m) << 4;
        h ^= std::hash<int>{}(key.tile_n) << 5;
        h ^= std::hash<int>{}(key.tile_k) << 6;
        h ^= std::hash<int>{}(key.wave_m) << 7;
        h ^= std::hash<int>{}(key.wave_n) << 8;
        h ^= std::hash<int>{}(key.warp_m) << 9;
        h ^= std::hash<int>{}(key.warp_n) << 10;
        h ^= std::hash<std::string>{}(key.pipeline) << 11;
        h ^= std::hash<std::string>{}(key.arch) << 12;
        return h;
    }
};

// =============================================================================
// GroupedConvKernelInstance - Runtime representation of a kernel
// =============================================================================

// Forward declaration for shared_ptr type alias
class GroupedConvKernelInstance;
using GroupedConvKernelInstancePtr = std::shared_ptr<GroupedConvKernelInstance>;

class GroupedConvKernelInstance
{
    public:
    using RunFn = std::function<float(const GroupedConvProblem&, void*)>;

    GroupedConvKernelInstance(const GroupedConvKernelKey& key,
                              const std::string& name,
                              RunFn run_fn)
        : key_(key), name_(name), run_fn_(std::move(run_fn))
    {
    }

    const GroupedConvKernelKey& key() const { return key_; }
    const std::string& name() const { return name_; }

    float run(const GroupedConvProblem& problem, void* stream = nullptr) const
    {
        return run_fn_(problem, stream);
    }

    bool matches(const GroupedConvProblem& problem) const
    {
        // Check if this kernel can handle the problem
        return problem.op == key_.op;
    }

    private:
    GroupedConvKernelKey key_;
    std::string name_;
    RunFn run_fn_;
};

// =============================================================================
// GroupedConvRegistry - Stores and manages grouped convolution kernels
// =============================================================================

class GroupedConvRegistry : public BaseRegistry<GroupedConvRegistry,
                                                GroupedConvKernelKey,
                                                GroupedConvKernelInstance,
                                                GroupedConvKernelKeyHash>
{
    using Base = BaseRegistry<GroupedConvRegistry,
                              GroupedConvKernelKey,
                              GroupedConvKernelInstance,
                              GroupedConvKernelKeyHash>;

    public:
    GroupedConvRegistry() = default;

    /// Singleton instance for global kernel registration
    static GroupedConvRegistry& instance()
    {
        static GroupedConvRegistry registry;
        return registry;
    }

    /// Register kernels from a GroupedConvKernelSet (atomic batch registration)
    bool register_set(const GroupedConvKernelSet& kernel_set, Priority priority = Priority::Normal)
    {
        // Build all instances first, then register under a single lock hold
        // so readers never see a half-registered set.
        std::vector<std::pair<GroupedConvKernelKey, std::shared_ptr<GroupedConvKernelInstance>>>
            batch;
        batch.reserve(kernel_set.declarations().size());

        for(const auto& decl : kernel_set.declarations())
        {
            GroupedConvKernelKey key;
            key.dtype_in        = decl.signature.dtype_in_;
            key.dtype_wei       = decl.signature.dtype_wei_;
            key.dtype_out       = decl.signature.dtype_out_;
            key.layout          = decl.signature.layout_;
            key.ndim_spatial    = decl.signature.num_dims_;
            key.op              = (decl.signature.conv_op_ == "forward") ? GroupedConvOp::Forward
                                  : (decl.signature.conv_op_ == "bwd_data") ? GroupedConvOp::BackwardData
                                                                            : GroupedConvOp::BackwardWeight;
            key.tile_m          = decl.algorithm.tile_m_;
            key.tile_n          = decl.algorithm.tile_n_;
            key.tile_k          = decl.algorithm.tile_k_;
            key.wave_m          = decl.algorithm.wave_m_;
            key.wave_n          = decl.algorithm.wave_n_;
            key.wave_k          = decl.algorithm.wave_k_;
            key.warp_m          = decl.algorithm.warp_m_;
            key.warp_n          = decl.algorithm.warp_n_;
            key.warp_k          = decl.algorithm.warp_k_;
            key.pipeline        = decl.algorithm.pipeline_;
            key.scheduler       = decl.algorithm.scheduler_;
            key.epilogue        = decl.algorithm.epilogue_;
            key.vector_size_a   = decl.algorithm.vector_a_;
            key.vector_size_b   = decl.algorithm.vector_b_;
            key.vector_size_c   = decl.algorithm.vector_c_;
            key.block_per_cu    = decl.algorithm.block_per_cu_;
            key.num_wave_groups = decl.algorithm.num_wave_groups_;
            key.num_groups_to_merge = decl.algorithm.num_groups_to_merge_;
            key.arch                = decl.arch;

            batch.emplace_back(key,
                               std::make_shared<GroupedConvKernelInstance>(
                                   key, decl.name(), [](const GroupedConvProblem&, void*) -> float {
                                       return 0.0f;
                                   }));
        }

        std::lock_guard<std::mutex> lock(mutex());
        bool any_registered = false;
        for(auto& [key, instance] : batch)
        {
            auto it = entries().find(key);
            if(it == entries().end() || it->second.priority <= priority)
            {
                entries_mut()[key] = typename Base::Entry{std::move(instance), priority};
                any_registered     = true;
            }
        }
        return any_registered;
    }

    /// Find the best kernel for a problem
    const GroupedConvKernelInstance* find(const GroupedConvProblem& problem) const
    {
        std::lock_guard<std::mutex> lock(mutex());
        const GroupedConvKernelInstance* best = nullptr;
        Priority best_priority                = Priority::Low;

        for(const auto& [key, entry] : entries())
        {
            if(entry.instance->matches(problem))
            {
                if(!best || entry.priority > best_priority)
                {
                    best          = entry.instance.get();
                    best_priority = entry.priority;
                }
            }
        }

        return best;
    }

    /// Get all registered kernels
    std::vector<const GroupedConvKernelInstance*> all_kernels() const
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::vector<const GroupedConvKernelInstance*> result;
        for(const auto& [key, entry] : entries())
        {
            result.push_back(entry.instance.get());
        }
        return result;
    }

    /// Export registry to JSON string
    std::string export_json(bool include_statistics = false) const
    {
        // Note: get_name() acquires the mutex internally, so we must NOT hold
        // the registry mutex here (std::mutex is not recursive).
        std::string reg_name = get_name();

        std::lock_guard<std::mutex> lock(mutex());
        std::ostringstream json;

        json << "{\n";
        json << "  \"metadata\": {\n";
        json << "    \"registry_name\": \"" << json_escape(reg_name) << "\",\n";
        json << "    \"total_kernels\": " << entries().size() << "\n";
        json << "  }";

        if(include_statistics && !entries().empty())
        {
            std::map<std::string, int> by_datatype;
            std::map<std::string, int> by_pipeline;
            std::map<std::string, int> by_arch;

            for(const auto& [key, entry] : entries())
            {
                std::string dtype_key = key.dtype_in + "_" + key.dtype_wei + "_" + key.dtype_out;
                by_datatype[dtype_key]++;
                by_pipeline[key.pipeline]++;
                by_arch[key.arch]++;
            }

            json << ",\n  \"statistics\": {\n";
            json << "    \"by_datatype\": {";
            bool first = true;
            for(const auto& [dtype, count] : by_datatype)
            {
                if(!first)
                    json << ",";
                json << "\"" << json_escape(dtype) << "\":" << count;
                first = false;
            }
            json << "},\n";
            json << "    \"by_pipeline\": {";
            first = true;
            for(const auto& [pipeline, count] : by_pipeline)
            {
                if(!first)
                    json << ",";
                json << "\"" << json_escape(pipeline) << "\":" << count;
                first = false;
            }
            json << "},\n";
            json << "    \"by_arch\": {";
            first = true;
            for(const auto& [arch, count] : by_arch)
            {
                if(!first)
                    json << ",";
                json << "\"" << json_escape(arch) << "\":" << count;
                first = false;
            }
            json << "}\n  }";
        }

        json << ",\n  \"kernels\": [\n";
        bool first = true;
        for(const auto& [key, entry] : entries())
        {
            if(!first)
                json << ",\n";
            json << "    " << export_kernel_json(*entry.instance);
            first = false;
        }
        json << "\n  ]\n";
        json << "}\n";

        return json.str();
    }

    /// Export registry to JSON file
    void export_json_to_file(const std::string& filename, bool include_statistics = false) const
    {
        std::string json_str = export_json(include_statistics);
        std::ofstream file(filename);
        if(!file.is_open())
        {
            throw std::runtime_error("Failed to open file for export: " + filename);
        }
        file << json_str;
    }

    /// Get kernels matching a predicate
    std::vector<const GroupedConvKernelInstance*>
    filter(std::function<bool(const GroupedConvKernelInstance&)> predicate) const
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::vector<const GroupedConvKernelInstance*> result;
        for(const auto& [key, entry] : entries())
        {
            if(predicate(*entry.instance))
            {
                result.push_back(entry.instance.get());
            }
        }
        return result;
    }

    /// Remove kernels not matching the arch
    std::size_t filter_by_arch(const std::string& gpu_arch)
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::vector<GroupedConvKernelKey> to_remove;
        for(const auto& [key, entry] : entries())
        {
            if(key.arch != gpu_arch)
            {
                to_remove.push_back(key);
            }
        }
        for(const auto& key : to_remove)
        {
            entries_mut().erase(key);
        }
        return to_remove.size();
    }

    private:
    static std::string json_escape(const std::string& str)
    {
        std::ostringstream oss;
        for(char c : str)
        {
            switch(c)
            {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if(c < 0x20)
                {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                }
                else
                {
                    oss << c;
                }
            }
        }
        return oss.str();
    }

    static std::string export_kernel_json(const GroupedConvKernelInstance& kernel)
    {
        std::ostringstream json;
        const auto& key = kernel.key();

        std::string op_str;
        switch(key.op)
        {
        case GroupedConvOp::Forward: op_str = "fwd"; break;
        case GroupedConvOp::BackwardData: op_str = "bwd_data"; break;
        case GroupedConvOp::BackwardWeight: op_str = "bwd_weight"; break;
        }

        json << "{\n";
        json << "      \"name\": \"" << json_escape(kernel.name()) << "\",\n";
        json << "      \"signature\": {\n";
        json << "        \"dtype_in\": \"" << json_escape(key.dtype_in) << "\",\n";
        json << "        \"dtype_wei\": \"" << json_escape(key.dtype_wei) << "\",\n";
        json << "        \"dtype_out\": \"" << json_escape(key.dtype_out) << "\",\n";
        json << "        \"layout\": \"" << json_escape(key.layout) << "\",\n";
        json << "        \"ndim_spatial\": " << key.ndim_spatial << ",\n";
        json << "        \"op\": \"" << op_str << "\"\n";
        json << "      },\n";
        json << "      \"algorithm\": {\n";
        json << "        \"tile_m\": " << key.tile_m << ",\n";
        json << "        \"tile_n\": " << key.tile_n << ",\n";
        json << "        \"tile_k\": " << key.tile_k << ",\n";
        json << "        \"wave\": \"" << key.wave_m << "x" << key.wave_n << "x" << key.wave_k
             << "\",\n";
        json << "        \"warp\": \"" << key.warp_m << "x" << key.warp_n << "x" << key.warp_k
             << "\",\n";
        json << "        \"pipeline\": \"" << json_escape(key.pipeline) << "\",\n";
        json << "        \"scheduler\": \"" << json_escape(key.scheduler) << "\",\n";
        json << "        \"epilogue\": \"" << json_escape(key.epilogue) << "\",\n";
        json << "        \"vector_sizes\": [" << key.vector_size_a << "," << key.vector_size_b
             << "," << key.vector_size_c << "],\n";
        json << "        \"block_per_cu\": " << key.block_per_cu << ",\n";
        json << "        \"num_wave_groups\": " << key.num_wave_groups << ",\n";
        json << "        \"num_groups_to_merge\": " << key.num_groups_to_merge << "\n";
        json << "      },\n";
        json << "      \"arch\": \"" << json_escape(key.arch) << "\"\n";
        json << "    }";

        return json.str();
    }
};

// =============================================================================
// GroupedConvDispatcher - Selects and runs the best kernel for a problem
// =============================================================================

class GroupedConvDispatcher
{
    public:
    enum class SelectionStrategy
    {
        PriorityBased,
        Heuristic
    };

    using HeuristicFunction = std::function<std::vector<std::string>(const GroupedConvProblem&)>;

    explicit GroupedConvDispatcher(GroupedConvRegistry* registry)
        : registry_(registry), strategy_(SelectionStrategy::PriorityBased)
    {
    }

    void set_strategy(SelectionStrategy s) { strategy_ = s; }
    void set_heuristic(HeuristicFunction fn) { heuristic_ = std::move(fn); }

    /// Select the best kernel for a problem (does not run it)
    const GroupedConvKernelInstance* select_kernel(const GroupedConvProblem& problem) const
    {
        if(strategy_ == SelectionStrategy::Heuristic)
            return select_heuristic(problem);
        return registry_->find(problem);
    }

    /// Run convolution with automatic kernel selection (legacy - no buffers)
    float run(const GroupedConvProblem& problem, void* stream = nullptr)
    {
        const auto* kernel = select_kernel(problem);
        if(!kernel)
        {
            throw NoKernelFound("No suitable grouped convolution kernel found for problem: " +
                                problem.to_string());
        }
        return kernel->run(problem, stream);
    }

    /// Run convolution with buffer pointers and automatic kernel selection.
    /// Sets the thread-local buffer context before dispatching to the kernel.
    float run(const void* input_ptr,
              const void* weight_ptr,
              void* output_ptr,
              const GroupedConvProblem& problem,
              void* stream = nullptr,
              int warmup   = 3,
              int repeat   = 10)
    {
        const auto* kernel = select_kernel(problem);
        if(!kernel)
        {
            throw NoKernelFound("No suitable grouped convolution kernel found for problem: " +
                                problem.to_string());
        }
        g_conv_dispatch_buffers.input_ptr    = input_ptr;
        g_conv_dispatch_buffers.weight_ptr   = weight_ptr;
        g_conv_dispatch_buffers.output_ptr   = output_ptr;
        g_conv_dispatch_buffers.warmup       = warmup;
        g_conv_dispatch_buffers.repeat       = repeat;
        g_conv_dispatch_buffers.benchmarking = benchmarking_;
        g_conv_dispatch_buffers.split_k      = problem.split_k;
        return kernel->run(problem, stream);
    }

    /// Enable or disable GPU benchmarking (timing).
    /// When disabled, kernels execute once with no timing overhead.
    void set_benchmarking(bool enable) { benchmarking_ = enable; }
    [[nodiscard]] bool benchmarking_enabled() const { return benchmarking_; }

    /// Alias kept for backward compatibility
    const GroupedConvKernelInstance* select(const GroupedConvProblem& problem) const
    {
        return select_kernel(problem);
    }

    private:
    const GroupedConvKernelInstance* select_heuristic(const GroupedConvProblem& problem) const
    {
        if(!heuristic_)
            return registry_->find(problem);

        auto ranked_names = heuristic_(problem);
        auto all          = registry_->all_kernels();
        for(const auto& name : ranked_names)
        {
            for(const auto* kernel : all)
            {
                if(kernel->name().find(name) != std::string::npos && kernel->matches(problem))
                {
                    return kernel;
                }
            }
        }
        return registry_->find(problem);
    }

    GroupedConvRegistry* registry_;
    SelectionStrategy strategy_;
    HeuristicFunction heuristic_;
    bool benchmarking_ = true;
};

} // namespace dispatcher
} // namespace ck_tile
