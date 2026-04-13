// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file grouped_conv_kernel_decl.hpp
 * @brief Declarative grouped convolution kernel specification
 *
 * USAGE:
 * ======
 *
 * // Named kernel sets for grouped convolution
 * DECL_GROUPED_CONV_KERNEL_SET(gconv_fwd,
 *     .add("fp16", "nhwc", "forward", 128, 128, 32)
 *     .add("fp16", "nhwc", "forward", 256, 256, 64)
 * );
 *
 * // Access at runtime
 * auto& set = GroupedConvKernelSetRegistry::instance().get("gconv_fwd");
 */

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <sstream>

namespace ck_tile {
namespace dispatcher {
namespace grouped_conv_decl {

// =============================================================================
// Wildcard constants
// =============================================================================

constexpr const char* ANY = "*";
constexpr int ANY_INT     = -1;

// =============================================================================
// GroupedConvSignature - WHAT operation
// =============================================================================

class GroupedConvSignature
{
    public:
    std::string dtype_in_        = "fp16";    // Input data type
    std::string dtype_wei_       = "fp16";    // Weight data type
    std::string dtype_out_       = "fp16";    // Output data type
    std::string dtype_acc_       = "fp32";    // Accumulator type
    std::string dtype_workspace_ = "fp32";    // Workspace type (two-stage algorithms)
    std::string dtype_bias_      = "fp16";    // Bias type (bias epilogue)
    std::string layout_          = "nhwc";    // Data layout: nhwc, nchw
    std::string conv_op_         = "forward"; // forward, bwd_data, bwd_weight
    int num_dims_                = 2;         // Spatial dimensions: 1, 2, or 3
    int groups_                  = 1;         // Group grouped convolution
    std::string specialization_  = "default"; // Filter specialization

    GroupedConvSignature& dtype(const std::string& in,
                                const std::string& wei,
                                const std::string& out,
                                const std::string& acc = "fp32")
    {
        dtype_in_  = in;
        dtype_wei_ = wei;
        dtype_out_ = out;
        dtype_acc_ = acc;
        return *this;
    }

    GroupedConvSignature& dtype(const std::string& all)
    {
        dtype_in_ = dtype_wei_ = dtype_out_ = dtype_bias_ = all;
        dtype_acc_ = dtype_workspace_ = "fp32";
        return *this;
    }

    GroupedConvSignature& dtype_workspace(const std::string& ws)
    {
        dtype_workspace_ = ws;
        return *this;
    }

    GroupedConvSignature& dtype_bias(const std::string& b)
    {
        dtype_bias_ = b;
        return *this;
    }

    GroupedConvSignature& layout(const std::string& l)
    {
        layout_ = l;
        return *this;
    }
    GroupedConvSignature& conv_type(const std::string& op)
    {
        conv_op_ = op;
        return *this;
    }
    GroupedConvSignature& dims(int d)
    {
        num_dims_ = d;
        return *this;
    }
    GroupedConvSignature& groups(int g)
    {
        groups_ = g;
        return *this;
    }
    GroupedConvSignature& spec(const std::string& s)
    {
        specialization_ = s;
        return *this;
    }

    std::string op_str() const
    {
        if(conv_op_ == "forward")
            return "fwd";
        if(conv_op_ == "bwd_data")
            return "bwd_data";
        if(conv_op_ == "bwd_weight")
            return "bwd_weight";
        return conv_op_;
    }
};

// =============================================================================
// GroupedConvAlgorithm - HOW it's implemented
// =============================================================================

class GroupedConvAlgorithm
{
    public:
    // Tile shape (M, N, K per tile - M=spatial*N, N=K_out, K=C_in)
    int tile_m_ = 1;   // Tile M (output spatial * batch)
    int tile_n_ = 128; // Tile N (output channels K)
    int tile_k_ = 128; // Tile K (input channels C)

    // Output spatial tile
    int tile_ho_ = 1;
    int tile_wo_ = 16;

    // Wave/warp shape
    int wave_m_ = ANY_INT;
    int wave_n_ = ANY_INT;
    int wave_k_ = 1;
    int warp_m_ = ANY_INT;
    int warp_n_ = ANY_INT;
    int warp_k_ = 16;

    // Vector sizes
    int vector_a_ = 4; // Input vector size
    int vector_b_ = 8; // Weight vector size
    int vector_c_ = 8; // Output vector size

    // Pipeline configuration
    std::string pipeline_  = "compv4";
    std::string scheduler_ = "intrawave";
    std::string epilogue_  = "cshuffle";
    std::string memory_op_ = "set"; // Memory operation: set, atomic_add, atomic_max, add

    // Occupancy/performance hints
    int block_size_          = 256;
    int block_per_cu_        = 1;
    int num_wave_groups_     = 1;
    int num_groups_to_merge_ = 1;
    bool double_smem_buffer_ = false;

    // Padding -- always enabled for convolution (MNK padding assumed)
    static constexpr bool pad_m_ = true;
    static constexpr bool pad_n_ = true;
    static constexpr bool pad_k_ = true;

    // Tile setter (M, N, K)
    GroupedConvAlgorithm& tile(int m, int n, int k)
    {
        tile_m_ = m;
        tile_n_ = n;
        tile_k_ = k;
        return *this;
    }

    GroupedConvAlgorithm& tile_output(int ho, int wo)
    {
        tile_ho_ = ho;
        tile_wo_ = wo;
        return *this;
    }

    GroupedConvAlgorithm& wave(int m, int n, int k = 1)
    {
        wave_m_ = m;
        wave_n_ = n;
        wave_k_ = k;
        return *this;
    }

    GroupedConvAlgorithm& warp(int m, int n, int k = 16)
    {
        warp_m_ = m;
        warp_n_ = n;
        warp_k_ = k;
        return *this;
    }

    GroupedConvAlgorithm& vector_sizes(int a, int b, int c)
    {
        vector_a_ = a;
        vector_b_ = b;
        vector_c_ = c;
        return *this;
    }

    GroupedConvAlgorithm& pipeline(const std::string& p)
    {
        pipeline_ = p;
        return *this;
    }
    GroupedConvAlgorithm& scheduler(const std::string& s)
    {
        scheduler_ = s;
        return *this;
    }
    GroupedConvAlgorithm& epilogue(const std::string& e)
    {
        epilogue_ = e;
        return *this;
    }
    GroupedConvAlgorithm& memory_op(const std::string& m)
    {
        memory_op_ = m;
        return *this;
    }

    // Occupancy setters
    GroupedConvAlgorithm& block_per_cu(int b)
    {
        block_per_cu_ = b;
        return *this;
    }
    GroupedConvAlgorithm& num_wave_groups(int n)
    {
        num_wave_groups_ = n;
        return *this;
    }
    GroupedConvAlgorithm& num_groups_to_merge(int n)
    {
        num_groups_to_merge_ = n;
        return *this;
    }
    GroupedConvAlgorithm& double_smem_buffer(bool d)
    {
        double_smem_buffer_ = d;
        return *this;
    }

    bool needs_expansion() const
    {
        return wave_m_ == ANY_INT || warp_m_ == ANY_INT || pipeline_ == "*" || scheduler_ == "*";
    }

    /// Check if specific parameter needs expansion
    bool needs_wave_expansion() const { return wave_m_ == ANY_INT || wave_n_ == ANY_INT; }
    bool needs_warp_expansion() const { return warp_m_ == ANY_INT || warp_n_ == ANY_INT; }
    bool needs_pipeline_expansion() const { return pipeline_ == "*"; }
    bool needs_scheduler_expansion() const { return scheduler_ == "*"; }

    /// Auto-fill with defaults (for single kernel generation)
    void auto_fill()
    {
        if(wave_m_ == ANY_INT)
            wave_m_ = 2;
        if(wave_n_ == ANY_INT)
            wave_n_ = 2;
        if(warp_m_ == ANY_INT)
            warp_m_ = 32;
        if(warp_n_ == ANY_INT)
            warp_n_ = 32;
        if(pipeline_ == "*")
            pipeline_ = "compv4";
        if(scheduler_ == "*")
            scheduler_ = "intrawave";
    }

    /// Get all valid wave configurations for arch
    static std::vector<std::tuple<int, int, int>> valid_wave_configs(const std::string& arch)
    {
        // Match arch_specs_generated.py WARP_SUPPORTED_COMBINATIONS
        if(arch == "gfx942" || arch == "gfx90a" || arch == "gfx950")
        {
            return {{1, 4, 1}, {2, 2, 1}, {4, 1, 1}};
        }
        return {{2, 2, 1}}; // Default
    }

    /// Get all valid warp tile configurations
    static std::vector<std::tuple<int, int, int>> valid_warp_configs(const std::string& arch,
                                                                     const std::string& dtype)
    {
        // Match arch_specs_generated.py WARP_TILE_SUPPORTED_COMBINATIONS
        if(arch == "gfx942" && (dtype == "fp16" || dtype == "bf16"))
        {
            return {{16, 16, 16}, {32, 32, 16}};
        }
        return {{32, 32, 16}}; // Default
    }

    /// Get all valid pipeline/scheduler combinations for forward conv.
    /// Backward operations (bwd_data/bwd_weight) only support compv3 and mem
    /// due to transpose_tile2d and get_length constraints in CK Tile.
    static std::vector<std::pair<std::string, std::string>> valid_trait_configs()
    {
        return {
            {"compv3", "intrawave"},
            {"compv4", "intrawave"},
            {"compv5", "intrawave"},
            {"mem", "intrawave"},
            {"mem", "interwave"},
        };
    }
};

// =============================================================================
// GroupedConvKernelDecl
// =============================================================================

struct GroupedConvKernelDecl
{
    GroupedConvSignature signature;
    GroupedConvAlgorithm algorithm;
    std::string arch = "gfx942";

    GroupedConvKernelDecl() = default;

    GroupedConvKernelDecl(const GroupedConvSignature& sig,
                          const GroupedConvAlgorithm& algo,
                          const std::string& a = "gfx942")
        : signature(sig), algorithm(algo), arch(a)
    {
    }

    std::string name() const
    {
        std::ostringstream oss;
        // Generate full kernel name similar to GEMM:
        // grouped_conv_<op>_<dtype>_<layout>_<ndim>d_<pipeline>_<epilogue>_<scheduler>_<tile>_<wave>_<warp>
        oss << "grouped_conv_" << signature.op_str() << "_" << signature.dtype_in_ << "_"
            << signature.layout_ << "_" << signature.num_dims_ << "d" << "_" << algorithm.pipeline_
            << "_" << algorithm.epilogue_ << "_" << algorithm.scheduler_ << "_" << algorithm.tile_m_
            << "x" << algorithm.tile_n_ << "x" << algorithm.tile_k_ << "_" << algorithm.wave_m_
            << "x" << algorithm.wave_n_ << "x" << algorithm.wave_k_ << "_" << algorithm.warp_m_
            << "x" << algorithm.warp_n_ << "x" << algorithm.warp_k_;
        return oss.str();
    }

    bool has_wildcards() const { return algorithm.needs_expansion() || arch == "*"; }
};

// =============================================================================
// GroupedConvKernelSet
// =============================================================================

class GroupedConvKernelSet
{
    public:
    GroupedConvKernelSet() = default;

    GroupedConvKernelSet& add(const GroupedConvSignature& sig,
                              const GroupedConvAlgorithm& algo,
                              const std::string& arch = "gfx942")
    {
        decls_.emplace_back(sig, algo, arch);
        return *this;
    }

    // Simple add: dtype, layout, conv_type, tile_k, tile_c
    GroupedConvKernelSet& add(const std::string& dtype,
                              const std::string& layout,
                              const std::string& conv_type,
                              int tile_k,
                              int tile_c,
                              const std::string& arch = "gfx942")
    {
        GroupedConvSignature sig;
        sig.dtype(dtype).layout(layout).conv_type(conv_type);
        GroupedConvAlgorithm algo;
        algo.tile(1, tile_k, tile_c);
        decls_.emplace_back(sig, algo, arch);
        return *this;
    }

    GroupedConvKernelSet& merge(const GroupedConvKernelSet& other)
    {
        decls_.insert(decls_.end(), other.decls_.begin(), other.decls_.end());
        return *this;
    }

    const std::vector<GroupedConvKernelDecl>& declarations() const { return decls_; }
    size_t size() const { return decls_.size(); }

    void print(std::ostream& os = std::cout) const
    {
        os << "GroupedConvKernelSet (" << size() << " declarations):\n";
        for(const auto& d : decls_)
        {
            os << "  - " << d.name();
            if(d.algorithm.needs_expansion())
                os << " [expands]";
            os << "\n";
        }
    }

    GroupedConvKernelSet& tag(const std::string& t)
    {
        tag_ = t;
        return *this;
    }
    std::string tag() const { return tag_; }

    private:
    std::vector<GroupedConvKernelDecl> decls_;
    std::string tag_;
};

// =============================================================================
// GroupedConvKernelSetRegistry
// =============================================================================

class GroupedConvKernelSetRegistry
{
    public:
    static GroupedConvKernelSetRegistry& instance()
    {
        static GroupedConvKernelSetRegistry reg;
        return reg;
    }

    void add(const std::string& name, const GroupedConvKernelSet& set)
    {
        sets_[name] = set;
        if(std::find(order_.begin(), order_.end(), name) == order_.end())
        {
            order_.push_back(name);
        }
    }

    // Alias for add() for consistency with GEMM API
    void register_set(const std::string& name, const GroupedConvKernelSet& set) { add(name, set); }

    const GroupedConvKernelSet& get(const std::string& name) const
    {
        static GroupedConvKernelSet empty;
        auto it = sets_.find(name);
        return it != sets_.end() ? it->second : empty;
    }

    bool has(const std::string& name) const { return sets_.find(name) != sets_.end(); }

    std::vector<std::string> names() const { return order_; }
    size_t size() const { return sets_.size(); }

    void clear()
    {
        sets_.clear();
        order_.clear();
    }

    void print() const
    {
        std::cout << "Grouped Conv Kernel Sets (" << size() << "):\n";
        for(const auto& name : order_)
        {
            const auto& set = sets_.at(name);
            std::cout << "  " << name << ": " << set.size() << " declarations\n";
        }
    }

    private:
    GroupedConvKernelSetRegistry() = default;
    std::unordered_map<std::string, GroupedConvKernelSet> sets_;
    std::vector<std::string> order_;
};

// =============================================================================
// Static Registrar
// =============================================================================

struct GroupedConvKernelSetRegistrar
{
    GroupedConvKernelSetRegistrar(const std::string& name, const GroupedConvKernelSet& set)
    {
        GroupedConvKernelSetRegistry::instance().add(name, set);
    }
};

} // namespace grouped_conv_decl

// Convenience aliases
using GroupedConvSignature         = grouped_conv_decl::GroupedConvSignature;
using GroupedConvAlgorithm         = grouped_conv_decl::GroupedConvAlgorithm;
using GroupedConvKernelDecl        = grouped_conv_decl::GroupedConvKernelDecl;
using GroupedConvKernelSet         = grouped_conv_decl::GroupedConvKernelSet;
using GroupedConvKernelSetRegistry = grouped_conv_decl::GroupedConvKernelSetRegistry;

} // namespace dispatcher
} // namespace ck_tile

// =============================================================================
// Declaration Macros
// =============================================================================

#define CK_GROUPED_CONV_DECL_CAT_(a, b) CK_GROUPED_CONV_DECL_CAT_IMPL_(a, b)
#define CK_GROUPED_CONV_DECL_CAT_IMPL_(a, b) a##b

// Note: __extension__ suppresses warnings about __COUNTER__ being a GCC/Clang extension
#define DECL_GROUPED_CONV_KERNEL_SET(name, ...)                                                  \
    __extension__ static ::ck_tile::dispatcher::grouped_conv_decl::GroupedConvKernelSetRegistrar \
    CK_GROUPED_CONV_DECL_CAT_(_gconv_kset_reg_, __COUNTER__)(                                    \
        #name,                                                                                   \
        ::ck_tile::dispatcher::grouped_conv_decl::GroupedConvKernelSet() __VA_ARGS__.tag(#name))

#define DECL_GROUPED_CONV_KERNEL_ALL(dtype, layout)                                                \
    __extension__ static ::ck_tile::dispatcher::grouped_conv_decl::GroupedConvKernelSetRegistrar   \
    CK_GROUPED_CONV_DECL_CAT_(_gconv_kset_reg_, __COUNTER__)(                                      \
        #dtype "_" #layout "_all",                                                                 \
        ::ck_tile::dispatcher::grouped_conv_decl::GroupedConvKernelSet().add(                      \
            ::ck_tile::dispatcher::grouped_conv_decl::GroupedConvSignature().dtype(#dtype).layout( \
                #layout),                                                                          \
            ::ck_tile::dispatcher::grouped_conv_decl::GroupedConvAlgorithm(),                      \
            "*"))

#define GROUPED_CONV_KERNEL_SET(name) \
    ::ck_tile::dispatcher::grouped_conv_decl::GroupedConvKernelSet name
#define BEGIN_GROUPED_CONV_KERNEL_SET() \
    ::ck_tile::dispatcher::grouped_conv_decl::GroupedConvKernelSet()
