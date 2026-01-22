// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file kernel_decl.hpp
 * @brief Declarative kernel specification with KernelSet
 *
 * USAGE:
 * ======
 *
 * // Named kernel sets
 * DECL_KERNEL_SET(compute_bound,
 *     .add("fp16", "rcr", 256, 256, 64)
 *     .add("fp16", "rcr", 128, 128, 32)
 * );
 *
 * // Access at runtime
 * auto& set = KernelSetRegistry::instance().get("compute_bound");
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>

namespace ck_tile {
namespace dispatcher {
namespace decl {

// =============================================================================
// Wildcard constants
// =============================================================================

constexpr const char* ANY = "*";
constexpr int ANY_INT     = -1;

// =============================================================================
// Signature Builder
// =============================================================================

class Signature
{
    public:
    std::string dtype_a_        = "fp16";
    std::string dtype_b_        = "fp16";
    std::string dtype_c_        = "fp16";
    std::string dtype_acc_      = "fp32";
    std::string layout_a_       = "row";
    std::string layout_b_       = "col";
    std::string layout_c_       = "row";
    std::string elementwise_op_ = "PassThrough";
    int num_d_tensors_          = 0;
    bool structured_sparsity_   = false;

    Signature& dtype(const std::string& a,
                     const std::string& b,
                     const std::string& c,
                     const std::string& acc = "fp32")
    {
        dtype_a_   = a;
        dtype_b_   = b;
        dtype_c_   = c;
        dtype_acc_ = acc;
        return *this;
    }

    Signature& dtype(const std::string& all)
    {
        dtype_a_ = dtype_b_ = dtype_c_ = all;
        dtype_acc_                     = "fp32";
        return *this;
    }

    Signature& layout(const std::string& a, const std::string& b, const std::string& c)
    {
        layout_a_ = a;
        layout_b_ = b;
        layout_c_ = c;
        return *this;
    }

    Signature& layout(const std::string& combined)
    {
        if(combined.size() >= 3)
        {
            layout_a_ = (combined[0] == 'r') ? "row" : "col";
            layout_b_ = (combined[1] == 'r') ? "row" : "col";
            layout_c_ = (combined[2] == 'r') ? "row" : "col";
        }
        return *this;
    }

    Signature& elementwise(const std::string& op, int num_d = 0)
    {
        elementwise_op_ = op;
        num_d_tensors_  = num_d;
        return *this;
    }

    std::string layout_str() const
    {
        std::string r;
        r += (layout_a_ == "col") ? 'c' : 'r';
        r += (layout_b_ == "col") ? 'c' : 'r';
        r += (layout_c_ == "col") ? 'c' : 'r';
        return r;
    }
};

// =============================================================================
// Algorithm Builder
// =============================================================================

class Algorithm
{
    public:
    int tile_m_ = 128, tile_n_ = 128, tile_k_ = 32;
    int wave_m_ = ANY_INT, wave_n_ = ANY_INT, wave_k_ = 1;
    int warp_m_ = ANY_INT, warp_n_ = ANY_INT, warp_k_ = 16;
    std::string pipeline_  = "compv4";
    std::string scheduler_ = "intrawave";
    std::string epilogue_  = "cshuffle";
    int block_size_        = 256;
    int pad_m_ = 1, pad_n_ = 1, pad_k_ = 1;
    bool preshuffle_ = false;

    Algorithm& tile(int m, int n, int k)
    {
        tile_m_ = m;
        tile_n_ = n;
        tile_k_ = k;
        return *this;
    }

    Algorithm& wave(int m, int n, int k = 1)
    {
        wave_m_ = m;
        wave_n_ = n;
        wave_k_ = k;
        return *this;
    }

    Algorithm& warp(int m, int n, int k = 16)
    {
        warp_m_ = m;
        warp_n_ = n;
        warp_k_ = k;
        return *this;
    }

    Algorithm& pipeline(const std::string& p)
    {
        pipeline_ = p;
        return *this;
    }
    Algorithm& scheduler(const std::string& s)
    {
        scheduler_ = s;
        return *this;
    }
    Algorithm& epilogue(const std::string& e)
    {
        epilogue_ = e;
        return *this;
    }

    Algorithm& pad(bool m, bool n, bool k)
    {
        pad_m_ = m ? 1 : 0;
        pad_n_ = n ? 1 : 0;
        pad_k_ = k ? 1 : 0;
        return *this;
    }

    Algorithm& preshuffle(bool v)
    {
        preshuffle_ = v;
        return *this;
    }

    bool needs_expansion() const
    {
        return wave_m_ == ANY_INT || warp_m_ == ANY_INT || pipeline_ == "*" || pad_m_ == ANY_INT;
    }

    void auto_fill()
    {
        if(wave_m_ == ANY_INT)
            wave_m_ = 2;
        if(wave_n_ == ANY_INT)
            wave_n_ = 2;
        if(wave_k_ == ANY_INT)
            wave_k_ = 1;
        if(warp_m_ == ANY_INT)
            warp_m_ = 32;
        if(warp_n_ == ANY_INT)
            warp_n_ = 32;
        if(warp_k_ == ANY_INT)
            warp_k_ = 16;
    }
};

// =============================================================================
// Kernel Declaration
// =============================================================================

struct KernelDecl
{
    Signature signature;
    Algorithm algorithm;
    std::string arch = "gfx942";

    KernelDecl() = default;

    KernelDecl(const Signature& sig, const Algorithm& algo, const std::string& a = "gfx942")
        : signature(sig), algorithm(algo), arch(a)
    {
    }

    std::string name() const
    {
        std::ostringstream oss;
        oss << signature.dtype_a_ << "_" << signature.layout_str();
        if(algorithm.tile_m_ > 0)
        {
            oss << "_" << algorithm.tile_m_ << "x" << algorithm.tile_n_ << "x" << algorithm.tile_k_;
        }
        return oss.str();
    }

    bool has_wildcards() const { return algorithm.needs_expansion() || arch == "*"; }
};

// =============================================================================
// KernelSet - Collection of declarations
// =============================================================================

class KernelSet
{
    public:
    KernelSet() = default;

    KernelSet& add(const Signature& sig, const Algorithm& algo, const std::string& arch = "gfx942")
    {
        decls_.emplace_back(sig, algo, arch);
        return *this;
    }

    KernelSet& add(const std::string& dtype,
                   const std::string& layout,
                   int tm,
                   int tn,
                   int tk,
                   const std::string& arch = "gfx942")
    {
        Signature sig;
        sig.dtype(dtype).layout(layout);
        Algorithm algo;
        algo.tile(tm, tn, tk);
        decls_.emplace_back(sig, algo, arch);
        return *this;
    }

    KernelSet& add(const KernelDecl& decl)
    {
        decls_.push_back(decl);
        return *this;
    }

    KernelSet& merge(const KernelSet& other)
    {
        decls_.insert(decls_.end(), other.decls_.begin(), other.decls_.end());
        return *this;
    }

    const std::vector<KernelDecl>& declarations() const { return decls_; }
    size_t size() const { return decls_.size(); }

    bool needs_expansion() const
    {
        for(const auto& d : decls_)
        {
            if(d.algorithm.needs_expansion())
                return true;
        }
        return false;
    }

    void print(std::ostream& os = std::cout) const
    {
        os << "KernelSet (" << size() << " declarations):\n";
        for(const auto& d : decls_)
        {
            os << "  - " << d.name();
            if(d.algorithm.needs_expansion())
                os << " [expands]";
            os << "\n";
        }
    }

    KernelSet& tag(const std::string& t)
    {
        tag_ = t;
        return *this;
    }
    std::string tag() const { return tag_; }

    private:
    std::vector<KernelDecl> decls_;
    std::string tag_;
};

// =============================================================================
// KernelSet Registry
// =============================================================================

class KernelSetRegistry
{
    public:
    static KernelSetRegistry& instance()
    {
        static KernelSetRegistry reg;
        return reg;
    }

    void add(const std::string& name, const KernelSet& set)
    {
        sets_[name] = set;
        order_.push_back(name);
    }

    const KernelSet& get(const std::string& name) const
    {
        static KernelSet empty;
        auto it = sets_.find(name);
        return it != sets_.end() ? it->second : empty;
    }

    bool has(const std::string& name) const { return sets_.find(name) != sets_.end(); }

    // Return const reference to avoid deep copy
    const std::vector<std::string>& names() const { return order_; }
    size_t size() const { return sets_.size(); }

    void print() const
    {
        std::cout << "Named Kernel Sets (" << size() << "):\n";
        for(const auto& name : order_)
        {
            const auto& set = sets_.at(name);
            std::cout << "  " << name << ": " << set.size() << " declarations\n";
        }
    }

    private:
    KernelSetRegistry() = default;
    std::unordered_map<std::string, KernelSet> sets_;
    std::vector<std::string> order_;
};

// =============================================================================
// Declaration Registry (for DECL_KERNEL)
// =============================================================================

class Registry
{
    public:
    static Registry& instance()
    {
        static Registry reg;
        return reg;
    }

    void add(const KernelDecl& decl)
    {
        std::string key    = decl.has_wildcards()
                                 ? ("wildcard_" + std::to_string(declarations_.size()))
                                 : decl.name();
        declarations_[key] = decl;
        order_.push_back(key);
    }

    std::vector<KernelDecl> all() const
    {
        std::vector<KernelDecl> result;
        for(const auto& key : order_)
        {
            result.push_back(declarations_.at(key));
        }
        return result;
    }

    size_t size() const { return declarations_.size(); }

    void print() const
    {
        std::cout << "Declared kernels (" << size() << "):\n";
        for(const auto& key : order_)
        {
            const auto& d = declarations_.at(key);
            std::cout << "  " << d.name();
            if(d.has_wildcards())
                std::cout << " [wildcards]";
            std::cout << "\n";
        }
    }

    private:
    Registry() = default;
    std::unordered_map<std::string, KernelDecl> declarations_;
    std::vector<std::string> order_;
};

// =============================================================================
// Static Registrars
// =============================================================================

struct Declarator
{
    Declarator(const Signature& sig, const Algorithm& algo, const std::string& arch = "gfx942")
    {
        Registry::instance().add(KernelDecl(sig, algo, arch));
    }

    Declarator(const std::string& dtype,
               const std::string& layout,
               int tm,
               int tn,
               int tk,
               const std::string& arch = "gfx942")
    {
        Signature sig;
        sig.dtype(dtype).layout(layout);
        Algorithm algo;
        algo.tile(tm, tn, tk);
        Registry::instance().add(KernelDecl(sig, algo, arch));
    }

    Declarator(const std::string& dtype, const std::string& layout, const std::string& arch)
    {
        Signature sig;
        sig.dtype(dtype).layout(layout);
        Algorithm algo;
        algo.tile(ANY_INT, ANY_INT, ANY_INT);
        Registry::instance().add(KernelDecl(sig, algo, arch));
    }
};

struct KernelSetRegistrar
{
    KernelSetRegistrar(const std::string& name, const KernelSet& set)
    {
        KernelSetRegistry::instance().add(name, set);
    }
};

} // namespace decl

// =============================================================================
// Convenience Aliases
// =============================================================================

using KernelSignature    = decl::Signature;
using KernelAlgorithm    = decl::Algorithm;
using KernelDecl         = decl::KernelDecl;
using KernelDeclRegistry = decl::Registry;
using KernelSet          = decl::KernelSet;
using KernelSetRegistry  = decl::KernelSetRegistry;

constexpr const char* ANY = decl::ANY;
constexpr int ANY_INT     = decl::ANY_INT;

} // namespace dispatcher
} // namespace ck_tile

// =============================================================================
// Declaration Macros
// =============================================================================

#define CK_DECL_CAT_(a, b) CK_DECL_CAT_IMPL_(a, b)
#define CK_DECL_CAT_IMPL_(a, b) a##b

// Note: __extension__ suppresses warnings about __COUNTER__ being a GCC/Clang extension
#define DECL_KERNEL(sig, algo, ...)                                            \
    __extension__ static ::ck_tile::dispatcher::decl::Declarator CK_DECL_CAT_( \
        _kdecl_, __COUNTER__)(sig, algo, ##__VA_ARGS__)

#define DECL_KERNEL_SIMPLE(dtype, layout, tm, tn, tk)                          \
    __extension__ static ::ck_tile::dispatcher::decl::Declarator CK_DECL_CAT_( \
        _kdecl_, __COUNTER__)(#dtype, #layout, tm, tn, tk)

#define DECL_KERNEL_ALL(dtype, layout)                                         \
    __extension__ static ::ck_tile::dispatcher::decl::Declarator CK_DECL_CAT_( \
        _kdecl_, __COUNTER__)(#dtype, #layout, "*")

#define DECL_KERNEL_SET(name, ...)                                                     \
    __extension__ static ::ck_tile::dispatcher::decl::KernelSetRegistrar CK_DECL_CAT_( \
        _kset_reg_, __COUNTER__)(#name,                                                \
                                 ::ck_tile::dispatcher::decl::KernelSet() __VA_ARGS__.tag(#name))

#define KERNEL_SET(name) ::ck_tile::dispatcher::decl::KernelSet name
#define BEGIN_KERNEL_SET() ::ck_tile::dispatcher::decl::KernelSet()

// Legacy compatibility
// Legacy aliases removed - use DECL_KERNEL_SET instead
