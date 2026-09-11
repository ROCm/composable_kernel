// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ck_tile {
namespace dispatcher {
namespace fmha_decl {

constexpr const char* ANY = "*";
constexpr int ANY_INT     = -1;

class FmhaSignature
{
    public:
    std::string family_           = "fwd";
    std::string data_type_        = "fp16";
    std::string mode_             = "batch";
    std::string vlayout_          = "r";
    int hdim_q_                   = 128;
    int hdim_v_                   = 128;
    std::string mask_             = "no_mask";
    std::string bias_             = "no_bias";
    bool lse_                     = false;
    bool dropout_                 = false;
    std::string qscale_           = "no_scale";
    std::string rope_             = "none";
    bool logits_                  = false;
    bool paged_kv_                = false;
    bool fp8_static_quant_        = false;
    bool skip_min_seqlen_q_       = false;
    bool sink_                    = false;
    bool dbias_                   = false;
    bool store_randval_           = false;
    bool deterministic_           = false;
    std::string kv_memory_layout_ = "vectorized";
    std::string kv_lookup_table_  = "sglang";
    int page_size_                = 1;
    std::string profile_;
    int receipt_ = -1;

    FmhaSignature& family(const std::string& family)
    {
        family_ = family;
        return *this;
    }

    FmhaSignature& dtype(const std::string& dtype)
    {
        data_type_ = dtype;
        return *this;
    }

    FmhaSignature& mode(const std::string& mode)
    {
        mode_ = mode;
        return *this;
    }

    FmhaSignature& vlayout(const std::string& layout)
    {
        vlayout_ = layout;
        return *this;
    }

    FmhaSignature& hdim(int q, int v = -1)
    {
        hdim_q_ = q;
        hdim_v_ = (v < 0 ? q : v);
        return *this;
    }

    FmhaSignature& mask(const std::string& mask)
    {
        mask_ = mask;
        return *this;
    }

    FmhaSignature& bias(const std::string& bias)
    {
        bias_ = bias;
        return *this;
    }

    FmhaSignature& lse(bool value = true)
    {
        lse_ = value;
        return *this;
    }

    FmhaSignature& dropout(bool value = true)
    {
        dropout_ = value;
        return *this;
    }

    FmhaSignature& qscale(const std::string& qscale)
    {
        qscale_ = qscale;
        return *this;
    }

    FmhaSignature& rope(const std::string& rope)
    {
        rope_ = rope;
        return *this;
    }

    FmhaSignature& logits(bool value = true)
    {
        logits_ = value;
        return *this;
    }

    FmhaSignature& paged_kv(bool value = true)
    {
        paged_kv_ = value;
        return *this;
    }

    FmhaSignature& fp8_static_quant(bool value = true)
    {
        fp8_static_quant_ = value;
        return *this;
    }

    FmhaSignature& skip(bool value = true)
    {
        skip_min_seqlen_q_ = value;
        return *this;
    }

    FmhaSignature& sink(bool value = true)
    {
        sink_ = value;
        return *this;
    }

    FmhaSignature& dbias(bool value = true)
    {
        dbias_ = value;
        return *this;
    }

    FmhaSignature& store_randval(bool value = true)
    {
        store_randval_ = value;
        return *this;
    }

    FmhaSignature& deterministic(bool value = true)
    {
        deterministic_ = value;
        return *this;
    }

    FmhaSignature&
    kv_cache(const std::string& memory_layout, const std::string& lookup_table, int page_size = 1)
    {
        kv_memory_layout_ = memory_layout;
        kv_lookup_table_  = lookup_table;
        page_size_        = page_size;
        return *this;
    }

    FmhaSignature& profile(const std::string& profile)
    {
        profile_ = profile;
        return *this;
    }

    FmhaSignature& receipt(int receipt)
    {
        receipt_ = receipt;
        return *this;
    }
};

class FmhaAlgorithm
{
    public:
    int tile_m0_    = 128;
    int tile_n0_    = 64;
    int tile_k0_    = 32;
    int tile_n1_    = 128;
    int tile_k1_    = 32;
    int tile_k0max_ = 128;

    int wave_m0_ = 2;
    int wave_n0_ = 2;
    int wave_k0_ = 1;
    int wave_m1_ = 2;
    int wave_n1_ = 2;
    int wave_k1_ = 1;
    int wave_m2_ = 1;
    int wave_n2_ = 1;
    int wave_k2_ = 1;

    int warp_m0_ = 32;
    int warp_n0_ = 32;
    int warp_k0_ = 16;
    int warp_m1_ = 32;
    int warp_n1_ = 32;
    int warp_k1_ = 16;
    int warp_m2_ = 16;
    int warp_n2_ = 16;
    int warp_k2_ = 16;

    std::string pipeline_ = "qr";
    bool pad_s_           = true;
    bool pad_sk_          = true;
    bool pad_d_           = true;
    bool pad_dv_          = true;
    bool use_trload_      = false;
    int hdim_q_alignment_ = 0;
    int hdim_v_alignment_ = 0;
    int block_per_cu_     = 1;
    int num_wave_groups_  = 1;
    int max_splits_log2_  = 0;
    int max_seq_len_q_    = 0;
    int selection_rank_   = 0;
    std::string constraint_tag_;

    // Bulk setters (positional, for backward compatibility)
    FmhaAlgorithm& tile(int m0, int n0, int k0, int n1, int k1, int k0max)
    {
        tile_m0_    = m0;
        tile_n0_    = n0;
        tile_k0_    = k0;
        tile_n1_    = n1;
        tile_k1_    = k1;
        tile_k0max_ = k0max;
        return *this;
    }

    FmhaAlgorithm& wave(int m0,
                        int n0,
                        int k0,
                        int m1 = 2,
                        int n1 = 2,
                        int k1 = 1,
                        int m2 = 1,
                        int n2 = 1,
                        int k2 = 1)
    {
        wave_m0_ = m0;
        wave_n0_ = n0;
        wave_k0_ = k0;
        wave_m1_ = m1;
        wave_n1_ = n1;
        wave_k1_ = k1;
        wave_m2_ = m2;
        wave_n2_ = n2;
        wave_k2_ = k2;
        return *this;
    }

    FmhaAlgorithm& warp(int m0,
                        int n0,
                        int k0,
                        int m1 = 32,
                        int n1 = 32,
                        int k1 = 16,
                        int m2 = 16,
                        int n2 = 16,
                        int k2 = 16)
    {
        warp_m0_ = m0;
        warp_n0_ = n0;
        warp_k0_ = k0;
        warp_m1_ = m1;
        warp_n1_ = n1;
        warp_k1_ = k1;
        warp_m2_ = m2;
        warp_n2_ = n2;
        warp_k2_ = k2;
        return *this;
    }

    // Named individual setters for clarity (preferred over positional bulk setters)
    // Stage 0: Q * K^T  (seqlen_q x seqlen_k x hdim_q)
    FmhaAlgorithm& tile_m0(int v)
    {
        tile_m0_ = v;
        return *this;
    }
    FmhaAlgorithm& tile_n0(int v)
    {
        tile_n0_ = v;
        return *this;
    }
    FmhaAlgorithm& tile_k0(int v)
    {
        tile_k0_ = v;
        return *this;
    }
    // Stage 1: Attn * V  (seqlen_q x hdim_v x seqlen_k)
    FmhaAlgorithm& tile_n1(int v)
    {
        tile_n1_ = v;
        return *this;
    }
    FmhaAlgorithm& tile_k1(int v)
    {
        tile_k1_ = v;
        return *this;
    }
    FmhaAlgorithm& tile_k0max(int v)
    {
        tile_k0max_ = v;
        return *this;
    }

    FmhaAlgorithm& wave_m0(int v)
    {
        wave_m0_ = v;
        return *this;
    }
    FmhaAlgorithm& wave_n0(int v)
    {
        wave_n0_ = v;
        return *this;
    }
    FmhaAlgorithm& wave_k0(int v)
    {
        wave_k0_ = v;
        return *this;
    }
    FmhaAlgorithm& wave_m1(int v)
    {
        wave_m1_ = v;
        return *this;
    }
    FmhaAlgorithm& wave_n1(int v)
    {
        wave_n1_ = v;
        return *this;
    }
    FmhaAlgorithm& wave_k1(int v)
    {
        wave_k1_ = v;
        return *this;
    }

    FmhaAlgorithm& warp_m0(int v)
    {
        warp_m0_ = v;
        return *this;
    }
    FmhaAlgorithm& warp_n0(int v)
    {
        warp_n0_ = v;
        return *this;
    }
    FmhaAlgorithm& warp_k0(int v)
    {
        warp_k0_ = v;
        return *this;
    }
    FmhaAlgorithm& warp_m1(int v)
    {
        warp_m1_ = v;
        return *this;
    }
    FmhaAlgorithm& warp_n1(int v)
    {
        warp_n1_ = v;
        return *this;
    }
    FmhaAlgorithm& warp_k1(int v)
    {
        warp_k1_ = v;
        return *this;
    }

    FmhaAlgorithm& pipeline(const std::string& pipeline)
    {
        pipeline_ = pipeline;
        return *this;
    }

    FmhaAlgorithm& padding(bool s, bool sk, bool d, bool dv)
    {
        pad_s_  = s;
        pad_sk_ = sk;
        pad_d_  = d;
        pad_dv_ = dv;
        return *this;
    }

    FmhaAlgorithm& trload(bool value = true)
    {
        use_trload_ = value;
        return *this;
    }

    FmhaAlgorithm& alignments(int q_alignment, int v_alignment)
    {
        hdim_q_alignment_ = q_alignment;
        hdim_v_alignment_ = v_alignment;
        return *this;
    }

    FmhaAlgorithm& block_per_cu(int value)
    {
        block_per_cu_ = value;
        return *this;
    }

    FmhaAlgorithm& num_wave_groups(int value)
    {
        num_wave_groups_ = value;
        return *this;
    }

    FmhaAlgorithm& max_splits_log2(int value)
    {
        max_splits_log2_ = value;
        return *this;
    }

    FmhaAlgorithm& max_seq_len_q(int value)
    {
        max_seq_len_q_ = value;
        return *this;
    }

    FmhaAlgorithm& selection_rank(int value)
    {
        selection_rank_ = value;
        return *this;
    }

    FmhaAlgorithm& constraint(const std::string& tag)
    {
        constraint_tag_ = tag;
        return *this;
    }

    void auto_fill()
    {
        if(tile_n1_ <= 0)
        {
            tile_n1_ = tile_n0_;
        }
        if(tile_k1_ <= 0)
        {
            tile_k1_ = tile_k0_;
        }
        if(tile_k0max_ <= 0)
        {
            tile_k0max_ = tile_k0_;
        }
        if(hdim_q_alignment_ <= 0)
        {
            hdim_q_alignment_ = tile_k0max_;
        }
        if(hdim_v_alignment_ <= 0)
        {
            hdim_v_alignment_ = tile_k0max_;
        }
    }
};

struct FmhaKernelDecl
{
    FmhaSignature signature;
    FmhaAlgorithm algorithm;
    std::string arch = "gfx942";

    FmhaKernelDecl() = default;
    FmhaKernelDecl(const FmhaSignature& sig,
                   const FmhaAlgorithm& algo,
                   const std::string& target_arch = "gfx942")
        : signature(sig), algorithm(algo), arch(target_arch)
    {
    }

    std::string name() const
    {
        std::ostringstream oss;
        oss << "fmha_" << signature.family_ << "_" << signature.data_type_ << "_" << signature.mode_
            << "_dq" << signature.hdim_q_ << "_dv" << signature.hdim_v_ << "_" << signature.vlayout_
            << "_" << algorithm.pipeline_;
        return oss.str();
    }

    bool has_wildcards() const { return arch == "*"; }
};

class FmhaKernelSet
{
    public:
    FmhaKernelSet() = default;

    FmhaKernelSet&
    add(const FmhaSignature& sig, const FmhaAlgorithm& algo, const std::string& arch = "gfx942")
    {
        decls_.emplace_back(sig, algo, arch);
        return *this;
    }

    FmhaKernelSet& add(const FmhaKernelDecl& decl)
    {
        decls_.push_back(decl);
        return *this;
    }

    FmhaKernelSet& merge(const FmhaKernelSet& other)
    {
        decls_.insert(decls_.end(), other.decls_.begin(), other.decls_.end());
        return *this;
    }

    const std::vector<FmhaKernelDecl>& declarations() const { return decls_; }
    std::size_t size() const { return decls_.size(); }

    bool needs_expansion() const
    {
        for(const auto& d : decls_)
        {
            if(d.has_wildcards())
                return true;
        }
        return false;
    }

    void print(std::ostream& os = std::cout) const
    {
        os << "FmhaKernelSet (" << size() << " declarations):\n";
        for(const auto& decl : decls_)
        {
            os << "  - " << decl.name();
            if(decl.has_wildcards())
                os << " [expands]";
            os << "\n";
        }
    }

    FmhaKernelSet& tag(const std::string& tag)
    {
        tag_ = tag;
        return *this;
    }

    const std::string& tag() const { return tag_; }

    private:
    std::vector<FmhaKernelDecl> decls_;
    std::string tag_;
};

/// Singleton registry for declarative kernel sets.
/// Thread safety: only populated during static initialization (pre-main)
/// via DECL_FMHA_KERNEL_SET macros. Do NOT call add() after main() starts.
class FmhaKernelSetRegistry
{
    public:
    static FmhaKernelSetRegistry& instance()
    {
        static FmhaKernelSetRegistry registry;
        return registry;
    }

    void add(const std::string& name, const FmhaKernelSet& set)
    {
        sets_[name] = set;
        if(std::find(order_.begin(), order_.end(), name) == order_.end())
        {
            order_.push_back(name);
        }
    }

    const FmhaKernelSet& get(const std::string& name) const
    {
        static FmhaKernelSet empty;
        auto it = sets_.find(name);
        return it != sets_.end() ? it->second : empty;
    }

    bool has(const std::string& name) const { return sets_.find(name) != sets_.end(); }

    const std::vector<std::string>& names() const { return order_; }

    std::size_t size() const { return sets_.size(); }

    void clear()
    {
        sets_.clear();
        order_.clear();
    }

    void print() const
    {
        std::cout << "FMHA Kernel Sets (" << sets_.size() << "):\n";
        for(const auto& name : order_)
        {
            const auto& set = sets_.at(name);
            std::cout << "  " << name << ": " << set.size() << " declarations\n";
        }
    }

    private:
    std::unordered_map<std::string, FmhaKernelSet> sets_;
    std::vector<std::string> order_;
};

struct FmhaKernelSetRegistrar
{
    FmhaKernelSetRegistrar(const std::string& name, const FmhaKernelSet& set)
    {
        FmhaKernelSetRegistry::instance().add(name, set);
    }
};

} // namespace fmha_decl

using FmhaSignature         = fmha_decl::FmhaSignature;
using FmhaAlgorithm         = fmha_decl::FmhaAlgorithm;
using FmhaKernelDecl        = fmha_decl::FmhaKernelDecl;
using FmhaKernelSet         = fmha_decl::FmhaKernelSet;
using FmhaKernelSetRegistry = fmha_decl::FmhaKernelSetRegistry;

} // namespace dispatcher
} // namespace ck_tile

#define CK_FMHA_DECL_CAT_(a, b) CK_FMHA_DECL_CAT_IMPL_(a, b)
#define CK_FMHA_DECL_CAT_IMPL_(a, b) a##b

#if defined(__GNUC__) || defined(__clang__)
#define CK_FMHA_DECL_EXT_ __extension__
#else
#define CK_FMHA_DECL_EXT_
#endif

#define DECL_FMHA_KERNEL_SET(name, ...)                                               \
    CK_FMHA_DECL_EXT_ static ::ck_tile::dispatcher::fmha_decl::FmhaKernelSetRegistrar \
    CK_FMHA_DECL_CAT_(_fmha_kset_reg_, __COUNTER__)(                                  \
        #name, ::ck_tile::dispatcher::fmha_decl::FmhaKernelSet() __VA_ARGS__.tag(#name))
