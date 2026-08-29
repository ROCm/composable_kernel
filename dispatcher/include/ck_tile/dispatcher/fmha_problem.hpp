// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/fmha_types.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

namespace ck_tile {
namespace dispatcher {

enum class FmhaApiFamily : std::uint8_t
{
    Fwd,
    FwdPagedKv,
    FwdSplitKv,
    FwdAppendKv,
    BatchPrefill,
    Bwd
};

enum class FmhaKernelFamily : std::uint8_t
{
    Fwd,
    FwdPagedKv,
    FwdSplitKv,
    FwdSplitKvCombine,
    FwdAppendKv,
    BatchPrefill,
    BwdDotDoO,
    BwdDqDkDv,
    BwdConvertDq
};

inline std::string to_string(FmhaApiFamily family)
{
    switch(family)
    {
    case FmhaApiFamily::Fwd: return "fwd";
    case FmhaApiFamily::FwdPagedKv: return "fwd_pagedkv";
    case FmhaApiFamily::FwdSplitKv: return "fwd_splitkv";
    case FmhaApiFamily::FwdAppendKv: return "fwd_appendkv";
    case FmhaApiFamily::BatchPrefill: return "batch_prefill";
    case FmhaApiFamily::Bwd: return "bwd";
    default: return "unknown";
    }
}

inline std::string to_string(FmhaKernelFamily family)
{
    switch(family)
    {
    case FmhaKernelFamily::Fwd: return "fwd";
    case FmhaKernelFamily::FwdPagedKv: return "fwd_pagedkv";
    case FmhaKernelFamily::FwdSplitKv: return "fwd_splitkv";
    case FmhaKernelFamily::FwdSplitKvCombine: return "fwd_splitkv_combine";
    case FmhaKernelFamily::FwdAppendKv: return "fwd_appendkv";
    case FmhaKernelFamily::BatchPrefill: return "batch_prefill";
    case FmhaKernelFamily::BwdDotDoO: return "bwd_dot_do_o";
    case FmhaKernelFamily::BwdDqDkDv: return "bwd_dq_dk_dv";
    case FmhaKernelFamily::BwdConvertDq: return "bwd_convert_dq";
    default: return "unknown";
    }
}

// Combined variants containing both forward and backward types.
// Both fwd and bwd types are always available via fallback definitions
// in fmha_types.hpp (they are conditionally guarded but the fallback
// provides them when the example headers don't).
using FmhaTraitsVariant = std::variant<fmha_fwd_traits,
                                       fmha_fwd_pagedkv_traits,
                                       fmha_fwd_splitkv_traits,
                                       fmha_fwd_appendkv_traits,
                                       fmha_batch_prefill_traits,
                                       fmha_bwd_traits>;

using FmhaArgsVariant = std::variant<fmha_fwd_args,
                                     fmha_fwd_pagedkv_args,
                                     fmha_fwd_splitkv_args,
                                     fmha_fwd_appendkv_args,
                                     fmha_batch_prefill_args,
                                     fmha_bwd_args>;

struct FmhaInvocation
{
    FmhaApiFamily api_family = FmhaApiFamily::Fwd;
    FmhaTraitsVariant traits;
    FmhaArgsVariant args;

    static FmhaInvocation make(fmha_fwd_traits t, fmha_fwd_args a)
    {
        return {FmhaApiFamily::Fwd, std::move(t), std::move(a)};
    }

    static FmhaInvocation make(fmha_fwd_pagedkv_traits t, fmha_fwd_pagedkv_args a)
    {
        return {FmhaApiFamily::FwdPagedKv, std::move(t), std::move(a)};
    }

    static FmhaInvocation make(fmha_fwd_splitkv_traits t, fmha_fwd_splitkv_args a)
    {
        return {FmhaApiFamily::FwdSplitKv, std::move(t), std::move(a)};
    }

    static FmhaInvocation make(fmha_fwd_appendkv_traits t, fmha_fwd_appendkv_args a)
    {
        return {FmhaApiFamily::FwdAppendKv, std::move(t), std::move(a)};
    }

    static FmhaInvocation make(fmha_batch_prefill_traits t, fmha_batch_prefill_args a)
    {
        return {FmhaApiFamily::BatchPrefill, std::move(t), std::move(a)};
    }

    static FmhaInvocation make(fmha_bwd_traits t, fmha_bwd_args a)
    {
        return {FmhaApiFamily::Bwd, std::move(t), std::move(a)};
    }
};

struct FmhaProblem
{
    FmhaApiFamily api_family          = FmhaApiFamily::Fwd;
    FmhaKernelFamily requested_family = FmhaKernelFamily::Fwd;
    std::string gfx_arch;
    std::string data_type;

    bool is_group_mode       = false;
    bool is_v_rowmajor       = true;
    bool has_logits_soft_cap = false;
    int mask_type            = 0;
    int bias_type            = 0;
    bool has_lse             = false;
    bool has_dropout         = false;
    int qscale_type          = 0;
    int rope_type            = 0;
    bool use_paged_kv        = false;
    bool do_fp8_static_quant = false;
    bool skip_min_seqlen_q   = false;
    bool has_sink            = false;
    bool has_dbias           = false;
    bool is_store_randval    = false;
    bool is_deterministic    = false;
    int kv_memory_layout     = 0;
    int kv_lookup_table      = 0;
    int page_size            = 1;

    std::int64_t seqlen_q          = 0;
    std::int64_t seqlen_k          = 0;
    std::int64_t max_seqlen_q      = 0;
    std::int64_t max_seqlen_k      = 0;
    std::int64_t batch             = 0;
    std::int64_t hdim_q            = 0;
    std::int64_t hdim_v            = 0;
    std::int64_t nhead_q           = 0;
    std::int64_t nhead_k           = 0;
    std::int64_t num_splits        = 1;
    std::int64_t window_size_left  = 0;
    std::int64_t window_size_right = 0;
    std::int64_t sink_size         = 0;
    std::int64_t min_seqlen_q      = 0;
    std::int64_t rotary_dim        = 0;

    bool has_seqstart_q_ptr  = false;
    bool has_seqstart_k_ptr  = false;
    bool has_seqlen_q_ptr    = false;
    bool has_seqlen_k_ptr    = false;
    bool has_cu_seqlen_q_ptr = false;
    bool has_cu_seqlen_k_ptr = false;
    bool has_block_table_ptr = false;
    bool has_cache_batch_idx = false;
    bool is_gappy            = false;
    bool has_rotary_cos_sin  = false;

    [[nodiscard]] bool is_valid() const
    {
        return !data_type.empty() && batch > 0 && hdim_q > 0 && hdim_v > 0 && nhead_q > 0 &&
               nhead_k > 0;
    }

    [[nodiscard]] std::int64_t effective_max_seqlen_q() const
    {
        return max_seqlen_q > 0 ? max_seqlen_q : seqlen_q;
    }

    [[nodiscard]] std::int64_t effective_max_seqlen_k() const
    {
        return max_seqlen_k > 0 ? max_seqlen_k : seqlen_k;
    }

    [[nodiscard]] bool has_variable_seqlen_q() const
    {
        return has_seqstart_q_ptr || has_seqlen_q_ptr || has_cu_seqlen_q_ptr;
    }

    [[nodiscard]] bool has_variable_seqlen_k() const
    {
        return has_seqstart_k_ptr || has_seqlen_k_ptr || has_cu_seqlen_k_ptr || is_gappy;
    }

    [[nodiscard]] std::uint64_t num_ops() const
    {
        const auto sq = effective_max_seqlen_q();
        const auto sk = effective_max_seqlen_k();
        if(batch <= 0 || nhead_q <= 0 || sq <= 0 || sk <= 0 || hdim_q <= 0 || hdim_v <= 0)
            return 0;
        return 2ULL * static_cast<std::uint64_t>(batch) * static_cast<std::uint64_t>(nhead_q) *
               static_cast<std::uint64_t>(sq) * static_cast<std::uint64_t>(sk) *
               static_cast<std::uint64_t>(hdim_q + hdim_v);
    }

    [[nodiscard]] std::string to_string() const
    {
        std::string s;
        s += "FmhaProblem(";
        s += "api=" + ck_tile::dispatcher::to_string(api_family);
        s += ", family=" + ck_tile::dispatcher::to_string(requested_family);
        s += ", dtype=" + data_type;
        s += ", arch=" + gfx_arch;
        s += ", batch=" + std::to_string(batch);
        s += ", sq=" + std::to_string(seqlen_q);
        s += ", sk=" + std::to_string(seqlen_k);
        s += ", dq=" + std::to_string(hdim_q);
        s += ", dv=" + std::to_string(hdim_v);
        s += ", hq=" + std::to_string(nhead_q);
        s += ", hk=" + std::to_string(nhead_k);
        s += ", group=" + std::string(is_group_mode ? "y" : "n");
        s += ", mask=" + std::to_string(mask_type);
        s += ", bias=" + std::to_string(bias_type);
        s += ")";
        return s;
    }

    /// Canonical key for caching -- includes ALL fields used by fmha_signature_matches().
    /// Safe to use as a cache key (unlike to_string() which omits many fields).
    [[nodiscard]] std::string canonical_key() const
    {
        constexpr char S = '\x1f'; // ASCII unit separator -- unambiguous delimiter
        std::string k;
        k.reserve(256);
        k += ck_tile::dispatcher::to_string(api_family);
        k += S;
        k += ck_tile::dispatcher::to_string(requested_family);
        k += S;
        k += data_type;
        k += S;
        k += gfx_arch;
        k += S;
        k += std::to_string(hdim_q);
        k += ',';
        k += std::to_string(hdim_v);
        k += S;
        k += is_group_mode ? '1' : '0';
        k += is_v_rowmajor ? '1' : '0';
        k += has_logits_soft_cap ? '1' : '0';
        k += has_lse ? '1' : '0';
        k += has_dropout ? '1' : '0';
        k += use_paged_kv ? '1' : '0';
        k += do_fp8_static_quant ? '1' : '0';
        k += skip_min_seqlen_q ? '1' : '0';
        k += has_sink ? '1' : '0';
        k += has_dbias ? '1' : '0';
        k += is_store_randval ? '1' : '0';
        k += is_deterministic ? '1' : '0';
        k += S;
        k += std::to_string(mask_type);
        k += ',';
        k += std::to_string(bias_type);
        k += ',';
        k += std::to_string(qscale_type);
        k += ',';
        k += std::to_string(rope_type);
        k += S;
        k += std::to_string(kv_memory_layout);
        k += ',';
        k += std::to_string(kv_lookup_table);
        k += ',';
        k += std::to_string(page_size);
        return k;
    }

    [[nodiscard]] static FmhaProblem from_invocation(const FmhaInvocation& invocation,
                                                     const std::string& gfx_arch = "")
    {
        FmhaProblem p;
        p.api_family = invocation.api_family;
        p.gfx_arch   = gfx_arch;

        std::visit(
            [&](const auto& traits) {
                using T = std::decay_t<decltype(traits)>;

                if constexpr(std::is_same_v<T, fmha_fwd_traits>)
                {
                    p.requested_family    = FmhaKernelFamily::Fwd;
                    p.data_type           = traits.data_type;
                    p.is_group_mode       = traits.is_group_mode;
                    p.is_v_rowmajor       = traits.is_v_rowmajor;
                    p.has_logits_soft_cap = traits.has_logits_soft_cap;
                    p.mask_type           = static_cast<int>(traits.mask_type);
                    p.bias_type           = static_cast<int>(traits.bias_type);
                    p.has_lse             = traits.has_lse;
                    p.has_dropout         = traits.has_dropout;
                    p.qscale_type         = static_cast<int>(traits.qscale_type);
                    p.skip_min_seqlen_q   = traits.skip_min_seqlen_q;
                    p.has_sink            = traits.has_sink;
                    p.hdim_q              = traits.hdim_q;
                    p.hdim_v              = traits.hdim_v;
                }
                else if constexpr(std::is_same_v<T, fmha_fwd_pagedkv_traits>)
                {
                    p.requested_family    = FmhaKernelFamily::FwdPagedKv;
                    p.data_type           = traits.data_type;
                    p.is_group_mode       = traits.is_group_mode;
                    p.is_v_rowmajor       = traits.is_v_rowmajor;
                    p.has_logits_soft_cap = traits.has_logits_soft_cap;
                    p.mask_type           = static_cast<int>(traits.mask_type);
                    p.bias_type           = static_cast<int>(traits.bias_type);
                    p.has_lse             = traits.has_lse;
                    p.use_paged_kv        = traits.use_pagedkv;
                    p.do_fp8_static_quant = traits.do_fp8_static_quant;
                    p.skip_min_seqlen_q   = traits.skip_min_seqlen_q;
                    p.has_sink            = traits.has_sink;
                    p.hdim_q              = traits.hdim_q;
                    p.hdim_v              = traits.hdim_v;
                }
                else if constexpr(std::is_same_v<T, fmha_fwd_splitkv_traits>)
                {
                    p.requested_family    = FmhaKernelFamily::FwdSplitKv;
                    p.data_type           = traits.data_type;
                    p.is_group_mode       = traits.is_group_mode;
                    p.is_v_rowmajor       = traits.is_v_rowmajor;
                    p.has_logits_soft_cap = traits.has_logits_soft_cap;
                    p.mask_type           = static_cast<int>(traits.mask_type);
                    p.bias_type           = static_cast<int>(traits.bias_type);
                    p.has_lse             = traits.has_lse;
                    p.do_fp8_static_quant = traits.do_fp8_static_quant;
                    p.has_sink            = traits.has_sink;
                    p.hdim_q              = traits.hdim_q;
                    p.hdim_v              = traits.hdim_v;
                    // Explicit defaults for fields not in splitkv traits
                    p.has_dropout       = false;
                    p.skip_min_seqlen_q = false;
                    p.use_paged_kv      = false;
                    p.has_dbias         = false;
                    p.is_store_randval  = false;
                    p.is_deterministic  = false;
                }
                else if constexpr(std::is_same_v<T, fmha_fwd_appendkv_traits>)
                {
                    p.requested_family = FmhaKernelFamily::FwdAppendKv;
                    p.data_type        = traits.data_type;
                    p.is_group_mode    = false;
                    p.is_v_rowmajor    = traits.is_v_rowmajor;
                    p.rope_type        = static_cast<int>(traits.rope_type);
                    p.hdim_q           = traits.hdim_q;
                    p.hdim_v           = traits.hdim_v;
                    // Explicit defaults for fields not in appendkv traits
                    p.has_logits_soft_cap = false;
                    p.mask_type           = 0;
                    p.bias_type           = 0;
                    p.has_lse             = false;
                    p.has_dropout         = false;
                    p.has_sink            = false;
                    p.skip_min_seqlen_q   = false;
                    p.use_paged_kv        = false;
                    p.has_dbias           = false;
                    p.is_store_randval    = false;
                    p.is_deterministic    = false;
                }
                else if constexpr(std::is_same_v<T, fmha_batch_prefill_traits>)
                {
                    p.requested_family    = FmhaKernelFamily::BatchPrefill;
                    p.data_type           = traits.data_type;
                    p.is_group_mode       = traits.is_group_mode;
                    p.is_v_rowmajor       = traits.is_v_rowmajor;
                    p.has_logits_soft_cap = traits.has_logits_soft_cap;
                    p.mask_type           = static_cast<int>(traits.mask_type);
                    p.bias_type           = static_cast<int>(traits.bias_type);
                    p.has_lse             = traits.has_lse;
                    p.has_dropout         = traits.has_dropout;
                    p.qscale_type         = static_cast<int>(traits.qscale_type);
                    p.skip_min_seqlen_q   = traits.skip_min_seqlen_q;
                    p.has_sink            = traits.has_sink;
                    p.kv_memory_layout    = static_cast<int>(traits.kv_memory_layout);
                    p.kv_lookup_table     = static_cast<int>(traits.kv_lookup_table);
                    p.page_size           = traits.page_size;
                    p.use_paged_kv        = true;
                    p.hdim_q              = traits.hdim_q;
                    p.hdim_v              = traits.hdim_v;
                }
                else if constexpr(std::is_same_v<T, fmha_bwd_traits>)
                {
                    p.requested_family = FmhaKernelFamily::BwdDqDkDv;
                    p.seqlen_q         = traits.seqlen_q;
                    p.seqlen_k         = traits.seqlen_k;
                    p.batch            = traits.batch;
                    p.max_seqlen_q     = traits.max_seqlen_q;
                    p.max_seqlen_k     = traits.max_seqlen_k;
                    p.hdim_q           = traits.hdim_q;
                    p.hdim_v           = traits.hdim_v;
                    p.nhead_q          = traits.nhead_q;
                    p.nhead_k          = traits.nhead_k;
                    p.data_type        = traits.data_type;
                    p.is_group_mode    = traits.is_group_mode;
                    p.mask_type        = static_cast<int>(traits.mask_type);
                    p.bias_type        = static_cast<int>(traits.bias_type);
                    p.has_dbias        = traits.has_dbias;
                    p.has_dropout      = traits.has_dropout;
                    p.is_store_randval = traits.is_store_randval;
                    p.is_deterministic = traits.is_deterministic;
                    // Explicit defaults for fields not in bwd traits
                    p.is_v_rowmajor       = true;
                    p.has_logits_soft_cap = false;
                    p.has_lse             = false;
                    p.has_sink            = false;
                    p.skip_min_seqlen_q   = false;
                    p.use_paged_kv        = false;
                }
            },
            invocation.traits);

        std::visit(
            [&](const auto& args) {
                using T = std::decay_t<decltype(args)>;

                if constexpr(std::is_same_v<T, fmha_fwd_args>)
                {
                    p.seqlen_q            = args.seqlen_q;
                    p.seqlen_k            = args.seqlen_k;
                    p.batch               = args.batch;
                    p.max_seqlen_q        = args.max_seqlen_q;
                    p.nhead_q             = args.nhead_q;
                    p.nhead_k             = args.nhead_k;
                    p.window_size_left    = args.window_size_left;
                    p.window_size_right   = args.window_size_right;
                    p.sink_size           = args.sink_size;
                    p.min_seqlen_q        = args.min_seqlen_q;
                    p.has_seqstart_q_ptr  = args.seqstart_q_ptr != nullptr;
                    p.has_seqstart_k_ptr  = args.seqstart_k_ptr != nullptr;
                    p.has_seqlen_q_ptr    = args.seqlen_q_ptr != nullptr;
                    p.has_seqlen_k_ptr    = args.seqlen_k_ptr != nullptr;
                    p.has_cu_seqlen_q_ptr = args.cu_seqlen_q_ptr != nullptr;
                    p.has_cu_seqlen_k_ptr = args.cu_seqlen_k_ptr != nullptr;
                }
                else if constexpr(std::is_same_v<T, fmha_fwd_pagedkv_args>)
                {
                    p.seqlen_q            = args.seqlen_q;
                    p.seqlen_k            = args.seqlen_k;
                    p.batch               = args.batch;
                    p.max_seqlen_q        = args.max_seqlen_q;
                    p.nhead_q             = args.nhead_q;
                    p.nhead_k             = args.nhead_k;
                    p.page_size           = args.page_block_size;
                    p.window_size_left    = args.window_size_left;
                    p.window_size_right   = args.window_size_right;
                    p.sink_size           = args.sink_size;
                    p.min_seqlen_q        = args.min_seqlen_q;
                    p.has_seqstart_q_ptr  = args.seqstart_q_ptr != nullptr;
                    p.has_seqstart_k_ptr  = args.seqstart_k_ptr != nullptr;
                    p.has_seqlen_k_ptr    = args.seqlen_k_ptr != nullptr;
                    p.has_block_table_ptr = args.block_table_ptr != nullptr;
                    p.has_cache_batch_idx = args.cache_batch_idx != nullptr;
                    p.is_gappy            = args.is_gappy;
                }
                else if constexpr(std::is_same_v<T, fmha_fwd_splitkv_args>)
                {
                    p.seqlen_q            = args.seqlen_q;
                    p.seqlen_k            = args.seqlen_k;
                    p.batch               = args.batch;
                    p.max_seqlen_q        = args.max_seqlen_q;
                    p.nhead_q             = args.nhead_q;
                    p.nhead_k             = args.nhead_k;
                    p.num_splits          = args.num_splits;
                    p.page_size           = args.page_block_size;
                    p.window_size_left    = args.window_size_left;
                    p.window_size_right   = args.window_size_right;
                    p.sink_size           = args.sink_size;
                    p.has_seqstart_q_ptr  = args.seqstart_q_ptr != nullptr;
                    p.has_seqstart_k_ptr  = args.seqstart_k_ptr != nullptr;
                    p.has_seqlen_k_ptr    = args.seqlen_k_ptr != nullptr;
                    p.has_block_table_ptr = args.block_table_ptr != nullptr;
                    p.has_cache_batch_idx = args.cache_batch_idx != nullptr;
                    p.is_gappy            = args.is_gappy;
                    p.use_paged_kv        = args.block_table_ptr != nullptr;
                }
                else if constexpr(std::is_same_v<T, fmha_fwd_appendkv_args>)
                {
                    p.seqlen_q            = args.seqlen_q;
                    p.seqlen_k            = args.seqlen_knew;
                    p.batch               = args.batch;
                    p.nhead_q             = args.nhead_q;
                    p.nhead_k             = args.nhead_k;
                    p.page_size           = args.page_block_size;
                    p.rotary_dim          = args.rotary_dim;
                    p.has_seqlen_k_ptr    = args.seqlen_k_ptr != nullptr;
                    p.has_block_table_ptr = args.block_table_ptr != nullptr;
                    p.has_cache_batch_idx = args.cache_batch_idx != nullptr;
                    p.has_rotary_cos_sin =
                        args.rotary_cos_ptr != nullptr && args.rotary_sin_ptr != nullptr;
                    p.use_paged_kv = args.block_table_ptr != nullptr;
                }
                else if constexpr(std::is_same_v<T, fmha_batch_prefill_args>)
                {
                    p.seqlen_q           = args.seqlen_q;
                    p.seqlen_k           = args.seqlen_k;
                    p.batch              = args.batch;
                    p.max_seqlen_q       = args.max_seqlen_q;
                    p.nhead_q            = args.nhead_q;
                    p.nhead_k            = args.nhead_k;
                    p.page_size          = args.page_block_size;
                    p.kv_memory_layout   = static_cast<int>(args.kv_memory_layout);
                    p.kv_lookup_table    = static_cast<int>(args.kv_lookup_table);
                    p.window_size_left   = args.window_size_left;
                    p.window_size_right  = args.window_size_right;
                    p.sink_size          = args.sink_size;
                    p.has_seqstart_q_ptr = args.seqstart_q_ptr != nullptr;
                    p.has_seqlen_k_ptr   = args.seqlen_k_ptr != nullptr;
                    p.use_paged_kv       = true;
                }
                else if constexpr(std::is_same_v<T, fmha_bwd_args>)
                {
                    p.seqlen_q            = args.seqlen_q;
                    p.seqlen_k            = args.seqlen_k;
                    p.batch               = args.batch;
                    p.max_seqlen_q        = args.max_seqlen_q;
                    p.max_seqlen_k        = args.max_seqlen_k;
                    p.nhead_q             = args.nhead_q;
                    p.nhead_k             = args.nhead_k;
                    p.window_size_left    = args.window_size_left;
                    p.window_size_right   = args.window_size_right;
                    p.has_seqstart_q_ptr  = args.seqstart_q_ptr != nullptr;
                    p.has_seqstart_k_ptr  = args.seqstart_k_ptr != nullptr;
                    p.has_seqlen_q_ptr    = args.seqlen_q_ptr != nullptr;
                    p.has_seqlen_k_ptr    = args.seqlen_k_ptr != nullptr;
                    p.has_cu_seqlen_q_ptr = args.cu_seqlen_q_ptr != nullptr;
                    p.has_cu_seqlen_k_ptr = args.cu_seqlen_k_ptr != nullptr;
                }
            },
            invocation.args);

        return p;
    }
};

class FmhaProblemBuilder
{
    public:
    FmhaProblemBuilder() = default;

    FmhaProblemBuilder& api_family(FmhaApiFamily family)
    {
        problem_.api_family = family;
        return *this;
    }

    FmhaProblemBuilder& kernel_family(FmhaKernelFamily family)
    {
        problem_.requested_family = family;
        return *this;
    }

    FmhaProblemBuilder& gfx_arch(const std::string& arch)
    {
        problem_.gfx_arch = arch;
        return *this;
    }

    FmhaProblemBuilder& data_type(const std::string& dtype)
    {
        problem_.data_type = dtype;
        return *this;
    }

    FmhaProblemBuilder& dims(std::int64_t hdim_q,
                             std::int64_t hdim_v,
                             std::int64_t batch,
                             std::int64_t seqlen_q,
                             std::int64_t seqlen_k)
    {
        problem_.hdim_q   = hdim_q;
        problem_.hdim_v   = hdim_v;
        problem_.batch    = batch;
        problem_.seqlen_q = seqlen_q;
        problem_.seqlen_k = seqlen_k;
        return *this;
    }

    FmhaProblemBuilder& nheads(std::int64_t q, std::int64_t k)
    {
        problem_.nhead_q = q;
        problem_.nhead_k = k;
        return *this;
    }

    FmhaProblemBuilder& mask_type(int mask)
    {
        problem_.mask_type = mask;
        return *this;
    }

    FmhaProblemBuilder& bias_type(int bias)
    {
        problem_.bias_type = bias;
        return *this;
    }

    FmhaProblemBuilder& lse(bool value)
    {
        problem_.has_lse = value;
        return *this;
    }

    FmhaProblemBuilder& dropout(bool value)
    {
        problem_.has_dropout = value;
        return *this;
    }

    FmhaProblemBuilder& qscale_type(int qscale)
    {
        problem_.qscale_type = qscale;
        return *this;
    }

    FmhaProblemBuilder& rope_type(int rope)
    {
        problem_.rope_type = rope;
        return *this;
    }

    FmhaProblemBuilder& logits_soft_cap(bool value)
    {
        problem_.has_logits_soft_cap = value;
        return *this;
    }

    FmhaProblemBuilder& v_rowmajor(bool value)
    {
        problem_.is_v_rowmajor = value;
        return *this;
    }

    FmhaProblemBuilder& group_mode(bool value)
    {
        problem_.is_group_mode = value;
        return *this;
    }

    FmhaProblemBuilder& paged_kv(bool value)
    {
        problem_.use_paged_kv = value;
        return *this;
    }

    FmhaProblemBuilder& fp8_static_quant(bool value)
    {
        problem_.do_fp8_static_quant = value;
        return *this;
    }

    FmhaProblemBuilder& skip_min_seqlen_q(bool value)
    {
        problem_.skip_min_seqlen_q = value;
        return *this;
    }

    FmhaProblemBuilder& sink(bool value)
    {
        problem_.has_sink = value;
        return *this;
    }

    FmhaProblemBuilder& kv_cache(int memory_layout, int lookup_table, int page_size)
    {
        problem_.kv_memory_layout = memory_layout;
        problem_.kv_lookup_table  = lookup_table;
        problem_.page_size        = page_size;
        return *this;
    }

    FmhaProblemBuilder& window(std::int64_t left, std::int64_t right)
    {
        problem_.window_size_left  = left;
        problem_.window_size_right = right;
        return *this;
    }

    FmhaProblemBuilder& sink_size(std::int64_t value)
    {
        problem_.sink_size = value;
        problem_.has_sink  = (value > 0);
        return *this;
    }

    FmhaProblemBuilder& max_seqlen(std::int64_t q, std::int64_t k)
    {
        problem_.max_seqlen_q = q;
        problem_.max_seqlen_k = k;
        return *this;
    }

    FmhaProblemBuilder& num_splits(std::int64_t value)
    {
        problem_.num_splits = value;
        return *this;
    }

    FmhaProblemBuilder& bwd_flags(bool dbias, bool store_randval, bool deterministic)
    {
        problem_.has_dbias        = dbias;
        problem_.is_store_randval = store_randval;
        problem_.is_deterministic = deterministic;
        return *this;
    }

    [[nodiscard]] FmhaProblem build() const
    {
        if(!problem_.is_valid())
        {
            throw std::invalid_argument("Invalid FMHA problem: " + problem_.to_string());
        }

        const auto fam = problem_.api_family;
        if(fam == FmhaApiFamily::Bwd)
        {
            if(problem_.has_lse == false)
            {
                throw std::invalid_argument(
                    "FMHA BWD requires has_lse=true (LSE from forward pass)");
            }
        }

        if(problem_.is_group_mode && problem_.max_seqlen_q <= 0)
        {
            throw std::invalid_argument("FMHA group mode requires max_seqlen_q > 0");
        }

        return problem_;
    }

    private:
    FmhaProblem problem_;
};

// =============================================================================
// Backward workspace sizing
// =============================================================================

struct FmhaBwdWorkspaceInfo
{
    size_t d_bytes         = 0; // B * Hq * Sq * sizeof(float)
    size_t dq_acc_bytes    = 0; // B * Hq * Sq * Dq * sizeof(float)
    size_t rand_val_bytes  = 0; // 0 unless is_store_randval
    size_t total_bytes     = 0; // aligned sum
    size_t d_offset        = 0; // always 0
    size_t dq_acc_offset   = 0; // align(d_bytes, 256)
    size_t rand_val_offset = 0; // align(d_bytes + dq_acc_bytes, 256)
};

inline FmhaBwdWorkspaceInfo bwd_workspace_info(const FmhaProblem& problem)
{
    constexpr size_t kAlign = 256;
    auto align_up           = [](size_t n, size_t a) -> size_t { return (n + a - 1) / a * a; };

    FmhaBwdWorkspaceInfo info;
    const auto B  = static_cast<size_t>(problem.batch);
    const auto Hq = static_cast<size_t>(problem.nhead_q);
    const auto Sq = static_cast<size_t>(problem.seqlen_q);
    const auto Dq = static_cast<size_t>(problem.hdim_q);
    const auto Sk = static_cast<size_t>(problem.seqlen_k);

    info.d_bytes      = B * Hq * Sq * sizeof(float);
    info.dq_acc_bytes = B * Hq * Sq * Dq * sizeof(float);

    if(problem.is_store_randval)
        info.rand_val_bytes = B * Hq * Sq * Sk * sizeof(uint8_t);

    info.d_offset        = 0;
    info.dq_acc_offset   = align_up(info.d_bytes, kAlign);
    info.rand_val_offset = align_up(info.dq_acc_offset + info.dq_acc_bytes, kAlign);
    info.total_bytes     = info.rand_val_bytes > 0
                               ? align_up(info.rand_val_offset + info.rand_val_bytes, kAlign)
                               : align_up(info.dq_acc_offset + info.dq_acc_bytes, kAlign);

    return info;
}

} // namespace dispatcher
} // namespace ck_tile
