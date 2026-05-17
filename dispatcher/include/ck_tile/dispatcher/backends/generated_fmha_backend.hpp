// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/fmha_kernel_instance.hpp"

#include <functional>
#include <stdexcept>
#include <utility>

namespace ck_tile {
namespace dispatcher {
namespace backends {

// mask_top_left(1) and mask_bottom_right(2) share the same compiled kernel
// (both use SimplifiedGenericAttentionMask<true>). The actual mask
// coordinates are determined at runtime from the args, not the template.
inline bool fmha_mask_compatible(int kernel_mask, int problem_mask)
{
    if(kernel_mask == problem_mask)
        return true;
    // Both causal variants are served by the same kernel
    constexpr int kTopLeft     = 1; // mask_enum::mask_top_left
    constexpr int kBottomRight = 2; // mask_enum::mask_bottom_right
    if((kernel_mask == kTopLeft || kernel_mask == kBottomRight) &&
       (problem_mask == kTopLeft || problem_mask == kBottomRight))
        return true;
    return false;
}

inline bool fmha_signature_matches(const FmhaKernelKey& key, const FmhaProblem& problem)
{
    const auto& sig = key.signature;
    const bool compare_page_size =
        sig.family == FmhaKernelFamily::FwdPagedKv ||
        problem.requested_family == FmhaKernelFamily::FwdPagedKv ||
        sig.family == FmhaKernelFamily::FwdAppendKv ||
        problem.requested_family == FmhaKernelFamily::FwdAppendKv ||
        sig.family == FmhaKernelFamily::FwdSplitKv ||
        problem.requested_family == FmhaKernelFamily::FwdSplitKv ||
        sig.family == FmhaKernelFamily::FwdSplitKvCombine ||
        problem.requested_family == FmhaKernelFamily::FwdSplitKvCombine ||
        sig.family == FmhaKernelFamily::BatchPrefill ||
        problem.requested_family == FmhaKernelFamily::BatchPrefill;
    const bool compare_kv_layout_lookup =
        sig.family == FmhaKernelFamily::BatchPrefill ||
        problem.requested_family == FmhaKernelFamily::BatchPrefill;

    if(!(sig.family == problem.requested_family && sig.data_type == problem.data_type &&
         sig.is_group_mode == problem.is_group_mode && sig.is_v_rowmajor == problem.is_v_rowmajor &&
         sig.has_logits_soft_cap == problem.has_logits_soft_cap &&
         fmha_mask_compatible(sig.mask_type, problem.mask_type) &&
         sig.bias_type == problem.bias_type && sig.has_lse == problem.has_lse &&
         sig.has_dropout == problem.has_dropout && sig.qscale_type == problem.qscale_type &&
         sig.rope_type == problem.rope_type && sig.use_paged_kv == problem.use_paged_kv &&
         sig.do_fp8_static_quant == problem.do_fp8_static_quant &&
         sig.skip_min_seqlen_q == problem.skip_min_seqlen_q && sig.has_sink == problem.has_sink &&
         sig.has_dbias == problem.has_dbias && sig.is_store_randval == problem.is_store_randval &&
         sig.is_deterministic == problem.is_deterministic && problem.hdim_q <= sig.hdim_q &&
         problem.hdim_v <= sig.hdim_v))
    {
        return false;
    }

    if(compare_kv_layout_lookup)
    {
        if(sig.kv_memory_layout != problem.kv_memory_layout ||
           sig.kv_lookup_table != problem.kv_lookup_table)
        {
            return false;
        }
    }

    if(compare_page_size && sig.page_size > 1 && sig.page_size != problem.page_size)
    {
        return false;
    }

    return true;
}

inline bool fmha_algorithm_supports(const FmhaKernelKey& key, const FmhaProblem& problem)
{
    const auto& alg = key.algorithm;

    if(problem.is_group_mode && problem.max_seqlen_q <= 0)
    {
        return false;
    }

    if(!alg.pad_s && alg.tile_shape.m0 > 0 &&
       problem.effective_max_seqlen_q() % alg.tile_shape.m0 != 0)
    {
        return false;
    }

    if(!alg.pad_sk)
    {
        if(problem.has_variable_seqlen_k())
        {
            return false;
        }
        if(alg.tile_shape.n0 > 0 && problem.effective_max_seqlen_k() % alg.tile_shape.n0 != 0)
        {
            return false;
        }
    }

    if(!alg.pad_d && alg.hdim_q_alignment > 0 && problem.hdim_q % alg.hdim_q_alignment != 0)
    {
        return false;
    }

    if(!alg.pad_dv && alg.hdim_v_alignment > 0 && problem.hdim_v % alg.hdim_v_alignment != 0)
    {
        return false;
    }

    if(alg.max_seq_len_q > 0 && problem.effective_max_seqlen_q() > alg.max_seq_len_q)
    {
        return false;
    }

    if(alg.max_splits_log2 > 0 &&
       problem.num_splits > (static_cast<std::int64_t>(1) << alg.max_splits_log2))
    {
        return false;
    }

    return true;
}

class GeneratedFmhaKernelInstance : public FmhaKernelInstance
{
    public:
    using SupportsFn = std::function<bool(const FmhaProblem&)>;
    using LaunchFn   = std::function<void(const FmhaInvocation&, const ck_tile::stream_config&)>;
    using RunFn      = std::function<float(const FmhaInvocation&, const ck_tile::stream_config&)>;

    GeneratedFmhaKernelInstance(FmhaKernelKey key,
                                std::string name,
                                SupportsFn supports_fn,
                                LaunchFn launch_fn,
                                RunFn run_fn = {})
        : key_(std::move(key)),
          name_(std::move(name)),
          supports_fn_(std::move(supports_fn)),
          launch_fn_(std::move(launch_fn)),
          run_fn_(std::move(run_fn))
    {
    }

    [[nodiscard]] const FmhaKernelKey& get_key() const override { return key_; }

    [[nodiscard]] bool supports(const FmhaProblem& problem) const override
    {
        return supports_fn_ ? supports_fn_(problem) : false;
    }

    [[nodiscard]] std::string get_name() const override { return name_; }

    void launch(const FmhaInvocation& invocation,
                const ck_tile::stream_config& stream_config) const override
    {
        if(!launch_fn_)
        {
            throw std::runtime_error("FMHA kernel launch function is not available");
        }
        launch_fn_(invocation, stream_config);
    }

    [[nodiscard]] float run(const FmhaInvocation& invocation,
                            const ck_tile::stream_config& stream_config) const override
    {
        if(run_fn_)
        {
            return run_fn_(invocation, stream_config);
        }
        return FmhaKernelInstance::run(invocation, stream_config);
    }

    private:
    FmhaKernelKey key_;
    std::string name_;
    SupportsFn supports_fn_;
    LaunchFn launch_fn_;
    RunFn run_fn_;
};

inline GeneratedFmhaKernelInstance::SupportsFn
make_default_supports_fn(const FmhaKernelKey& key,
                         GeneratedFmhaKernelInstance::SupportsFn extra = {})
{
    return [key, extra = std::move(extra)](const FmhaProblem& problem) {
        if(!fmha_signature_matches(key, problem) || !fmha_algorithm_supports(key, problem))
        {
            return false;
        }
        return extra ? extra(problem) : true;
    };
}

template <typename ArgsType, typename LaunchCallable>
inline FmhaKernelInstancePtr
make_oneshot_fmha_kernel(FmhaKernelKey key,
                         std::string name,
                         LaunchCallable&& launch_callable,
                         GeneratedFmhaKernelInstance::SupportsFn extra_support = {})
{
    auto launch_fn = [launch_callable = std::forward<LaunchCallable>(launch_callable)](
                         const FmhaInvocation& invocation, const ck_tile::stream_config& sc) {
        const auto* args = std::get_if<ArgsType>(&invocation.args);
        if(!args)
        {
            throw std::invalid_argument("FMHA invocation args do not match generated kernel type");
        }
        launch_callable(sc, *args);
    };

    auto supports_fn = make_default_supports_fn(key, std::move(extra_support));
    return std::make_shared<GeneratedFmhaKernelInstance>(
        std::move(key), std::move(name), std::move(supports_fn), std::move(launch_fn));
}

template <typename ArgsType, typename TimedCallable>
inline FmhaKernelInstancePtr
make_timed_fmha_kernel(FmhaKernelKey key,
                       std::string name,
                       TimedCallable&& timed_callable,
                       GeneratedFmhaKernelInstance::SupportsFn extra_support = {})
{
    auto callable = std::forward<TimedCallable>(timed_callable);

    auto launch_fn = [callable](const FmhaInvocation& invocation,
                                const ck_tile::stream_config& sc) {
        const auto* args = std::get_if<ArgsType>(&invocation.args);
        if(!args)
        {
            throw std::invalid_argument("FMHA invocation args do not match generated kernel type");
        }
        auto untimed         = sc;
        untimed.time_kernel_ = false;
        (void)callable(untimed, *args);
    };

    auto run_fn = [callable](const FmhaInvocation& invocation, const ck_tile::stream_config& sc) {
        const auto* args = std::get_if<ArgsType>(&invocation.args);
        if(!args)
        {
            throw std::invalid_argument("FMHA invocation args do not match generated kernel type");
        }
        return callable(sc, *args);
    };

    auto supports_fn = make_default_supports_fn(key, std::move(extra_support));
    return std::make_shared<GeneratedFmhaKernelInstance>(std::move(key),
                                                         std::move(name),
                                                         std::move(supports_fn),
                                                         std::move(launch_fn),
                                                         std::move(run_fn));
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
