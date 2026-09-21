// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/fmha_registry.hpp"

#include <functional>
#include <string>
#include <vector>

namespace ck_tile {
namespace dispatcher {

using FmhaHeuristicFunction = std::function<std::vector<std::string>(const FmhaProblem&)>;

struct FmhaExecutionStage
{
    FmhaKernelFamily family = FmhaKernelFamily::Fwd;
    std::string kernel_id;
};

struct FmhaExecutionPlan
{
    FmhaApiFamily api_family = FmhaApiFamily::Fwd;
    std::vector<FmhaExecutionStage> stages;

    [[nodiscard]] bool is_valid() const { return !stages.empty(); }
};

class FmhaDispatcher
{
    public:
    enum class SelectionStrategy
    {
        FirstFit,
        Heuristic
    };

    explicit FmhaDispatcher(FmhaRegistry* registry = nullptr, const std::string& gfx_arch = "");

    void set_heuristic(FmhaHeuristicFunction heuristic);
    void set_strategy(SelectionStrategy strategy);
    void set_timing(int cold_niters, int nrepeat);
    void set_arch(const std::string& arch) { gfx_arch_ = arch; }
    [[nodiscard]] const std::string& arch() const { return gfx_arch_; }

    [[nodiscard]] FmhaKernelInstancePtr select_kernel(const FmhaProblem& problem) const;
    [[nodiscard]] FmhaExecutionPlan plan(const FmhaProblem& problem) const;

    [[nodiscard]] float run(const FmhaInvocation& invocation, void* stream = nullptr) const;

    [[nodiscard]] float run_explicit(const std::string& kernel_id,
                                     const FmhaInvocation& invocation,
                                     void* stream = nullptr) const;

    [[nodiscard]] float
    run_fwd(fmha_fwd_traits traits, fmha_fwd_args args, void* stream = nullptr) const;
    [[nodiscard]] float run_fwd_pagedkv(fmha_fwd_pagedkv_traits traits,
                                        fmha_fwd_pagedkv_args args,
                                        void* stream = nullptr) const;
    [[nodiscard]] float run_fwd_splitkv(fmha_fwd_splitkv_traits traits,
                                        fmha_fwd_splitkv_args args,
                                        void* stream = nullptr) const;
    [[nodiscard]] float run_fwd_appendkv(fmha_fwd_appendkv_traits traits,
                                         fmha_fwd_appendkv_args args,
                                         void* stream = nullptr) const;
    [[nodiscard]] float run_batch_prefill(fmha_batch_prefill_traits traits,
                                          fmha_batch_prefill_args args,
                                          void* stream = nullptr) const;
    // run_bwd is available when bwd types exist (library builds, bwd kernel TUs,
    // or any TU that doesn't set CK_TILE_FMHA_BWD_TYPES_FROM_EXAMPLE).
    // In fwd-only TUs, bwd types come from the fallback in fmha_types.hpp.
    [[nodiscard]] float
    run_bwd(fmha_bwd_traits traits, fmha_bwd_args args, void* stream = nullptr) const;

    private:
    [[nodiscard]] FmhaKernelInstancePtr select_first_fit(const FmhaProblem& problem) const;
    [[nodiscard]] FmhaKernelInstancePtr select_heuristic(const FmhaProblem& problem) const;

    [[nodiscard]] FmhaProblem with_family(const FmhaProblem& base, FmhaKernelFamily family) const;
    [[nodiscard]] FmhaExecutionPlan plan_single_stage(const FmhaProblem& problem,
                                                      FmhaKernelFamily family) const;
    [[nodiscard]] float
    run_plan(const FmhaExecutionPlan& plan, const FmhaInvocation& invocation, void* stream) const;
    [[nodiscard]] ck_tile::stream_config make_stream_config(void* stream) const;

    FmhaRegistry* registry_;
    FmhaHeuristicFunction heuristic_;
    SelectionStrategy strategy_;
    std::string gfx_arch_;
    int cold_niters_           = 5;
    int nrepeat_               = 10;
    bool benchmarking_enabled_ = false;

    public:
    /// Enable or disable benchmarking (GPU timing).
    /// When disabled, kernels execute exactly once with no timing overhead
    /// (one-shot mode for production plugins).
    void set_benchmarking(bool enable) { benchmarking_enabled_ = enable; }
    [[nodiscard]] bool benchmarking_enabled() const { return benchmarking_enabled_; }
};

} // namespace dispatcher
} // namespace ck_tile
