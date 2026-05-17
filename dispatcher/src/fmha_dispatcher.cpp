// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include "ck_tile/dispatcher/fmha_dispatcher.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ck_tile {
namespace dispatcher {

FmhaDispatcher::FmhaDispatcher(FmhaRegistry* registry, const std::string& gfx_arch)
    : registry_(registry ? registry : &FmhaRegistry::instance()),
      heuristic_(nullptr),
      strategy_(SelectionStrategy::FirstFit),
      gfx_arch_(gfx_arch)
{
}

void FmhaDispatcher::set_heuristic(FmhaHeuristicFunction heuristic)
{
    heuristic_ = std::move(heuristic);
    if(heuristic_)
    {
        strategy_ = SelectionStrategy::Heuristic;
    }
}

void FmhaDispatcher::set_strategy(SelectionStrategy strategy) { strategy_ = strategy; }

void FmhaDispatcher::set_timing(int cold_niters, int nrepeat)
{
    cold_niters_ = cold_niters;
    nrepeat_     = nrepeat;
}

FmhaKernelInstancePtr FmhaDispatcher::select_kernel(const FmhaProblem& problem) const
{
    if(!problem.is_valid())
    {
        return nullptr;
    }

    switch(strategy_)
    {
    case SelectionStrategy::FirstFit: return select_first_fit(problem);
    case SelectionStrategy::Heuristic: return select_heuristic(problem);
    default: return nullptr;
    }
}

FmhaExecutionPlan FmhaDispatcher::plan_single_stage(const FmhaProblem& problem,
                                                    FmhaKernelFamily family) const
{
    FmhaExecutionPlan plan;
    plan.api_family = problem.api_family;

    auto stage_problem = with_family(problem, family);
    auto kernel        = select_kernel(stage_problem);
    if(kernel)
    {
        plan.stages.push_back({family, kernel->get_key().encode_identifier()});
    }
    return plan;
}

FmhaExecutionPlan FmhaDispatcher::plan(const FmhaProblem& problem) const
{
    switch(problem.api_family)
    {
    case FmhaApiFamily::Fwd: return plan_single_stage(problem, FmhaKernelFamily::Fwd);
    case FmhaApiFamily::FwdPagedKv: return plan_single_stage(problem, FmhaKernelFamily::FwdPagedKv);
    case FmhaApiFamily::FwdAppendKv:
        return plan_single_stage(problem, FmhaKernelFamily::FwdAppendKv);
    case FmhaApiFamily::BatchPrefill:
        return plan_single_stage(problem, FmhaKernelFamily::BatchPrefill);
    case FmhaApiFamily::FwdSplitKv: {
        FmhaExecutionPlan plan;
        plan.api_family = problem.api_family;

        auto split_problem = with_family(problem, FmhaKernelFamily::FwdSplitKv);
        auto split_kernel  = select_kernel(split_problem);
        if(!split_kernel)
        {
            return plan;
        }

        auto combine_problem = with_family(problem, FmhaKernelFamily::FwdSplitKvCombine);
        auto combine_kernel  = select_kernel(combine_problem);
        if(!combine_kernel)
        {
            return {};
        }

        plan.stages.push_back(
            {FmhaKernelFamily::FwdSplitKv, split_kernel->get_key().encode_identifier()});
        plan.stages.push_back(
            {FmhaKernelFamily::FwdSplitKvCombine, combine_kernel->get_key().encode_identifier()});
        return plan;
    }
    case FmhaApiFamily::Bwd: {
        FmhaExecutionPlan plan;
        plan.api_family = problem.api_family;

        auto dot_problem = with_family(problem, FmhaKernelFamily::BwdDotDoO);
        auto dot_kernel  = select_kernel(dot_problem);
        if(!dot_kernel)
        {
            return plan;
        }

        auto dq_problem = with_family(problem, FmhaKernelFamily::BwdDqDkDv);
        auto dq_kernel  = select_kernel(dq_problem);
        if(!dq_kernel)
        {
            return {};
        }

        plan.stages.push_back(
            {FmhaKernelFamily::BwdDotDoO, dot_kernel->get_key().encode_identifier()});
        plan.stages.push_back(
            {FmhaKernelFamily::BwdDqDkDv, dq_kernel->get_key().encode_identifier()});

        auto convert_problem = with_family(problem, FmhaKernelFamily::BwdConvertDq);
        auto convert_kernel  = select_kernel(convert_problem);
        if(convert_kernel)
        {
            plan.stages.push_back(
                {FmhaKernelFamily::BwdConvertDq, convert_kernel->get_key().encode_identifier()});
        }
        return plan;
    }
    default: return {};
    }
}

float FmhaDispatcher::run(const FmhaInvocation& invocation, void* stream) const
{
    auto problem = FmhaProblem::from_invocation(invocation, gfx_arch_);
    auto exec    = plan(problem);
    if(!exec.is_valid())
    {
        std::ostringstream oss;
        oss << "No suitable FMHA execution plan for API family " << to_string(problem.api_family)
            << " and dtype " << problem.data_type;
        throw NoKernelFound(oss.str());
    }

    return run_plan(exec, invocation, stream);
}

float FmhaDispatcher::run_explicit(const std::string& kernel_id,
                                   const FmhaInvocation& invocation,
                                   void* stream) const
{
    auto kernel = registry_->lookup(kernel_id);
    if(!kernel)
    {
        throw NoKernelFound("FMHA kernel not found: " + kernel_id);
    }
    auto sc = make_stream_config(stream);
    return kernel->run(invocation, sc);
}

float FmhaDispatcher::run_fwd(fmha_fwd_traits traits, fmha_fwd_args args, void* stream) const
{
    return run(FmhaInvocation::make(std::move(traits), std::move(args)), stream);
}

float FmhaDispatcher::run_fwd_pagedkv(fmha_fwd_pagedkv_traits traits,
                                      fmha_fwd_pagedkv_args args,
                                      void* stream) const
{
    return run(FmhaInvocation::make(std::move(traits), std::move(args)), stream);
}

float FmhaDispatcher::run_fwd_splitkv(fmha_fwd_splitkv_traits traits,
                                      fmha_fwd_splitkv_args args,
                                      void* stream) const
{
    return run(FmhaInvocation::make(std::move(traits), std::move(args)), stream);
}

float FmhaDispatcher::run_fwd_appendkv(fmha_fwd_appendkv_traits traits,
                                       fmha_fwd_appendkv_args args,
                                       void* stream) const
{
    return run(FmhaInvocation::make(std::move(traits), std::move(args)), stream);
}

float FmhaDispatcher::run_batch_prefill(fmha_batch_prefill_traits traits,
                                        fmha_batch_prefill_args args,
                                        void* stream) const
{
    return run(FmhaInvocation::make(std::move(traits), std::move(args)), stream);
}

float FmhaDispatcher::run_bwd(fmha_bwd_traits traits, fmha_bwd_args args, void* stream) const
{
    return run(FmhaInvocation::make(std::move(traits), std::move(args)), stream);
}

FmhaKernelInstancePtr FmhaDispatcher::select_first_fit(const FmhaProblem& problem) const
{
    // Seqtune-aware selection per fmhaarch.md Section 7.3.3:
    //   1. For short sequences (seqlen_q <= tile_m0): prefer smallest fitting tile
    //   2. tile_m0 == 64: unconditional fallback
    //   3. Prefer unpadded over padded
    //   4. Within same category: selection_rank, then smaller tile_m0

    auto kernels      = registry_->get_all();
    const auto max_sq = problem.effective_max_seqlen_q();

    // Find max tile_m0 across all compatible kernels
    int max_tile_m0_all = 0;
    for(const auto& kernel : kernels)
    {
        if(kernel->supports(problem))
        {
            max_tile_m0_all = std::max(max_tile_m0_all,
                                       static_cast<int>(kernel->get_key().algorithm.tile_shape.m0));
        }
    }

    FmhaKernelInstancePtr best           = nullptr;
    std::tuple<int, int, int> best_score = {std::numeric_limits<int>::max(),
                                            std::numeric_limits<int>::max(),
                                            std::numeric_limits<int>::max()};

    for(const auto& kernel : kernels)
    {
        if(!kernel->supports(problem))
            continue;

        const auto& key = kernel->get_key();
        int tile_m0     = key.algorithm.tile_shape.m0;
        int rank        = key.algorithm.selection_rank;
        bool aligned    = (tile_m0 > 0) && (max_sq > 0) && (max_sq % tile_m0 == 0);

        // Seqtune scoring (lower tuple is better):
        //   Category 0: seqlen_q <= tile_m0 AND aligned (perfect fit, smallest tile wins)
        //   Category 1: tile_m0 == 64 (unconditional fallback)
        //   Category 2: tile_m0 == max_tile_m0 (catch-all)
        //   Category 3: aligned (no padding needed)
        //   Category 4: needs padding (last resort)
        int category;
        if(tile_m0 > 0 && max_sq <= tile_m0 && aligned)
            category = 0;
        else if(tile_m0 == 64)
            category = 1;
        else if(tile_m0 == max_tile_m0_all)
            category = 2;
        else if(aligned)
            category = 3;
        else
            category = 4;

        auto score = std::make_tuple(category, rank, tile_m0);

        if(score < best_score)
        {
            best       = kernel;
            best_score = score;
        }
    }

    return best;
}

FmhaKernelInstancePtr FmhaDispatcher::select_heuristic(const FmhaProblem& problem) const
{
    if(!heuristic_)
    {
        return select_first_fit(problem);
    }

    for(const auto& kernel_id : heuristic_(problem))
    {
        auto kernel = registry_->lookup(kernel_id);
        if(kernel && kernel->supports(problem))
        {
            return kernel;
        }
    }

    return select_first_fit(problem);
}

FmhaProblem FmhaDispatcher::with_family(const FmhaProblem& base, FmhaKernelFamily family) const
{
    auto copy             = base;
    copy.requested_family = family;
    return copy;
}

float FmhaDispatcher::run_plan(const FmhaExecutionPlan& plan,
                               const FmhaInvocation& invocation,
                               void* stream) const
{
    auto sc = make_stream_config(stream);

    if(plan.stages.size() == 1)
    {
        auto kernel = registry_->lookup(plan.stages.front().kernel_id);
        if(!kernel)
        {
            throw NoKernelFound("Missing FMHA kernel: " + plan.stages.front().kernel_id);
        }
        return kernel->run(invocation, sc);
    }

    // Multi-stage lambdas capture by reference. This is safe because
    // launch_kernel dispatches all stages on the same HIP stream before
    // returning. If launch_kernel ever becomes async, these must capture
    // by value or use shared_ptr.
    if(plan.stages.size() == 2)
    {
        auto first  = registry_->lookup(plan.stages[0].kernel_id);
        auto second = registry_->lookup(plan.stages[1].kernel_id);
        if(!first || !second)
        {
            throw NoKernelFound("Missing FMHA kernel in two-stage plan");
        }

        return ck_tile::launch_kernel(
            sc,
            [&](const ck_tile::stream_config& inner) { first->launch(invocation, inner); },
            [&](const ck_tile::stream_config& inner) { second->launch(invocation, inner); });
    }

    if(plan.stages.size() == 3)
    {
        auto first  = registry_->lookup(plan.stages[0].kernel_id);
        auto second = registry_->lookup(plan.stages[1].kernel_id);
        auto third  = registry_->lookup(plan.stages[2].kernel_id);
        if(!first || !second || !third)
        {
            throw NoKernelFound("Missing FMHA kernel in three-stage plan");
        }

        return ck_tile::launch_kernel(
            sc,
            [&](const ck_tile::stream_config& inner) { first->launch(invocation, inner); },
            [&](const ck_tile::stream_config& inner) { second->launch(invocation, inner); },
            [&](const ck_tile::stream_config& inner) { third->launch(invocation, inner); });
    }

    throw std::runtime_error("Unsupported FMHA execution plan length");
}

ck_tile::stream_config FmhaDispatcher::make_stream_config(void* stream) const
{
    ck_tile::stream_config sc;
    sc.stream_id_      = reinterpret_cast<hipStream_t>(stream);
    sc.time_kernel_    = benchmarking_enabled_;
    sc.log_level_      = 0;
    sc.cold_niters_    = benchmarking_enabled_ ? cold_niters_ : 0;
    sc.nrepeat_        = benchmarking_enabled_ ? nrepeat_ : 1;
    sc.is_gpu_timer_   = benchmarking_enabled_;
    sc.flush_cache_    = false;
    sc.rotating_count_ = 1;
    return sc;
}

} // namespace dispatcher
} // namespace ck_tile
