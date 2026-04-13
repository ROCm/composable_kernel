// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/dispatcher_error.hpp"
#include <sstream>
#include <iostream>

namespace ck_tile {
namespace dispatcher {

Dispatcher::Dispatcher(Registry* registry, const std::string& gfx_arch)
    : registry_(registry ? registry : &Registry::instance()),
      heuristic_(nullptr),
      strategy_(SelectionStrategy::FirstFit),
      gfx_arch_(gfx_arch)
{
}

void Dispatcher::set_heuristic(HeuristicFunction heuristic)
{
    heuristic_ = heuristic;
    if(heuristic_)
    {
        strategy_ = SelectionStrategy::Heuristic;
    }
}

void Dispatcher::set_strategy(SelectionStrategy strategy) { strategy_ = strategy; }

KernelInstancePtr Dispatcher::select_kernel(const Problem& problem) const
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

float Dispatcher::run(
    const void* a_ptr, const void* b_ptr, void* c_ptr, const Problem& problem, void* stream) const
{
    return run_fused(a_ptr, b_ptr, c_ptr, nullptr, problem, stream);
}

float Dispatcher::run_fused(const void* a_ptr,
                            const void* b_ptr,
                            void* c_ptr,
                            const void** d_ptrs,
                            const Problem& problem,
                            void* stream) const
{
    auto kernel = select_kernel(problem);
    if(!kernel)
    {
        std::ostringstream oss;
        oss << "No suitable kernel found for problem: M=" << problem.M << " N=" << problem.N
            << " K=" << problem.K;
        throw NoKernelFound(oss.str());
    }

    return kernel->run(a_ptr, b_ptr, c_ptr, d_ptrs, problem, stream);
}

float Dispatcher::run_explicit(const std::string& kernel_id,
                               const void* a_ptr,
                               const void* b_ptr,
                               void* c_ptr,
                               const void** d_ptrs,
                               const Problem& problem,
                               void* stream) const
{
    auto kernel = registry_->lookup(kernel_id);
    if(!kernel)
    {
        throw NoKernelFound("Kernel not found: " + kernel_id);
    }

    if(!kernel->supports(problem))
    {
        std::ostringstream oss;
        oss << "Kernel " << kernel_id << " does not support problem: M=" << problem.M
            << " N=" << problem.N << " K=" << problem.K;
        throw UnsupportedProblem(oss.str());
    }

    return kernel->run(a_ptr, b_ptr, c_ptr, d_ptrs, problem, stream);
}

bool Dispatcher::validate(const void* a_ptr,
                          const void* b_ptr,
                          const void* c_ptr,
                          const void** d_ptrs,
                          const Problem& problem,
                          float tolerance) const
{
    auto kernel = select_kernel(problem);
    if(!kernel)
    {
        return false;
    }

    return kernel->validate(a_ptr, b_ptr, c_ptr, d_ptrs, problem, tolerance);
}

KernelInstancePtr Dispatcher::select_first_fit(const Problem& problem) const
{
    auto all_kernels = registry_->get_all();

    for(const auto& kernel : all_kernels)
    {
        if(kernel->supports(problem))
        {
            return kernel;
        }
    }

    return nullptr;
}

KernelInstancePtr Dispatcher::select_heuristic(const Problem& problem) const
{
    if(!heuristic_)
    {
        // Fall back to first-fit if no heuristic available
        return select_first_fit(problem);
    }

    // Get ranked list of kernel identifiers from heuristic
    auto candidates = heuristic_(problem);

    // Try each candidate in order
    for(const auto& kernel_id : candidates)
    {
        auto kernel = registry_->lookup(kernel_id);
        if(kernel && kernel->supports(problem))
        {
            return kernel;
        }
    }

    // If no heuristic candidate works, fall back to first-fit
    return select_first_fit(problem);
}

} // namespace dispatcher
} // namespace ck_tile
