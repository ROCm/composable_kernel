// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "pool_benchmark.hpp"

class PoolProfiler
{
    public:
    static PoolProfiler& instance(Settings settings)
    {
        static PoolProfiler instance{settings};
        return instance;
    }

    // Overload for single kernel benchmarking
    void benchmark(PoolProblem& pool_problem,
                   std::function<float(const ck_tile::PoolHostArgs&, const ck_tile::stream_config&)>
                       kernel_func)
    {
        // Create a vector with a single callable that returns both name and time
        std::vector<std::function<std::tuple<std::string, float>(ck_tile::PoolHostArgs&,
                                                                 const ck_tile::stream_config&)>>
            callables;

        callables.push_back(
            [kernel_func](ck_tile::PoolHostArgs& args, const ck_tile::stream_config& stream) {
                float time = kernel_func(args, stream);
                return std::make_tuple(std::string(KERNEL_NAME), time);
            });

        benchmark(pool_problem, callables);
    }
////



////
    PoolProfiler(const PoolProfiler&)            = delete;
    PoolProfiler& operator=(const PoolProfiler&) = delete;

    private:
    ~PoolProfiler() { kernel_instances_.clear(); }
    PoolProfiler(Settings settings) : settings_(settings) {}

    Settings settings_;

    std::vector<KernelInstance> kernel_instances_;
}