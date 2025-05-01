// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>
#include <filesystem>
#include <memory>

#include "ck/version.h"
#include "ck/host_utility/device_prop.hpp"

class Profiler
{
    public:
    static Profiler& instance()
    {
        static Profiler instance;
        return instance;
    }

    static std::string get_rocm_version()
    {
        std::ifstream version_file("/opt/rocm/.info/version");
        if(version_file.is_open())
        {
            std::string version;
            std::getline(version_file, version);
            return version;
        }
        return "Unknown";
    }

    template <typename Kernel>
    void benchmark_kernel(ck_tile::DeviceMem& c_m_n_dev_buf,
                          ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                          ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                          int verify,
                          ck_tile::GemmHostArgs& args,
                          const ck_tile::stream_config& s)
    {
        std::string description = Kernel::get_name();

        GemmProblem problem{args.k_batch,
                            args.M,
                            args.N,
                            args.K,
                            args.stride_A,
                            args.stride_B,
                            args.stride_C,
                            DataTypeTraits<ADataType>::name,
                            DataTypeTraits<BDataType>::name,
                            DataTypeTraits<AccDataType>::name,
                            DataTypeTraits<CDataType>::name,
                            ALayout::name,
                            BLayout::name,
                            CLayout::name};

        KernelInstance kernel_instance{environment_, description, problem, {-1.0f, -1.0f, -1.0f}};

        float avg_time = Kernel::launch(args, s);
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

        std::size_t flop     = std::size_t(2) * args.M * args.N * args.K;
        std::size_t num_byte = sizeof(ADataType) * args.M * args.K +
                               sizeof(BDataType) * args.N * args.K +
                               sizeof(CDataType) * args.M * args.N;
        float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
        float gb_per_sec = num_byte / 1.E6 / avg_time;

        kernel_instance.perf_result.latency   = avg_time;
        kernel_instance.perf_result.tflops    = tflops;
        kernel_instance.perf_result.bandwidth = gb_per_sec;

        std::cout << kernel_instance << std::endl;

        bool verified_correct =
            !verify || compare(args.K, args.k_batch, c_m_n_dev_result, c_m_n_host_result);

        if(verified_correct)
        {
            kernel_instances_.emplace_back(kernel_instance);
        }
        else
        {
            std::cout << "Verification failed, skip kernel: " << description << std::endl;
        }

        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();
    }

    KernelInstance select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto kernel_instance = *std::max_element(kernel_instances_.begin(),
                                                 kernel_instances_.end(),
                                                 [metric](const auto& a, const auto& b) {
                                                     return PerformanceResult::compare(
                                                         b.perf_result, a.perf_result, metric);
                                                 });

        std::cout << "**********************************" << std::endl;
        std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                  << "The best kernel instance is: " << kernel_instance << std::endl;
        std::cout << "**********************************" << std::endl;

        return kernel_instance;
    }

    private:
    Profiler()
    {
        environment_ = Environment{
            get_rocm_version(),
            ck::get_device_name(),
        };
    }
    ~Profiler() { kernel_instances_.clear(); }

    Profiler(const Profiler&)            = delete;
    Profiler& operator=(const Profiler&) = delete;

    Environment environment_;

    std::vector<KernelInstance> kernel_instances_;
};
