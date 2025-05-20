// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>

#include "ck_tile/host/device_prop.hpp"
#include "benchmark_gemm.hpp"

class GemmProfiler
{
    public:
    static GemmProfiler& instance(Setting setting)
    {
        static GemmProfiler instance{setting};
        return instance;
    }

    void benchmark(const GemmProblem& gemm_problem,
                   ck_tile::DeviceMem& c_m_n_dev_buf,
                   ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                   ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                   const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;

        KernelInstance kernel_instance{name, gemm_problem, {-1.0f, -1.0f, -1.0f}};

        // compute performance metric
        std::size_t flop     = std::size_t(2) * gemm_problem.m_ * gemm_problem.n_ * gemm_problem.k_;
        std::size_t num_byte = sizeof(ADataType) * gemm_problem.m_ * gemm_problem.k_ +
                               sizeof(BDataType) * gemm_problem.n_ * gemm_problem.k_ +
                               sizeof(CDataType) * gemm_problem.m_ * gemm_problem.n_;

        // update
        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log_ > 0)
        {
            std::cout << kernel_instance << std::endl;
        }

        // verify result
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool verified_correct =
            !setting_.verify_ ||
            compare(gemm_problem.k_, gemm_problem.split_k_, c_m_n_dev_result, c_m_n_host_result);

        if(verified_correct)
        {
            kernel_instances_.emplace_back(kernel_instance);
        }
        else
        {
            std::cout << "Verification failed, skip kernel: " << name << std::endl;
        }

        // clear tensor
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
                                                         b.perf_result_, a.perf_result_, metric);
                                                 });

        std::cout << "**********************************" << std::endl;
        std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                  << "The best kernel instance is: " << kernel_instance << std::endl;
        std::cout << "**********************************" << std::endl;

        if(!setting_.csv_filename_.empty())
        {
            std::ofstream file(setting_.csv_filename_ + ".csv", std::ios::app);

            if(!file.is_open())
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
            else
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version,device_name,"
                         << "split_k,m,n,k,stride_a,stride_b,stride_c,"
                         << "dtype_a,dtype_b,dtype_acc,dtype_c,"
                         << "layout_a,layout_b,layout_c,"
                         << "structured_sparsity,"
                         << "name,"
                         << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& problem = kernel_instance.problem_;
                const auto& name    = kernel_instance.name_;
                const auto& perf    = kernel_instance.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << ","
                     << problem.split_k_ << "," << problem.m_ << "," << problem.n_ << ","
                     << problem.k_ << "," << problem.stride_a_ << "," << problem.stride_b_ << ","
                     << problem.stride_c_ << "," << problem.dtype_a_ << "," << problem.dtype_b_
                     << "," << problem.dtype_acc_ << "," << problem.dtype_c_ << ","
                     << problem.layout_a_ << "," << problem.layout_b_ << "," << problem.layout_c_
                     << "," << problem.structured_sparsity_ << "," << name << "," << std::fixed
                     << std::setprecision(4) << perf.latency_ << "," << std::fixed
                     << std::setprecision(4) << perf.tflops_ << "," << std::fixed
                     << std::setprecision(4) << perf.bandwidth_ << "," << get_metric_name(metric)
                     << "\n";

                if(!file)
                {
                    std::cerr << "Warning: Error occurred while writing to CSV file." << std::endl;
                }
            }
        }

        return kernel_instance;
    }

    GemmProfiler(const GemmProfiler&) = delete;
    GemmProfiler& operator=(const GemmProfiler&) = delete;

    private:
    ~GemmProfiler() { kernel_instances_.clear(); }
    GemmProfiler(Setting setting) : setting_(setting) {}

    Setting setting_;

    std::vector<KernelInstance> kernel_instances_;
};
