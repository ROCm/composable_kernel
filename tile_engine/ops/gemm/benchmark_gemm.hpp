// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>
#include <filesystem>
#include <memory>
#include <fstream>
#include <iomanip>

#include "ck/host_utility/device_prop.hpp"

enum class Metric
{
    LATENCY   = 0,
    TFLOPS    = 1,
    BANDWIDTH = 2
};

inline constexpr auto get_metric_name(Metric m)
{
    switch(m)
    {
    case Metric::LATENCY: return "latency";
    case Metric::TFLOPS: return "tflops";
    case Metric::BANDWIDTH: return "bandwidth";
    default: throw std::invalid_argument("Unsupported metric type");
    }
}

struct GemmProblem
{
    int split_k;
    int m, n, k;
    int stride_a, stride_b, stride_c;

    std::string dtype_a, dtype_b, dtype_acc, dtype_c;
    std::string layout_a, layout_b, layout_c;

    friend std::ostream& operator<<(std::ostream& os, const GemmProblem& problem)
    {
        os << "{\n"
           << "   \"split_k\":" << problem.split_k << ",\n"
           << "   \"m\":" << problem.m << ",\n"
           << "   \"n\":" << problem.n << ",\n"
           << "   \"k\":" << problem.k << ",\n"
           << "   \"stride_a\":" << problem.stride_a << ",\n"
           << "   \"stride_b\":" << problem.stride_b << ",\n"
           << "   \"stride_c\":" << problem.stride_c << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc << "\",\n"
           << "   \"dtype_c\":\"" << problem.dtype_c << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b << "\",\n"
           << "   \"layout_c\":\"" << problem.layout_c << "\"\n"
           << "}";
        return os;
    }
};

struct PerformanceResult
{
    double latency;
    double tflops;
    double bandwidth;

    static bool compare(const PerformanceResult& a, const PerformanceResult& b, Metric m)
    {
        switch(m)
        {
        case Metric::LATENCY: return a.latency < b.latency;
        case Metric::TFLOPS: return a.tflops > b.tflops;
        case Metric::BANDWIDTH: return a.bandwidth > b.bandwidth;
        default: throw std::invalid_argument("Unsupported metric type");
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const PerformanceResult& result)
    {
        os << "{\n"
           << "   \"latency(ms)\": " << std::fixed << std::setprecision(2) << result.latency
           << ",\n"
           << "   \"tflops(TFlops)\": " << result.tflops << ",\n"
           << "   \"bandwidth(GB/s)\": " << result.bandwidth << "\n"
           << "}";
        return os;
    }
};

struct KernelInstance
{
    std::string name;
    GemmProblem problem;
    PerformanceResult perf_result;

    static bool compare(const KernelInstance& a, const KernelInstance& b, Metric m)
    {
        return PerformanceResult::compare(a.perf_result, b.perf_result, m);
    }

    friend std::ostream& operator<<(std::ostream& os, const KernelInstance& obj)
    {
        os << "{\n"
           << " \"name\": \""
           << "{\n"
           << obj.name << "\n}"
           << "\",\n"
           << " \"problem\": \"" << obj.problem << "\",\n"
           << " \"perf_result\": " << obj.perf_result << "\n"
           << "}";
        return os;
    }
};

class GemmProfiler
{
    public:
    static GemmProfiler& instance()
    {
        static GemmProfiler instance;
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
                          const ck_tile::stream_config& stream)
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

        KernelInstance kernel_instance{description, problem, {-1.0f, -1.0f, -1.0f}};

        float avg_time = Kernel::launch(args, stream);
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

    KernelInstance select_best_instance(Metric metric,
                                        const std::string& csv_filename = "gemm_kernels.csv")
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

        if(!csv_filename.empty())
        {
            std::ofstream file(csv_filename, std::ios::app);

            if(!file.is_open())
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
            else
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version, device_name,"
                         << "split_k,m,n,k,stride_a,stride_b,stride_c,"
                         << "dtype_a,dtype_b,dtype_acc,dtype_c,"
                         << "layout_a,layout_b,layout_c,"
                         << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& p   = kernel_instance.problem;
                const auto& res = kernel_instance.perf_result;

                file << get_rocm_version() << "," << ck::get_device_name() << "," << p.split_k
                     << "," << p.m << "," << p.n << "," << p.k << "," << p.stride_a << ","
                     << p.stride_b << "," << p.stride_c << "," << p.dtype_a << "," << p.dtype_b
                     << "," << p.dtype_acc << "," << p.dtype_c << "," << p.layout_a << ","
                     << p.layout_b << "," << p.layout_c << "," << std::fixed << std::setprecision(2)
                     << res.latency << "," << std::fixed << std::setprecision(2) << res.tflops
                     << "," << std::fixed << std::setprecision(2) << res.bandwidth << ","
                     << get_metric_name(metric) << "\n";

                if(!file)
                {
                    std::cerr << "Warning: Error occurred while writing to CSV file." << std::endl;
                }
            }
        }

        return kernel_instance;
    }

    std::vector<KernelInstance> kernel_instances_;
};
