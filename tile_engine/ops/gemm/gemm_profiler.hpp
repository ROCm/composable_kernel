// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <type_traits>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gemm_benchmark.hpp"

template <typename T, typename = void>
struct has_split_k_member : std::false_type
{
};

template <typename T>
struct has_split_k_member<T, std::void_t<decltype(std::declval<T>().split_k_)>> : std::true_type
{
};

template <typename Gemm, typename Problem, typename GemmArgs>
class GemmProfiler
{
    public:
    static Gemm& instance(Settings setting)
    {
        static Gemm instance{setting};
        return instance;
    }

    // Overload for single kernel benchmarking
    void benchmark(Problem& gemm_problem,
                   std::function<float(const GemmArgs&, const ck_tile::stream_config&)> kernel_func)
    {
        // Create a vector with a single callable that returns both name and time
        std::vector<
            std::function<std::tuple<std::string, float>(GemmArgs&, const ck_tile::stream_config&)>>
            callables;

        callables.push_back([kernel_func](GemmArgs& args, const ck_tile::stream_config& stream) {
            float time = kernel_func(args, stream);
            return std::make_tuple(std::string(KERNEL_NAME), time);
        });

        benchmark(gemm_problem, callables);
    }

    virtual void benchmark(Problem& gemm_problem,
                           std::vector<std::function<std::tuple<std::string, float>(
                               GemmArgs&, const ck_tile::stream_config&)>>& callables) = 0;

    void process_result(const Problem& gemm_problem,
                        ck_tile::DeviceMem& c_m_n_dev_buf,
                        ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                        ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                        const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;

        KernelInstance<Problem> kernel_instance{name, gemm_problem, {-1.0f, -1.0f, -1.0f}};

        // compute performance metric
        std::size_t flop     = get_flop_count(gemm_problem);
        std::size_t num_byte = get_byte_count(gemm_problem);

        // update
        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log > 0 && !setting_.json_output)
        {
            std::cout << kernel_instance << std::endl;
        }

        // verify result
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool verified_correct =
            !setting_.verify || compare<Problem>(name,
                                                 gemm_problem.k_,
                                                 get_verification_split_k(gemm_problem),
                                                 c_m_n_dev_result,
                                                 c_m_n_host_result);

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

    KernelInstance<Problem> select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto kernel_instance = *std::max_element(kernel_instances_.begin(),
                                                 kernel_instances_.end(),
                                                 [metric](const auto& a, const auto& b) {
                                                     return PerformanceResult::compare(
                                                         b.perf_result_, a.perf_result_, metric);
                                                 });

        if(setting_.json_output)
        {
            // Output clean JSON only
            std::cout << kernel_instance << std::endl;
        }
        else
        {
            std::cout << "**********************************" << std::endl;
            std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                      << "Current kernel performance is: " << kernel_instance << std::endl;
            std::cout << "**********************************" << std::endl;
        }

        if(!setting_.csv_filename.empty())
        {
            std::ofstream file(setting_.csv_filename + ".csv", std::ios::app);

            if(!file.is_open())
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
            else
            {
                if(file.tellp() == 0)
                {
                    write_csv_header(file);
                }

                write_csv_row(file, kernel_instance, metric);

                if(!file)
                {
                    std::cerr << "Warning: Error occurred while writing to CSV file." << std::endl;
                }
            }
        }

        return kernel_instance;
    }

    GemmProfiler(const GemmProfiler&)            = delete;
    GemmProfiler& operator=(const GemmProfiler&) = delete;

    protected:
    virtual ~GemmProfiler() { kernel_instances_.clear(); }
    GemmProfiler(Settings setting) : setting_(setting) {}

    virtual std::size_t get_flop_count(const Problem& gemm_problem) const
    {
        using DDataType = typename get_DsDataType<Problem>::type;

        std::size_t flop = std::size_t(2) * gemm_problem.m_ * gemm_problem.n_ * gemm_problem.k_;
        if constexpr(!std::is_void_v<DDataType>)
        {
            ck_tile::static_for<0, DDataType::size(), 1>{}([&](auto i) {
                using DType = ck_tile::remove_cvref_t<std::tuple_element_t<i, DDataType>>;
                static_cast<void>(sizeof(DType));
                flop += gemm_problem.m_ * gemm_problem.n_;
            });
        }
        return flop;
    }

    virtual std::size_t get_byte_count(const Problem& gemm_problem) const
    {
        using DDataType = typename get_DsDataType<Problem>::type;

        std::size_t num_byte = sizeof(ADataType) * gemm_problem.m_ * gemm_problem.k_ +
                               sizeof(BDataType) * gemm_problem.n_ * gemm_problem.k_ +
                               sizeof(CDataType) * gemm_problem.m_ * gemm_problem.n_;

        if constexpr(!std::is_void_v<DDataType>)
        {
            ck_tile::static_for<0, DDataType::size(), 1>{}([&](auto i) {
                using DType = ck_tile::remove_cvref_t<std::tuple_element_t<i, DDataType>>;
                num_byte += sizeof(DType) * gemm_problem.m_ * gemm_problem.n_;
            });
        }
        return num_byte;
    }

    virtual int get_verification_split_k(const Problem& gemm_problem) const
    {
        if constexpr(has_split_k_member<Problem>::value)
        {
            return gemm_problem.split_k_;
        }
        return 1;
    }

    virtual void write_csv_header(std::ostream& os) const
    {
        os << "rocm_version,device_name," << "split_k,m,n,k,stride_a,stride_b,stride_c,"
           << "dtype_a,dtype_b,dtype_acc,dtype_c," << "layout_a,layout_b,layout_c,"
           << "structured_sparsity," << "name,"
           << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
    }

    virtual void write_csv_row(std::ostream& os,
                               const KernelInstance<Problem>& kernel_instance,
                               Metric metric) const
    {
        const auto& problem = kernel_instance.problem_;
        const auto& name    = kernel_instance.name_;
        const auto& perf    = kernel_instance.perf_result_;

        os << get_rocm_version() << "," << ck_tile::get_device_name() << ","
           << get_verification_split_k(problem) << "," << problem.m_ << "," << problem.n_ << ","
           << problem.k_ << "," << problem.stride_a_ << "," << problem.stride_b_ << ","
           << problem.stride_c_ << "," << problem.dtype_a_ << "," << problem.dtype_b_ << ","
           << problem.dtype_acc_ << "," << problem.dtype_c_ << "," << problem.layout_a_ << ","
           << problem.layout_b_ << "," << problem.layout_c_ << "," << problem.structured_sparsity_
           << "," << name << "," << std::fixed << std::setprecision(4) << perf.latency_ << ","
           << std::fixed << std::setprecision(4) << perf.tflops_ << "," << std::fixed
           << std::setprecision(4) << perf.bandwidth_ << "," << get_metric_name(metric) << "\n";
    }

    Settings setting_;

    std::vector<KernelInstance<Problem>> kernel_instances_;
};
