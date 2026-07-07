// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "gemm_aquant_benchmark.hpp"

class AQuantGemmProfiler
{
    public:
    explicit AQuantGemmProfiler(Setting setting) : setting_(setting) {}

    AQuantGemmProfiler(const AQuantGemmProfiler&)            = delete;
    AQuantGemmProfiler& operator=(const AQuantGemmProfiler&) = delete;

    void reset() { kernel_instances_.clear(); }

    void benchmark(AQuantGemmProblem& problem,
                   std::function<float(const ck_tile::QuantGemmHostArgs&,
                                       const ck_tile::stream_config&)> kernel_func)
    {
        std::vector<std::function<std::tuple<std::string, float>(ck_tile::QuantGemmHostArgs&,
                                                                 const ck_tile::stream_config&)>>
            callables;

        callables.push_back(
            [kernel_func](ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& stream) {
                float time = kernel_func(args, stream);
                return std::make_tuple(std::string(KERNEL_NAME), time);
            });

        benchmark(problem, callables);
    }

    void benchmark(AQuantGemmProblem& problem,
                   std::vector<std::function<std::tuple<std::string, float>(
                       ck_tile::QuantGemmHostArgs&, const ck_tile::stream_config&)>>& callables)
    {
        const ALayout layout_a   = ALayout{};
        const BLayout layout_b   = BLayout{};
        const CLayout layout_c   = CLayout{};
        const AQLayout layout_aq = AQLayout{};

        problem.stride_a_ = ck_tile::get_default_stride(
            problem.m_, problem.k_, problem.stride_a_, is_row_major(layout_a));
        problem.stride_b_ = ck_tile::get_default_stride(
            problem.k_, problem.n_, problem.stride_b_, is_row_major(layout_b));
        problem.stride_c_ = ck_tile::get_default_stride(
            problem.m_, problem.n_, problem.stride_c_, is_row_major(layout_c));

        // Compute AQ scale tensor dimensions: [M, K / group_size_k]
        const ck_tile::index_t QK_A = problem.k_ / problem.group_size_k_;
        problem.stride_aq_          = ck_tile::get_default_stride(
            problem.m_, QK_A, problem.stride_aq_, is_row_major(layout_aq));

        // Allocate host tensors
        ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
            problem.m_, problem.k_, problem.stride_a_, is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
            problem.k_, problem.n_, problem.stride_b_, is_row_major(layout_b)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(ck_tile::host_tensor_descriptor(
            problem.m_, problem.n_, problem.stride_c_, is_row_major(layout_c)));

        // AQ scale tensor: [M, QK_A]
        ck_tile::HostTensor<AQDataType> aq_m_qk(ck_tile::host_tensor_descriptor(
            problem.m_, QK_A, problem.stride_aq_, is_row_major(layout_aq)));

        // Initialize tensors
        if(setting_.init_method_ == 0)
        {
            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
            ck_tile::FillUniformDistribution<AQDataType>{0.5f, 1.5f}(aq_m_qk);
        }
        else if(setting_.init_method_ == 1)
        {
            ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
            ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
            ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(1)}(aq_m_qk);
        }
        else if(setting_.init_method_ == 2)
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_m_k);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_k_n);
            ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(1)}(aq_m_qk);
        }
        else
        {
            a_m_k.SetZero();
            b_k_n.SetZero();
            aq_m_qk.SetZero();
        }

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());
        ck_tile::DeviceMem aq_dev_buf(aq_m_qk.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        aq_dev_buf.ToDevice(aq_m_qk.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        // Build QuantGemmHostArgs
        ck_tile::QuantGemmHostArgs gemm_args(a_m_k_dev_buf.GetDeviceBuffer(),
                                             b_k_n_dev_buf.GetDeviceBuffer(),
                                             c_m_n_dev_buf.GetDeviceBuffer(),
                                             aq_dev_buf.GetDeviceBuffer(),
                                             nullptr, // bq_ptr not used for AQuant
                                             problem.split_k_,
                                             problem.m_,
                                             problem.n_,
                                             problem.k_,
                                             QK_A,
                                             0, // QK_B not used for AQuant
                                             problem.stride_a_,
                                             problem.stride_b_,
                                             problem.stride_c_,
                                             problem.stride_aq_,
                                             0 // stride_BQ not used for AQuant
        );

        // Host reference for verification
        ck_tile::HostTensor<CDataType> c_m_n_host_result(ck_tile::host_tensor_descriptor(
            problem.m_, problem.n_, problem.stride_c_, is_row_major(layout_c)));

        if(setting_.verify_)
        {
            aquant_gemm_host_reference(setting_.verify_, a_m_k, aq_m_qk, b_k_n, c_m_n_host_result);
        }

        // Run kernel(s)
        for(auto& callable : callables)
        {
            auto kernel_run_result = callable(gemm_args,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     setting_.log_,
                                                                     setting_.n_warmup_,
                                                                     setting_.n_repeat_,
                                                                     setting_.is_gpu_timer_,
                                                                     setting_.flush_cache_,
                                                                     setting_.rotating_count_});
            process_result(problem,
                           QK_A,
                           c_m_n_dev_buf,
                           c_m_n_host_result,
                           c_m_n_dev_result,
                           kernel_run_result);
        }
    }

    void process_result(const AQuantGemmProblem& problem,
                        ck_tile::index_t QK_A,
                        ck_tile::DeviceMem& c_m_n_dev_buf,
                        ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                        ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                        const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;

        KernelInstance kernel_instance{name, problem, {-1.0f, -1.0f, -1.0f}};

        // Compute performance metrics
        std::size_t flop     = std::size_t(2) * problem.m_ * problem.n_ * problem.k_;
        std::size_t num_byte = sizeof(ADataType) * problem.m_ * problem.k_ +
                               sizeof(BDataType) * problem.n_ * problem.k_ +
                               sizeof(AQDataType) * problem.m_ * QK_A +
                               sizeof(CDataType) * problem.m_ * problem.n_;

        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log_ > 0 && !setting_.json_output_)
        {
            std::cout << kernel_instance << std::endl;
        }

        // Verify result
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool verified_correct =
            !setting_.verify_ ||
            compare_aquant(name, problem.k_, problem.split_k_, c_m_n_dev_result, c_m_n_host_result);

        if(verified_correct)
        {
            kernel_instances_.emplace_back(kernel_instance);
        }
        else
        {
            std::cout << "Verification failed, skip kernel: " << name << std::endl;
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
                                                         b.perf_result_, a.perf_result_, metric);
                                                 });

        if(setting_.json_output_)
        {
            std::cout << kernel_instance << std::endl;
        }
        else
        {
            std::cout << "**********************************" << std::endl;
            std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                      << "Current kernel performance is: " << kernel_instance << std::endl;
            std::cout << "**********************************" << std::endl;
        }

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
                         << "split_k,m,n,k,stride_a,stride_b,stride_c,stride_aq,group_size_k,"
                         << "dtype_a,dtype_b,dtype_aq,dtype_acc,dtype_c,"
                         << "layout_a,layout_b,layout_c," << "name,"
                         << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& prob = kernel_instance.problem_;
                const auto& perf = kernel_instance.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << ","
                     << prob.split_k_ << "," << prob.m_ << "," << prob.n_ << "," << prob.k_ << ","
                     << prob.stride_a_ << "," << prob.stride_b_ << "," << prob.stride_c_ << ","
                     << prob.stride_aq_ << "," << prob.group_size_k_ << "," << prob.dtype_a_ << ","
                     << prob.dtype_b_ << "," << prob.dtype_aq_ << "," << prob.dtype_acc_ << ","
                     << prob.dtype_c_ << "," << prob.layout_a_ << "," << prob.layout_b_ << ","
                     << prob.layout_c_ << "," << kernel_instance.name_ << "," << std::fixed
                     << std::setprecision(4) << perf.latency_ << "," << perf.tflops_ << ","
                     << perf.bandwidth_ << "," << get_metric_name(metric) << "\n";
            }
        }

        return kernel_instance;
    }

    private:
    Setting setting_;
    std::vector<KernelInstance> kernel_instances_;
};
