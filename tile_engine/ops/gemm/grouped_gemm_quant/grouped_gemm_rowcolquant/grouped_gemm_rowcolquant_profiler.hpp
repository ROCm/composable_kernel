// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "grouped_gemm_rowcolquant_benchmark.hpp"

class GroupedRowColQuantGemmProfiler
{
    public:
    static GroupedRowColQuantGemmProfiler& instance(Settings settings)
    {
        static GroupedRowColQuantGemmProfiler instance{settings};
        return instance;
    }

    // Overload for single kernel benchmarking
    void benchmark(GroupedRowColQuantGemmProblem& problem,
                   std::function<float(const std::vector<ck_tile::QuantGroupedGemmHostArgs>&,
                                       const ck_tile::stream_config&,
                                       void*)> kernel_func)
    {
        // Create a vector with a single callable that returns both name and time
        std::vector<std::function<std::tuple<std::string, float>(
            std::vector<ck_tile::QuantGroupedGemmHostArgs>&, const ck_tile::stream_config&, void*)>>
            callables;

        callables.push_back([kernel_func](std::vector<ck_tile::QuantGroupedGemmHostArgs>& descs,
                                          const ck_tile::stream_config& stream,
                                          void* kargs_ptr) {
            float time = kernel_func(descs, stream, kargs_ptr);
            return std::make_tuple(std::string(KERNEL_NAME), time);
        });

        benchmark(problem, callables);
    }

    void benchmark(GroupedRowColQuantGemmProblem& problem,
                   std::vector<std::function<std::tuple<std::string, float>(
                       std::vector<ck_tile::QuantGroupedGemmHostArgs>&,
                       const ck_tile::stream_config&,
                       void*)>>& callables)
    {
        const ALayout layout_a = ALayout{};
        const BLayout layout_b = BLayout{};
        const CLayout layout_c = CLayout{};

        const int group_count = problem.group_count_;

        // Compute default strides for each group
        for(int i = 0; i < group_count; ++i)
        {
            problem.stride_As_[i] = ck_tile::get_default_stride(
                problem.Ms_[i], problem.Ks_[i], problem.stride_As_[i], is_row_major(layout_a));
            problem.stride_Bs_[i] = ck_tile::get_default_stride(
                problem.Ks_[i], problem.Ns_[i], problem.stride_Bs_[i], is_row_major(layout_b));
            problem.stride_Cs_[i] = ck_tile::get_default_stride(
                problem.Ms_[i], problem.Ns_[i], problem.stride_Cs_[i], is_row_major(layout_c));
        }

        // Create per-group tensors
        std::vector<ck_tile::HostTensor<ADataType>> a_tensors;
        std::vector<ck_tile::HostTensor<BDataType>> b_tensors;
        std::vector<ck_tile::HostTensor<CDataType>> c_dev_results;
        std::vector<ck_tile::HostTensor<AQDataType>> aq_tensors;
        std::vector<ck_tile::HostTensor<BQDataType>> bq_tensors;

        a_tensors.reserve(group_count);
        b_tensors.reserve(group_count);
        c_dev_results.reserve(group_count);
        aq_tensors.reserve(group_count);
        bq_tensors.reserve(group_count);

        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_dev_bufs;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_dev_bufs;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_dev_bufs;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> aq_dev_bufs;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> bq_dev_bufs;

        a_dev_bufs.reserve(group_count);
        b_dev_bufs.reserve(group_count);
        c_dev_bufs.reserve(group_count);
        aq_dev_bufs.reserve(group_count);
        bq_dev_bufs.reserve(group_count);

        std::vector<ck_tile::QuantGroupedGemmHostArgs> gemm_descs;
        gemm_descs.reserve(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = problem.Ms_[i];
            const ck_tile::index_t N = problem.Ns_[i];
            const ck_tile::index_t K = problem.Ks_[i];

            a_tensors.push_back(ck_tile::HostTensor<ADataType>(ck_tile::host_tensor_descriptor(
                M, K, problem.stride_As_[i], is_row_major(layout_a))));
            b_tensors.push_back(ck_tile::HostTensor<BDataType>(ck_tile::host_tensor_descriptor(
                K, N, problem.stride_Bs_[i], is_row_major(layout_b))));
            c_dev_results.push_back(ck_tile::HostTensor<CDataType>(ck_tile::host_tensor_descriptor(
                M, N, problem.stride_Cs_[i], is_row_major(layout_c))));

            // RowColQuant: AQ is per-row scale [M, 1], BQ is per-col scale [1, N]
            aq_tensors.push_back(ck_tile::HostTensor<AccDataType>(
                ck_tile::host_tensor_descriptor(M, 1, 0, is_row_major(layout_a))));
            bq_tensors.push_back(ck_tile::HostTensor<AccDataType>(
                ck_tile::host_tensor_descriptor(1, N, 0, is_row_major(layout_b))));

            if(settings_.init_method == 0)
            {
                ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_tensors[i]);
                ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_tensors[i]);
                ck_tile::FillUniformDistribution<AccDataType>{-1.f, 1.f}(aq_tensors[i]);
                ck_tile::FillUniformDistribution<AccDataType>{-1.f, 1.f}(bq_tensors[i]);
            }
            else if(settings_.init_method == 1)
            {
                ck_tile::FillMonotonicSeq<ADataType>{}(a_tensors[i]);
                ck_tile::FillMonotonicSeq<BDataType>{}(b_tensors[i]);
                ck_tile::FillConstant<AccDataType>{1.0f}(aq_tensors[i]);
                ck_tile::FillConstant<AccDataType>{1.0f}(bq_tensors[i]);
            }
            else if(settings_.init_method == 2)
            {
                ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_tensors[i]);
                ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_tensors[i]);
                ck_tile::FillConstant<AccDataType>{1.0f}(aq_tensors[i]);
                ck_tile::FillConstant<AccDataType>{1.0f}(bq_tensors[i]);
            }
            else
            {
                a_tensors[i].SetZero();
                b_tensors[i].SetZero();
                ck_tile::FillConstant<AccDataType>{1.0f}(aq_tensors[i]);
                ck_tile::FillConstant<AccDataType>{1.0f}(bq_tensors[i]);
            }

            a_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(
                a_tensors[i].get_element_space_size_in_bytes()));
            b_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(
                b_tensors[i].get_element_space_size_in_bytes()));
            c_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(
                c_dev_results[i].get_element_space_size_in_bytes()));
            aq_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(
                aq_tensors[i].get_element_space_size_in_bytes()));
            bq_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(
                bq_tensors[i].get_element_space_size_in_bytes()));

            a_dev_bufs[i]->ToDevice(a_tensors[i].data());
            b_dev_bufs[i]->ToDevice(b_tensors[i].data());
            c_dev_bufs[i]->SetZero();
            c_dev_results[i].SetZero();
            aq_dev_bufs[i]->ToDevice(aq_tensors[i].data());
            bq_dev_bufs[i]->ToDevice(bq_tensors[i].data());

            const void* p_a  = a_dev_bufs[i]->GetDeviceBuffer();
            const void* p_b  = b_dev_bufs[i]->GetDeviceBuffer();
            void* p_c        = c_dev_bufs[i]->GetDeviceBuffer();
            const void* p_aq = aq_dev_bufs[i]->GetDeviceBuffer();
            const void* p_bq = bq_dev_bufs[i]->GetDeviceBuffer();

            // RowColQuant: QK_A and QK_B are not used by the RowColQuant kernel code path
            // (the kernel ignores them and uses M/N directly for the per-row/col scale tensors).
            // stride_AQ=0 and stride_BQ=0 broadcast the 1-D scale vectors across all columns/rows.
            gemm_descs.push_back({p_a,
                                  p_b,
                                  p_c,
                                  p_aq,
                                  p_bq,
                                  problem.kbatch_,
                                  M,
                                  N,
                                  K,
                                  1, // QK_A: unused by RowColQuant
                                  1, // QK_B: unused by RowColQuant
                                  problem.stride_As_[i],
                                  problem.stride_Bs_[i],
                                  problem.stride_Cs_[i],
                                  0,   // stride_AQ (broadcast, 1-D vector)
                                  0}); // stride_BQ (broadcast, 1-D vector)
        }

        // Allocate workspace for kernel args using QuantGemmTransKernelArg
        ck_tile::DeviceMem workspace(gemm_descs.size() * sizeof(ck_tile::QuantGemmTransKernelArg));

        // Compute host reference for verification
        std::vector<ck_tile::HostTensor<CDataType>> c_host_results;
        if(settings_.verify)
        {
            c_host_results.reserve(group_count);
            for(int i = 0; i < group_count; ++i)
            {
                c_host_results.push_back(ck_tile::HostTensor<CDataType>(
                    ck_tile::host_tensor_descriptor(problem.Ms_[i],
                                                    problem.Ns_[i],
                                                    problem.stride_Cs_[i],
                                                    is_row_major(layout_c))));
            }
            gemm_host_reference_grouped(settings_.verify,
                                        problem,
                                        a_tensors,
                                        b_tensors,
                                        c_host_results,
                                        aq_tensors,
                                        bq_tensors);
        }

        for(auto& callable : callables)
        {
            auto kernel_run_result = callable(gemm_descs,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     settings_.log,
                                                                     settings_.n_warmup,
                                                                     settings_.n_repeat,
                                                                     settings_.is_gpu_timer,
                                                                     settings_.flush_cache,
                                                                     settings_.rotating_count},
                                              workspace.GetDeviceBuffer());
            process_result(problem, c_dev_bufs, c_host_results, c_dev_results, kernel_run_result);
        }
    }

    void process_result(const GroupedRowColQuantGemmProblem& problem,
                        std::vector<std::unique_ptr<ck_tile::DeviceMem>>& c_dev_bufs,
                        std::vector<ck_tile::HostTensor<CDataType>>& c_host_results,
                        std::vector<ck_tile::HostTensor<CDataType>>& c_dev_results,
                        const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;

        KernelInstance<GroupedRowColQuantGemmProblem> kernel_instance{
            name, problem, {-1.0f, -1.0f, -1.0f}};

        // Compute performance metrics (sum across all groups)
        std::size_t flop     = 0;
        std::size_t num_byte = 0;
        for(int i = 0; i < problem.group_count_; ++i)
        {
            flop += std::size_t(2) * problem.Ms_[i] * problem.Ns_[i] * problem.Ks_[i];
            num_byte += sizeof(ADataType) * problem.Ms_[i] * problem.Ks_[i] +
                        sizeof(BDataType) * problem.Ks_[i] * problem.Ns_[i] +
                        sizeof(CDataType) * problem.Ms_[i] * problem.Ns_[i] +
                        sizeof(AccDataType) * problem.Ms_[i] + // AQ scale
                        sizeof(AccDataType) * problem.Ns_[i];  // BQ scale
        }

        // update
        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(settings_.log > 0 && !settings_.json_output)
        {
            std::cout << kernel_instance << std::endl;
        }

        // Copy results back from device and verify per-group
        for(int i = 0; i < problem.group_count_; ++i)
        {
            c_dev_bufs[i]->FromDevice(c_dev_results[i].data());
        }

        bool verified_correct =
            !settings_.verify || compare_grouped(name, problem, c_dev_results, c_host_results);

        if(verified_correct)
        {
            kernel_instances_.emplace_back(kernel_instance);
        }
        else
        {
            std::cout << "Verification failed, skip kernel: " << name << std::endl;
        }

        // Clear device tensors
        for(int i = 0; i < problem.group_count_; ++i)
        {
            c_dev_bufs[i]->SetZero();
            c_dev_results[i].SetZero();
        }
    }

    KernelInstance<GroupedRowColQuantGemmProblem> select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto kernel_instance = *std::max_element(kernel_instances_.begin(),
                                                 kernel_instances_.end(),
                                                 [metric](const auto& a, const auto& b) {
                                                     return PerformanceResult::compare(
                                                         b.perf_result_, a.perf_result_, metric);
                                                 });

        if(settings_.json_output)
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

        if(!settings_.csv_filename.empty())
        {
            std::ofstream file(settings_.csv_filename + ".csv", std::ios::app);

            if(!file.is_open())
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
            else
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version,device_name," << "group_count,kbatch,"
                         << "dtype_a,dtype_b,dtype_acc,dtype_c," << "layout_a,layout_b,layout_c,"
                         << "quant_type," << "name,"
                         << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& problem = kernel_instance.problem_;
                const auto& name    = kernel_instance.name_;
                const auto& perf    = kernel_instance.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << ","
                     << problem.group_count_ << "," << problem.kbatch_ << "," << problem.dtype_a_
                     << "," << problem.dtype_b_ << "," << problem.dtype_acc_ << ","
                     << problem.dtype_c_ << "," << problem.layout_a_ << "," << problem.layout_b_
                     << "," << problem.layout_c_ << "," << "RowColQuant," << name << ","
                     << std::fixed << std::setprecision(4) << perf.latency_ << "," << std::fixed
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

    GroupedRowColQuantGemmProfiler(const GroupedRowColQuantGemmProfiler&)            = delete;
    GroupedRowColQuantGemmProfiler& operator=(const GroupedRowColQuantGemmProfiler&) = delete;

    private:
    ~GroupedRowColQuantGemmProfiler() { kernel_instances_.clear(); }
    GroupedRowColQuantGemmProfiler(Settings settings) : settings_(settings) {}

    Settings settings_;

    std::vector<KernelInstance<GroupedRowColQuantGemmProblem>> kernel_instances_;
};
