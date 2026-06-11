// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <functional>
#include <tuple>
#include <string>
#include <vector>
#include <array>
#include <cstddef>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gemm/gemm_profiler.hpp"
#include "common/utils.hpp"
#include "gemm_multi_abd_benchmark.hpp"

class GemmMultiABDProfiler
    : public GemmProfiler<GemmMultiABDProfiler,
                          GemmMultiABDProblem,
                          ck_tile::GemmMultiABDHostArgs<NumATensors, NumBTensors, NumDTensors>>
{
    public:
    using BaseGemm =
        GemmProfiler<GemmMultiABDProfiler,
                     GemmMultiABDProblem,
                     ck_tile::GemmMultiABDHostArgs<NumATensors, NumBTensors, NumDTensors>>;
    using BaseGemm::benchmark;

    GemmMultiABDProfiler(Settings setting) : BaseGemm(setting) {}

    void benchmark(GemmMultiABDProblem& problem,
                   std::vector<std::function<std::tuple<std::string, float>(
                       ck_tile::GemmMultiABDHostArgs<NumATensors, NumBTensors, NumDTensors>&,
                       const ck_tile::stream_config&)>>& callables) override
    {
        const ALayout a_layout{};
        const BLayout b_layout{};
        const DLayout d_layout{};
        const ELayout e_layout{};

        // Compute default strides
        for(std::size_t i = 0; i < NumATensors; i++)
            problem.stride_as_[i] = ck_tile::get_default_stride(
                problem.m_, problem.k_, problem.stride_as_[i], is_row_major(a_layout));
        for(std::size_t i = 0; i < NumBTensors; i++)
            problem.stride_bs_[i] = ck_tile::get_default_stride(
                problem.k_, problem.n_, problem.stride_bs_[i], is_row_major(b_layout));
        for(std::size_t i = 0; i < NumDTensors; i++)
            problem.stride_ds_[i] = ck_tile::get_default_stride(
                problem.m_, problem.n_, problem.stride_ds_[i], is_row_major(d_layout));
        problem.stride_e_ = ck_tile::get_default_stride(
            problem.m_, problem.n_, problem.stride_e_, is_row_major(e_layout));

        const auto M = problem.m_;
        const auto N = problem.n_;
        const auto K = problem.k_;

        // Create host tensors
        auto a_tensors =
            make_host_tensor_array<ADataType, ALayout, NumATensors>(M, K, problem.stride_as_);
        auto b_tensors =
            make_host_tensor_array<BDataType, BLayout, NumBTensors>(K, N, problem.stride_bs_);
        auto d_tensors =
            make_host_tensor_array<DBaseDataType, DLayout, NumDTensors>(M, N, problem.stride_ds_);

        ck_tile::HostTensor<EDataType> e_dev_result(
            ck_tile::host_tensor_descriptor(M, N, problem.stride_e_, is_row_major(e_layout)));

        // Fill with data
        for(auto& t : a_tensors)
            ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(t);
        for(auto& t : b_tensors)
            ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(t);
        for(auto& t : d_tensors)
            ck_tile::FillUniformDistribution<DBaseDataType>{-1.f, 1.f}(t);

        // Create device buffers and copy
        std::vector<ck_tile::DeviceMem> a_dev_bufs(NumATensors), b_dev_bufs(NumBTensors),
            d_dev_bufs(NumDTensors);
        for(std::size_t i = 0; i < NumATensors; i++)
        {
            a_dev_bufs[i].Realloc(a_tensors[i].get_element_space_size_in_bytes());
            a_dev_bufs[i].ToDevice(a_tensors[i].mData.data());
        }
        for(std::size_t i = 0; i < NumBTensors; i++)
        {
            b_dev_bufs[i].Realloc(b_tensors[i].get_element_space_size_in_bytes());
            b_dev_bufs[i].ToDevice(b_tensors[i].mData.data());
        }
        for(std::size_t i = 0; i < NumDTensors; i++)
        {
            d_dev_bufs[i].Realloc(d_tensors[i].get_element_space_size_in_bytes());
            d_dev_bufs[i].ToDevice(d_tensors[i].mData.data());
        }

        ck_tile::DeviceMem e_device_buf(e_dev_result.get_element_space_size_in_bytes());
        e_device_buf.SetZero();
        e_dev_result.SetZero();

        // Build pointer and stride arrays for kernel args
        std::array<const void*, NumATensors> as_ptr;
        std::array<const void*, NumBTensors> bs_ptr;
        std::array<const void*, NumDTensors> ds_ptr;
        std::array<ck_tile::index_t, NumATensors> stride_as;
        std::array<ck_tile::index_t, NumBTensors> stride_bs;
        std::array<ck_tile::index_t, NumDTensors> stride_ds;

        for(std::size_t i = 0; i < NumATensors; i++)
        {
            as_ptr[i]    = a_dev_bufs[i].GetDeviceBuffer();
            stride_as[i] = problem.stride_as_[i];
        }
        for(std::size_t i = 0; i < NumBTensors; i++)
        {
            bs_ptr[i]    = b_dev_bufs[i].GetDeviceBuffer();
            stride_bs[i] = problem.stride_bs_[i];
        }
        for(std::size_t i = 0; i < NumDTensors; i++)
        {
            ds_ptr[i]    = d_dev_bufs[i].GetDeviceBuffer();
            stride_ds[i] = problem.stride_ds_[i];
        }

        // Build host args
        ck_tile::GemmMultiABDHostArgs<NumATensors, NumBTensors, NumDTensors> args{
            as_ptr,
            bs_ptr,
            ds_ptr,
            e_device_buf.GetDeviceBuffer(),
            problem.split_k_,
            M,
            N,
            K,
            stride_as,
            stride_bs,
            stride_ds,
            problem.stride_e_};

        // Host reference computation
        ck_tile::HostTensor<EDataType> e_host_result(
            ck_tile::host_tensor_descriptor(M, N, problem.stride_e_, is_row_major(e_layout)));

        if(setting_.verify)
        {
            gemm_multi_abd_host_reference<NumATensors, NumBTensors, NumDTensors>(
                setting_.verify, a_tensors, b_tensors, d_tensors, e_host_result);
        }

        for(auto& callable : callables)
        {
            auto kernel_run_result = callable(args,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     setting_.log,
                                                                     setting_.n_warmup,
                                                                     setting_.n_repeat,
                                                                     setting_.is_gpu_timer,
                                                                     setting_.flush_cache,
                                                                     setting_.rotating_count});

            process_result(problem, e_device_buf, e_host_result, e_dev_result, kernel_run_result);
        }
    }

    // Override process_result: multi-abd has NumA A-tensors and NumB B-tensors contributing
    // to bandwidth, and uses EDataType (aliased as CDataType) for the output buffer.
    void process_result(const GemmMultiABDProblem& problem,
                        ck_tile::DeviceMem& e_device_buf,
                        ck_tile::HostTensor<EDataType>& e_host_result,
                        ck_tile::HostTensor<EDataType>& e_dev_result,
                        const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;

        KernelInstance<GemmMultiABDProblem> kernel_instance{name, problem, {-1.0, -1.0, -1.0}};

        // Compute performance metrics
        std::size_t flop     = std::size_t(2) * problem.m_ * problem.n_ * problem.k_;
        std::size_t num_byte = 0;

        // A tensor bytes
        ck_tile::static_for<0, NumATensors, 1>{}([&](auto i) {
            using AType = ck_tile::remove_cvref_t<std::tuple_element_t<i, AsDataType>>;
            num_byte += sizeof(AType) * problem.m_ * problem.k_;
        });
        // B tensor bytes
        ck_tile::static_for<0, NumBTensors, 1>{}([&](auto i) {
            using BType = ck_tile::remove_cvref_t<std::tuple_element_t<i, BsDataType>>;
            num_byte += sizeof(BType) * problem.k_ * problem.n_;
        });
        // D tensor bytes + one elementwise flop per element
        ck_tile::static_for<0, NumDTensors, 1>{}([&](auto i) {
            using DType = ck_tile::remove_cvref_t<std::tuple_element_t<i, DsDataType>>;
            num_byte += sizeof(DType) * problem.m_ * problem.n_;
            flop += problem.m_ * problem.n_;
        });
        // E tensor bytes
        num_byte += sizeof(EDataType) * problem.m_ * problem.n_;

        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<double>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log > 0 && !setting_.json_output)
        {
            std::cout << kernel_instance << std::endl;
        }

        // Verify result
        e_device_buf.FromDevice(e_dev_result.data());
        bool verified_correct =
            !setting_.verify ||
            compare<ADataType, BDataType, AccDataType, EDataType, DBaseDataType>(
                name,
                problem.k_,
                1, // Multi ABD currently supports only k_batch = 1
                e_dev_result,
                e_host_result);

        if(verified_correct)
        {
            kernel_instances_.push_back(std::move(kernel_instance));
        }
        else
        {
            std::cout << "Verification failed, skip kernel: " << name << std::endl;
        }

        // Clear tensor for next run
        e_device_buf.SetZero();
        e_dev_result.SetZero();
    }

    // Shadow base select_best_instance to write the multi-abd CSV schema
    // (variable-count tensor columns instead of the base's fixed stride_a/b/c columns).
    KernelInstance<GemmMultiABDProblem> select_best_instance(Metric metric)
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
                    file << "rocm_version,device_name,split_k,m,n,k,";
                    for(std::size_t i = 0; i < NumATensors; i++)
                        file << "stride_a" << i << ",";
                    for(std::size_t i = 0; i < NumBTensors; i++)
                        file << "stride_b" << i << ",";
                    for(std::size_t i = 0; i < NumDTensors; i++)
                        file << "stride_d" << i << ",";
                    file << "stride_e,";
                    for(std::size_t i = 0; i < NumATensors; i++)
                        file << "dtype_a" << i << ",";
                    for(std::size_t i = 0; i < NumBTensors; i++)
                        file << "dtype_b" << i << ",";
                    for(std::size_t i = 0; i < NumDTensors; i++)
                        file << "dtype_d" << i << ",";
                    file << "dtype_acc,dtype_e,";
                    for(std::size_t i = 0; i < NumATensors; i++)
                        file << "layout_a" << i << ",";
                    for(std::size_t i = 0; i < NumBTensors; i++)
                        file << "layout_b" << i << ",";
                    for(std::size_t i = 0; i < NumDTensors; i++)
                        file << "layout_d" << i << ",";
                    file << "layout_e,a_elementwise,b_elementwise,cde_elementwise,"
                         << "name,latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& prob = kernel_instance.problem_;
                const auto& name = kernel_instance.name_;
                const auto& perf = kernel_instance.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << ","
                     << prob.split_k_ << "," << prob.m_ << "," << prob.n_ << "," << prob.k_ << ",";
                for(std::size_t i = 0; i < prob.stride_as_.size(); i++)
                    file << prob.stride_as_[i] << ",";
                for(std::size_t i = 0; i < prob.stride_bs_.size(); i++)
                    file << prob.stride_bs_[i] << ",";
                for(std::size_t i = 0; i < prob.stride_ds_.size(); i++)
                    file << prob.stride_ds_[i] << ",";
                file << prob.stride_e_ << ",";
                for(std::size_t i = 0; i < prob.dtype_as_.size(); i++)
                    file << prob.dtype_as_[i] << ",";
                for(std::size_t i = 0; i < prob.dtype_bs_.size(); i++)
                    file << prob.dtype_bs_[i] << ",";
                for(std::size_t i = 0; i < prob.dtype_ds_.size(); i++)
                    file << prob.dtype_ds_[i] << ",";
                file << prob.dtype_acc_ << "," << prob.dtype_e_ << ",";
                for(std::size_t i = 0; i < prob.layout_as_.size(); i++)
                    file << prob.layout_as_[i] << ",";
                for(std::size_t i = 0; i < prob.layout_bs_.size(); i++)
                    file << prob.layout_bs_[i] << ",";
                for(std::size_t i = 0; i < prob.layout_ds_.size(); i++)
                    file << prob.layout_ds_[i] << ",";
                file << prob.layout_e_ << "," << prob.a_elementwise_ << "," << prob.b_elementwise_
                     << "," << prob.cde_elementwise_ << "," << name << "," << std::fixed
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
};
