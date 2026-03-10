// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm_common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
// Data types and Layouts are defined by the generated kernel headers
// No hardcoded type definitions here to avoid conflicts

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

struct GroupedGemmProblem
{
    int group_count_;
    int kbatch_;
    std::vector<int> Ms_, Ns_, Ks_;
    std::vector<int> stride_As_, stride_Bs_, stride_Cs_;

    std::string dtype_a_, dtype_b_, dtype_acc_, dtype_c_;
    std::string layout_a_, layout_b_, layout_c_;

    friend std::ostream& operator<<(std::ostream& os, const GroupedGemmProblem& problem)
    {
        os << "{\n"
           << "   \"group_count\":" << problem.group_count_ << ",\n"
           << "   \"kbatch\":" << problem.kbatch_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_c\":\"" << problem.dtype_c_ << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "   \"layout_c\":\"" << problem.layout_c_ << "\"\n"
           << "}";
        return os;
    }
};

struct PerformanceResult
{
    double latency_;
    double tflops_;
    double bandwidth_;

    static bool compare(const PerformanceResult& a, const PerformanceResult& b, Metric m)
    {
        switch(m)
        {
        case Metric::LATENCY: return a.latency_ < b.latency_;
        case Metric::TFLOPS: return a.tflops_ > b.tflops_;
        case Metric::BANDWIDTH: return a.bandwidth_ > b.bandwidth_;
        default: throw std::invalid_argument("Unsupported metric type");
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const PerformanceResult& result)
    {
        os << "{\n"
           << "   \"latency(ms)\": " << std::fixed << std::setprecision(2) << result.latency_
           << ",\n"
           << "   \"tflops(TFlops)\": " << result.tflops_ << ",\n"
           << "   \"bandwidth(GB/s)\": " << result.bandwidth_ << "\n"
           << "}";
        return os;
    }
};

struct KernelInstance
{
    std::string name_;
    GroupedGemmProblem problem_;
    PerformanceResult perf_result_;

    static bool compare(const KernelInstance& a, const KernelInstance& b, Metric m)
    {
        return PerformanceResult::compare(a.perf_result_, b.perf_result_, m);
    }

    friend std::ostream& operator<<(std::ostream& os, const KernelInstance& obj)
    {
        os << "{\n"
           << " \"name\": \"" << obj.name_ << "\",\n"
           << " \"problem\": " << obj.problem_ << ",\n"
           << " \"perf_result\": " << obj.perf_result_ << "\n"
           << "}";
        return os;
    }
};

struct Setting
{
    int n_warmup_;
    int n_repeat_;
    bool is_gpu_timer_;
    int verify_;
    int init_method_;
    bool log_;
    std::string csv_filename_;
    bool flush_cache_;
    int rotating_count_;
    bool json_output_;
};

inline std::string get_rocm_version()
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

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

/// @brief Function to compare the results of the device and host computations for a single group
bool compare_single(std::string instanceName,
                    ck_tile::index_t K,
                    ck_tile::index_t kbatch,
                    ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                    ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    const float max_accumulated_value =
        *std::max_element(c_m_n_host_result.mData.begin(), c_m_n_host_result.mData.end());
    const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
        K, kbatch, max_accumulated_value);
    bool pass = ck_tile::check_err(c_m_n_dev_result,
                                   c_m_n_host_result,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "For " << instanceName << " Relative error threshold is "
              << rtol_atol.at(ck_tile::number<0>{}) << " Absolute error threshold is "
              << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The verification result is:" << (pass ? "correct" : "fail") << std::endl;

    return pass;
}

/// @brief Function to compare grouped gemm results across all groups
bool compare_grouped(std::string instanceName,
                     const GroupedGemmProblem& problem,
                     std::vector<ck_tile::HostTensor<CDataType>>& c_dev_results,
                     std::vector<ck_tile::HostTensor<CDataType>>& c_host_results)
{
    bool pass = true;
    for(int i = 0; i < problem.group_count_; ++i)
    {
        pass &= compare_single(instanceName + "[" + std::to_string(i) + "]",
                               problem.Ks_[i],
                               problem.kbatch_,
                               c_dev_results[i],
                               c_host_results[i]);
    }
    return pass;
}

/// @brief Function to get the kernel output with reference implementation on CPU/GPU for all groups
void gemm_host_reference_grouped(int verify,
                                 const GroupedGemmProblem& problem,
                                 std::vector<ck_tile::HostTensor<ADataType>>& a_tensors,
                                 std::vector<ck_tile::HostTensor<BDataType>>& b_tensors,
                                 std::vector<ck_tile::HostTensor<CDataType>>& c_host_results,
                                 std::vector<std::unique_ptr<ck_tile::DeviceMem>>& a_dev_bufs,
                                 std::vector<std::unique_ptr<ck_tile::DeviceMem>>& b_dev_bufs)
{
    const int group_count = problem.group_count_;

    if(verify == 1)
    {
        for(int i = 0; i < group_count; ++i)
        {
            c_host_results[i].SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_tensors[i], b_tensors[i], c_host_results[i]);
        }
    }
    else if(verify == 2)
    {
        for(int i = 0; i < group_count; ++i)
        {
            if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
            {
                b_dev_bufs[i]->ToDevice(b_tensors[i].data());
            }

            ck_tile::DeviceMem c_gpu_buf_ref(c_host_results[i].get_element_space_size_in_bytes());
            c_host_results[i].SetZero();
            c_gpu_buf_ref.SetZero();

            ADataType* d_A = static_cast<ADataType*>(a_dev_bufs[i]->GetDeviceBuffer());
            BDataType* d_B = static_cast<BDataType*>(b_dev_bufs[i]->GetDeviceBuffer());
            CDataType* d_C = static_cast<CDataType*>(c_gpu_buf_ref.GetDeviceBuffer());

            ck_tile::reference_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout>(d_A,
                                                 d_B,
                                                 d_C,
                                                 problem.Ms_[i],
                                                 problem.Ns_[i],
                                                 problem.Ks_[i],
                                                 problem.stride_As_[i],
                                                 problem.stride_Bs_[i],
                                                 problem.stride_Cs_[i]);

            c_gpu_buf_ref.FromDevice(c_host_results[i].data());
        }
    }
}
#pragma clang diagnostic pop
