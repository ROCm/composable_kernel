// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "gemm_abquant_common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

// Data types and Layouts are defined by the generated kernel headers:
//   ADataType, BDataType, AQDataType, BQDataType, AccDataType, CDataType
//   ALayout, BLayout, CLayout, AQLayout, BQLayout

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

struct ABQuantGemmProblem
{
    int split_k_;
    int m_, n_, k_;
    int stride_a_, stride_b_, stride_c_;
    int stride_aq_;
    int stride_bq_;
    int group_size_k_;
    int group_size_n_;

    std::string dtype_a_, dtype_b_, dtype_aq_, dtype_bq_, dtype_acc_, dtype_c_;
    std::string layout_a_, layout_b_, layout_c_;

    friend std::ostream& operator<<(std::ostream& os, const ABQuantGemmProblem& problem)
    {
        os << "{\n"
           << "   \"split_k\":" << problem.split_k_ << ",\n"
           << "   \"m\":" << problem.m_ << ",\n"
           << "   \"n\":" << problem.n_ << ",\n"
           << "   \"k\":" << problem.k_ << ",\n"
           << "   \"stride_a\":" << problem.stride_a_ << ",\n"
           << "   \"stride_b\":" << problem.stride_b_ << ",\n"
           << "   \"stride_c\":" << problem.stride_c_ << ",\n"
           << "   \"stride_aq\":" << problem.stride_aq_ << ",\n"
           << "   \"stride_bq\":" << problem.stride_bq_ << ",\n"
           << "   \"group_size_k\":" << problem.group_size_k_ << ",\n"
           << "   \"group_size_n\":" << problem.group_size_n_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_aq\":\"" << problem.dtype_aq_ << "\",\n"
           << "   \"dtype_bq\":\"" << problem.dtype_bq_ << "\",\n"
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
    ABQuantGemmProblem problem_;
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

template <typename ADataType_,
          typename BDataType_,
          typename AQDataType_,
          typename BQDataType_,
          typename AccDataType_,
          typename CDataType_>
auto calculate_rtol_atol_abquant(const ck_tile::index_t K,
                                 const ck_tile::index_t kbatch,
                                 const float max_accumulated_value)
{
    // Both A and B are the same FP8/BF8 type for abquant; assert so mixed-precision additions are
    // caught.
    static_assert(sizeof(ADataType_) == sizeof(BDataType_),
                  "calculate_rtol_atol_abquant assumes equal-width A and B types");
    using ComputeType = ADataType_;
    const auto rtol   = ck_tile::get_relative_threshold<ComputeType, CDataType_, AccDataType_>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType_, AccDataType_>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType_, CDataType_, CDataType_>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType_, CDataType_, CDataType_>(
        max_accumulated_value, kbatch);
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

/// @brief Compare device and host results for ABQuant GEMM
bool compare_abquant(std::string instanceName,
                     ck_tile::index_t K,
                     ck_tile::index_t kbatch,
                     const ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                     const ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    const float max_accumulated_value =
        std::abs(static_cast<float>(*std::max_element(c_m_n_host_result.mData.begin(),
                                                      c_m_n_host_result.mData.end(),
                                                      [](const auto& a, const auto& b) {
                                                          return std::abs(static_cast<float>(a)) <
                                                                 std::abs(static_cast<float>(b));
                                                      })));
    const auto rtol_atol = calculate_rtol_atol_abquant<ADataType,
                                                       BDataType,
                                                       AQDataType,
                                                       BQDataType,
                                                       AccDataType,
                                                       CDataType>(K, kbatch, max_accumulated_value);
    bool pass            = ck_tile::check_err(c_m_n_dev_result,
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

/// @brief CPU reference implementation for ABQuant GEMM
void abquant_gemm_host_reference(int verify,
                                 ck_tile::HostTensor<ADataType>& a_m_k,
                                 ck_tile::HostTensor<AQDataType>& aq_m_qk,
                                 ck_tile::HostTensor<BDataType>& b_k_n,
                                 ck_tile::HostTensor<BQDataType>& bq_qk_n,
                                 ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    if(verify == 1)
    {
        c_m_n_host_result.SetZero();
        ck_tile::reference_gemm_abquant<ADataType,
                                        AQDataType,
                                        BDataType,
                                        BQDataType,
                                        AccDataType,
                                        CDataType,
                                        SelectedKernel::AQuantGroupSize,
                                        SelectedKernel::BQuantGroupSize>(
            a_m_k, aq_m_qk, b_k_n, bq_qk_n, c_m_n_host_result);
    }
}
#pragma clang diagnostic pop
