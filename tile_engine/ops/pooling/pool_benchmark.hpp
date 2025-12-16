// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "pool_common.hpp"

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

struct PoolProblem
{
    std::string inDType;
    std::string outDType;
    std::string computeDType;
    std::string indexDType;
    std::string blockShape;
    std::string reduceOp;

    int poolDim;
    int N, D, H, W, C;

    int windowZ, windowY, windowX;
    int strideZ, strideY, strideX;
    int dilationZ, dilationY, dilationX;
    int leftPadZ, leftPadY, leftPadX;
    int rightPadZ, rightPadY, rightPadX;

    bool outputIndex;
    bool propagateNan;

    friend std::ostream& operator<<(std::ostream& os, const PoolProblem& problem)
    {
        os << "{\n"
           << "    \"inDType\": \"" << problem.inDType << "\",\n"
           << "    \"outDType\": \"" << problem.outDType << "\",\n"
           << "    \"computeDType\": \"" << problem.computeDType << "\",\n"
           << "    \"indexDType\": \"" << problem.indexDType << "\",\n"
           << "    \"blockShape\": \"" << problem.blockShape << "\",\n"
           << "    \"reduceOp\": \"" << problem.reduceOp << "\",\n"
           << "    \"poolDim\": " << problem.poolDim << ",\n"
           << "    \"N\": " << problem.N << ",\n"
           << "    \"D\": " << problem.D << ",\n"
           << "    \"H\": " << problem.H << ",\n"
           << "    \"W\": " << problem.W << ",\n"
           << "    \"C\": " << problem.C << ",\n"
           << "    \"windowZ\": " << problem.windowZ << ",\n"
           << "    \"windowY\": " << problem.windowY << ",\n"
           << "    \"windowX\": " << problem.windowX << ",\n"
           << "    \"strideZ\": " << problem.strideZ << ",\n"
           << "    \"strideY\": " << problem.strideY << ",\n"
           << "    \"strideX\": " << problem.strideX << ",\n"
           << "    \"dilationZ\": " << problem.dilationZ << ",\n"
           << "    \"dilationY\": " << problem.dilationY << ",\n"
           << "    \"dilationX\": " << problem.dilationX << ",\n"
           << "    \"leftPadZ\": " << problem.leftPadZ << ",\n"
           << "    \"leftPadY\": " << problem.leftPadY << ",\n"
           << "    \"leftPadX\": " << problem.leftPadX << ",\n"
           << "    \"rightPadZ\": " << problem.rightPadZ << ",\n"
           << "    \"rightPadY\": " << problem.rightPadY << ",\n"
           << "    \"rightPadX\": " << problem.rightPadX << ",\n"
           << "    \"outputIndex\": " << (problem.outputIndex ? "true" : "false") << ",\n"
           << "    \"propagateNan\": " << (problem.propagateNan ? "true" : "false") << "\n"
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
    PoolProblem problem_;
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

/// @brief Function to compare the results of the device and host computations
template <typename OutDataType>
bool compare_pool_results(std::string instanceName,
                          ck_tile::HostTensor<OutDataType>& out_dev_result,
                          ck_tile::HostTensor<OutDataType>& out_host_result)
{
    bool pass = ck_tile::check_err(out_dev_result, out_host_result, "Error: Incorrect results!");

    std::cout << "For " << instanceName
              << " verification result is: " << (pass ? "correct" : "fail") << std::endl;

    return pass;
}

template <typename IndexDataType>
bool compare_pool_index_results(std::string instanceName,
                                ck_tile::HostTensor<IndexDataType>& out_index_dev_result,
                                ck_tile::HostTensor<IndexDataType>& out_index_host_result)
{
    bool pass = ck_tile::check_err(
        out_index_dev_result, out_index_host_result, "Error: Incorrect index results!");

    std::cout << "For " << instanceName
              << " index verification result is: " << (pass ? "correct" : "fail") << std::endl;

    return pass;
}
