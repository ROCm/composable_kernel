// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"

namespace ck_tile {

/// @brief Performance metrics for benchmarking
enum class PoolMetric
{
    LATENCY,
    BANDWIDTH
};

/// @brief Pooling problem specification for 2D pooling
struct PoolProblem2D
{
    index_t N, H, W, C;              // Input dimensions (NHWC)
    index_t Y, X;                    // Window dimensions
    index_t stride_h, stride_w;      // Window strides
    index_t dilation_h, dilation_w;  // Window dilations
    index_t pad_h_left, pad_h_right; // Height padding
    index_t pad_w_left, pad_w_right; // Width padding
    std::string datatype;            // Data type name
    std::string reduce_op;           // "max", "min", or "avg"

    index_t Ho() const
    {
        index_t Ys = (Y - 1) * dilation_h + 1;
        return (H + pad_h_left + pad_h_right - Ys) / stride_h + 1;
    }

    index_t Wo() const
    {
        index_t Xs = (X - 1) * dilation_w + 1;
        return (W + pad_w_left + pad_w_right - Xs) / stride_w + 1;
    }

    index_t input_elements() const { return N * H * W * C; }
    index_t output_elements() const { return N * Ho() * Wo() * C; }

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "N" << N << "_H" << H << "_W" << W << "_C" << C << "_Y" << Y << "_X" << X << "_Sh"
            << stride_h << "_Sw" << stride_w << "_Dh" << dilation_h << "_Dw" << dilation_w;
        if(pad_h_left > 0 || pad_w_left > 0)
            oss << "_Ph" << pad_h_left << "_Pw" << pad_w_left;
        return oss.str();
    }
};

/// @brief Pooling problem specification for 3D pooling
struct PoolProblem3D
{
    index_t N, D, H, W, C;                      // Input dimensions (NDHWC)
    index_t Z, Y, X;                            // Window dimensions
    index_t stride_d, stride_h, stride_w;       // Window strides
    index_t dilation_d, dilation_h, dilation_w; // Window dilations
    index_t pad_d_left, pad_d_right;            // Depth padding
    index_t pad_h_left, pad_h_right;            // Height padding
    index_t pad_w_left, pad_w_right;            // Width padding
    std::string datatype;                       // Data type name
    std::string reduce_op;                      // "max", "min", or "avg"

    index_t Do() const
    {
        index_t Zs = (Z - 1) * dilation_d + 1;
        return (D + pad_d_left + pad_d_right - Zs) / stride_d + 1;
    }

    index_t Ho() const
    {
        index_t Ys = (Y - 1) * dilation_h + 1;
        return (H + pad_h_left + pad_h_right - Ys) / stride_h + 1;
    }

    index_t Wo() const
    {
        index_t Xs = (X - 1) * dilation_w + 1;
        return (W + pad_w_left + pad_w_right - Xs) / stride_w + 1;
    }

    index_t input_elements() const { return N * D * H * W * C; }
    index_t output_elements() const { return N * Do() * Ho() * Wo() * C; }

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "N" << N << "_D" << D << "_H" << H << "_W" << W << "_C" << C << "_Z" << Z << "_Y"
            << Y << "_X" << X;
        return oss.str();
    }
};

/// @brief Performance result for a pooling kernel
struct PoolPerformanceResult
{
    float latency_ms;
    float bandwidth_gb_s;

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "latency=" << latency_ms << "ms, bandwidth=" << bandwidth_gb_s << "GB/s";
        return oss.str();
    }
};

/// @brief Benchmark settings
struct PoolBenchmarkSetting
{
    int warmup      = 5;
    int repeat      = 20;
    bool verify     = true;
    int init_method = 0; // 0: uniform random, 1: integer sequence, 2: constant, 3: special
};

} // namespace ck_tile
