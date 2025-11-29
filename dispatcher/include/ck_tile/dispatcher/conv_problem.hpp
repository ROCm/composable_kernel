// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file conv_problem.hpp
 * @brief Convolution problem definition
 */

#pragma once

#include <cstdint>
#include <array>
#include <string>

namespace ck_tile {
namespace dispatcher {

/**
 * @brief Convolution operation type
 */
enum class ConvOp
{
    Forward,       // Y = Conv(X, W)
    BackwardData,  // dX = ConvBwdData(dY, W)
    BackwardWeight // dW = ConvBwdWeight(X, dY)
};

/**
 * @brief Convolution problem specification
 */
struct ConvProblem
{
    // Batch and channels
    std::int64_t N; // Batch size
    std::int64_t C; // Input channels
    std::int64_t K; // Output channels (filters)
    std::int64_t G; // Number of groups (1 for standard conv)

    // Spatial dimensions (supports 1D, 2D, 3D)
    std::array<std::int64_t, 3> input_spatial;  // {D, H, W} or {H, W, 1} for 2D
    std::array<std::int64_t, 3> filter_spatial; // {Z, Y, X} or {R, S, 1} for 2D
    std::array<std::int64_t, 3> output_spatial; // {Do, Ho, Wo}

    // Convolution parameters
    std::array<std::int64_t, 3> stride;   // Stride in each dimension
    std::array<std::int64_t, 3> padding;  // Padding in each dimension
    std::array<std::int64_t, 3> dilation; // Dilation in each dimension

    // Operation type
    ConvOp op = ConvOp::Forward;

    // Default constructor for 2D convolution
    ConvProblem()
        : N(1),
          C(64),
          K(64),
          G(1),
          input_spatial{1, 28, 28},
          filter_spatial{1, 3, 3},
          output_spatial{1, 26, 26},
          stride{1, 1, 1},
          padding{0, 0, 0},
          dilation{1, 1, 1},
          op(ConvOp::Forward)
    {
    }

    // Constructor for 2D convolution
    ConvProblem(std::int64_t n,
                std::int64_t c,
                std::int64_t k,
                std::int64_t hi,
                std::int64_t wi,
                std::int64_t y,
                std::int64_t x,
                std::int64_t stride_h   = 1,
                std::int64_t stride_w   = 1,
                std::int64_t pad_h      = 0,
                std::int64_t pad_w      = 0,
                std::int64_t dilation_h = 1,
                std::int64_t dilation_w = 1)
        : N(n),
          C(c),
          K(k),
          G(1),
          input_spatial{1, hi, wi},
          filter_spatial{1, y, x},
          stride{1, stride_h, stride_w},
          padding{0, pad_h, pad_w},
          dilation{1, dilation_h, dilation_w},
          op(ConvOp::Forward)
    {
        compute_output_size();
    }

    /// Compute output spatial dimensions
    void compute_output_size()
    {
        for(int i = 0; i < 3; ++i)
        {
            std::int64_t effective_filter = (filter_spatial[i] - 1) * dilation[i] + 1;
            output_spatial[i] =
                (input_spatial[i] + 2 * padding[i] - effective_filter) / stride[i] + 1;
        }
    }

    /// Get 2D height/width accessors
    std::int64_t Hi() const { return input_spatial[1]; }
    std::int64_t Wi() const { return input_spatial[2]; }
    std::int64_t Ho() const { return output_spatial[1]; }
    std::int64_t Wo() const { return output_spatial[2]; }
    std::int64_t Y() const { return filter_spatial[1]; } // Filter height
    std::int64_t X() const { return filter_spatial[2]; } // Filter width

    /// Get total FLOPs for this convolution
    double get_flops() const
    {
        // Forward: 2 * N * K * Ho * Wo * C * Y * X / G
        double spatial_out = 1.0;
        double filter_size = 1.0;
        for(int i = 0; i < 3; ++i)
        {
            spatial_out *= output_spatial[i];
            filter_size *= filter_spatial[i];
        }
        return 2.0 * N * K * spatial_out * (C / G) * filter_size;
    }

    /// Check if this is a depthwise convolution
    bool is_depthwise() const { return G == C && G == K; }

    /// Check if this is a pointwise (1x1) convolution
    bool is_pointwise() const
    {
        return filter_spatial[0] == 1 && filter_spatial[1] == 1 && filter_spatial[2] == 1;
    }

    /// String representation
    std::string to_string() const
    {
        std::string s = "ConvProblem(N=" + std::to_string(N);
        s += ", C=" + std::to_string(C) + ", K=" + std::to_string(K);
        s += ", Hi=" + std::to_string(Hi()) + ", Wi=" + std::to_string(Wi());
        s += ", Y=" + std::to_string(Y()) + ", X=" + std::to_string(X());
        s += ", Ho=" + std::to_string(Ho()) + ", Wo=" + std::to_string(Wo());
        s += ")";
        return s;
    }
};

} // namespace dispatcher
} // namespace ck_tile
