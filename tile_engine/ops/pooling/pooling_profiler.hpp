// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "tile_engine/ops/pooling/pooling_benchmark.hpp"

namespace ck_tile {

/// @brief Profiler for pooling kernels.
///
/// Handles tensor setup, kernel launch, reference computation, and verification
/// for 2D pooling benchmarks.
template <typename InDataType,
          typename OutDataType,
          typename ComputeDataType,
          typename IndexDataType>
class PoolProfiler2D
{
    public:
    PoolProfiler2D(const PoolBenchmarkSetting& setting) : setting_(setting) {}

    /// @brief Benchmark a 2D pooling kernel
    /// @param problem The pooling problem specification
    /// @param kernel_func Function that launches the kernel and returns latency
    template <typename KernelFunc>
    PoolPerformanceResult benchmark(const PoolProblem2D& problem, KernelFunc kernel_func)
    {
        const index_t Ho = problem.Ho();
        const index_t Wo = problem.Wo();

        // Create host tensors
        HostTensor<InDataType> h_in({problem.N, problem.H, problem.W, problem.C});
        HostTensor<OutDataType> h_out({problem.N, Ho, Wo, problem.C});
        HostTensor<OutDataType> h_out_ref({problem.N, Ho, Wo, problem.C});
        HostTensor<IndexDataType> h_out_index({problem.N, Ho, Wo, problem.C});
        HostTensor<IndexDataType> h_out_ref_index({problem.N, Ho, Wo, problem.C});

        // Initialize
        FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);
        h_out.SetZero();
        h_out_ref.SetZero();

        // Device memory
        DeviceMem d_in(h_in.get_element_space_size_in_bytes());
        DeviceMem d_out(h_out.get_element_space_size_in_bytes());
        DeviceMem d_out_index(h_out_index.get_element_space_size_in_bytes());

        d_in.ToDevice(h_in.data());
        d_out.SetZero();
        d_out_index.SetZero();

        // Build kernel args
        const auto input_shape  = make_tuple(problem.N, problem.H, problem.W, problem.C);
        const auto output_shape = make_tuple(problem.N, Ho, Wo, problem.C);
        const auto input_strides =
            make_tuple(problem.H * problem.W * problem.C, problem.W * problem.C, problem.C, 1);
        const auto output_strides   = make_tuple(Ho * Wo * problem.C, Wo * problem.C, problem.C, 1);
        const auto window_lengths   = make_tuple(problem.Y, problem.X);
        const auto window_strides   = make_tuple(problem.stride_h, problem.stride_w);
        const auto window_dilations = make_tuple(problem.dilation_h, problem.dilation_w);
        const auto input_left_pads  = make_tuple(problem.pad_h_left, problem.pad_w_left);
        const auto input_right_pads = make_tuple(problem.pad_h_right, problem.pad_w_right);

        auto host_args = PoolHostArgs<decltype(input_shape), decltype(window_lengths)>{
            d_in.GetDeviceBuffer(),
            d_out.GetDeviceBuffer(),
            d_out_index.GetDeviceBuffer(),
            input_shape,
            output_shape,
            input_strides,
            output_strides,
            window_lengths,
            window_strides,
            window_dilations,
            input_left_pads,
            input_right_pads};

        // Launch kernel
        float latency = kernel_func(host_args);

        // Copy back
        d_out.FromDevice(h_out.data());
        d_out_index.FromDevice(h_out_index.data());

        // Verify if requested
        if(setting_.verify)
        {
            auto kernel_args_ref = PoolKernelArgs<decltype(input_shape), decltype(window_lengths)>{
                h_in.data(),
                h_out_ref.data(),
                h_out_ref_index.data(),
                input_shape,
                output_shape,
                input_strides,
                output_strides,
                window_lengths,
                window_strides,
                window_dilations,
                input_left_pads,
                input_right_pads};

            // Use ReduceOp::Max as default for reference
            using ReduceOp = ReduceOp::Max;
            reference_pool2d<InDataType,
                             ComputeDataType,
                             OutDataType,
                             IndexDataType,
                             ReduceOp,
                             decltype(input_shape),
                             decltype(window_lengths),
                             true>(h_in, h_out_ref, h_out_ref_index, kernel_args_ref, ReduceOp{});

            bool pass = check_err(h_out, h_out_ref, "Error: Incorrect results!", 1e-3, 1e-3);
            if(!pass)
            {
                std::cerr << "Verification FAILED!" << std::endl;
            }
            else
            {
                std::cout << "Verification PASSED" << std::endl;
            }
        }

        // Calculate bandwidth
        size_t bytes_read    = problem.input_elements() * sizeof(InDataType);
        size_t bytes_written = problem.output_elements() * sizeof(OutDataType);
        float bandwidth      = (bytes_read + bytes_written) / (latency * 1e-3f) / 1e9f;

        return PoolPerformanceResult{latency, bandwidth};
    }

    private:
    PoolBenchmarkSetting setting_;
};

} // namespace ck_tile
