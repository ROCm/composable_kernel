// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/mhc.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_mhc.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem_v5_4warp.hpp"

// Parse command-line arguments for MHC benchmark
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("B", "1024", "Batch size")
        .insert("n", "4", "Expansion factor (number of streams)")
        .insert("C", "4096", "Channels per stream")
        .insert("v", "1", "CPU validation (0=no, 1=yes)")
        .insert("warmup", "5", "Number of warmup iterations")
        .insert("repeat", "20", "Number of benchmark iterations")
        .insert("r", "2.0", "Norm scaling factor")
        .insert("alpha_pre", "1.5", "Alpha for pre-activation")
        .insert("alpha_post", "2.5", "Alpha for post-activation")
        .insert("alpha_res", "3.5", "Alpha for residual")
        .insert("bias", "1.5", "Bias value");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename XDataType,
          typename PhiDataType,
          typename YDataType,
          typename ComputeDataType,
          typename ActivationFunc = ck_tile::element_wise::Sigmoid,
          ck_tile::index_t MTile  = 16> // Template parameter for M tile size
bool run_mhc_benchmark_impl(const ck_tile::ArgParser& arg_parser)
{
    const int B = arg_parser.get_int("B");
    const int n = arg_parser.get_int("n");
    const int C = arg_parser.get_int("C");

    const int nC         = n * C;
    const int output_dim = 2 * n + n * n;

    const int do_validation = arg_parser.get_int("v");
    const int warmup        = arg_parser.get_int("warmup");
    const int repeat        = arg_parser.get_int("repeat");

    const float r          = arg_parser.get_float("r");
    const float alpha_pre  = arg_parser.get_float("alpha_pre");
    const float alpha_post = arg_parser.get_float("alpha_post");
    const float alpha_res  = arg_parser.get_float("alpha_res");
    const float bias       = arg_parser.get_float("bias");

    std::cout << "\n========================================" << std::endl;
    std::cout << "MHC Kernel V5 Benchmark (BF16) - Split-K" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size (B): " << B << std::endl;
    std::cout << "  Expansion factor (n): " << n << std::endl;
    std::cout << "  Channels per stream (C): " << C << std::endl;
    std::cout << "  Input dimension (nC): " << nC << std::endl;
    std::cout << "  Output dimension (2n+n^2): " << output_dim << std::endl;
    std::cout << "  Data types: X=" << typeid(XDataType).name()
              << ", Phi=" << typeid(PhiDataType).name() << ", Y=" << typeid(YDataType).name()
              << ", Compute=" << typeid(ComputeDataType).name() << std::endl;
    std::cout << "  Warmup iterations: " << warmup << std::endl;
    std::cout << "  Benchmark iterations: " << repeat << std::endl;
    std::cout << "========================================" << std::endl;

    // Allocate host tensors
    ck_tile::HostTensor<XDataType> h_x({B, nC});
    ck_tile::HostTensor<PhiDataType> h_phi({nC, output_dim});
    ck_tile::HostTensor<YDataType> h_output({B, output_dim});

    // Initialize with random data
    ck_tile::FillUniformDistribution<XDataType>{-1.0f, 1.0f}(h_x);
    ck_tile::FillUniformDistribution<PhiDataType>{-0.5f, 0.5f}(h_phi);
    h_output.SetZero();

    // Allocate device memory
    ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_phi_mem(h_phi.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_output_mem(h_output.get_element_space_size_in_bytes());

    // Copy data to device
    d_x_mem.ToDevice(h_x.data());
    d_phi_mem.ToDevice(h_phi.data());
    d_output_mem.ToDevice(h_output.data());

    // Reverted to adaptive 2-warp (4-warp produced incorrect results)
    // using Problem = ck_tile::MHCProblemV5_4Warp<XDataType, ComputeDataType, YDataType>;
    using Problem = ck_tile::MHCProblemV5<XDataType, ComputeDataType, YDataType, MTile>;

    // V5 kernel - split-K implementation with adaptive problem
    using KernelV5 = ck_tile::MHCKernelV5<Problem, ck_tile::MHCDefaultPolicy, ActivationFunc>;
    using ReductionKernel = ck_tile::MHCReductionKernel<Problem, ActivationFunc>;

    const ck_tile::index_t kBlockSize = KernelV5::BlockSize();

    // 2D grid: (batch / kMTile) × (nC / kKTile)
    auto grid_size                   = KernelV5::GetGridSize(B, output_dim, nC);
    const ck_tile::index_t grid_m    = grid_size.at(ck_tile::number<0>{});
    const ck_tile::index_t grid_k    = grid_size.at(ck_tile::number<1>{});
    const ck_tile::index_t kGridSize = grid_m * grid_k;

    std::cout << "\nKernel Configuration:" << std::endl;
    std::cout << "  Grid: " << grid_m << " × " << grid_k << " = " << kGridSize << " blocks"
              << std::endl;
    std::cout << "  Block size: " << kBlockSize << " threads" << std::endl;
    std::cout << "  Shared memory: " << KernelV5::GetSmemSize() << " bytes" << std::endl;
    std::cout << "  Split-K factor: " << grid_k << std::endl;

    // Allocate workspace for split-K partial results
    const std::size_t workspace_size     = grid_k * B * output_dim * sizeof(ComputeDataType);
    const std::size_t partial_norms_size = grid_k * B * sizeof(ComputeDataType);

    ck_tile::DeviceMem d_workspace_mem(workspace_size);
    ck_tile::DeviceMem d_partial_norms_mem(partial_norms_size);

    // Initialize workspace to zero
    (void)hipMemset(d_workspace_mem.GetDeviceBuffer(), 0, workspace_size);
    (void)hipMemset(d_partial_norms_mem.GetDeviceBuffer(), 0, partial_norms_size);

    std::cout << "  Workspace size: " << workspace_size / (1024.0 * 1024.0) << " MB" << std::endl;

    constexpr ck_tile::index_t kBlockPerCu = 1;

    // Reduction kernel configuration
    const ck_tile::index_t reduction_threads = ReductionKernel::BlockSize();
    const ck_tile::index_t reduction_blocks =
        (B * output_dim + reduction_threads - 1) / reduction_threads;

    // Combined kernel launch with timing - warmup and repeat handled by launch_kernel
    auto launch_combined = [&]() {
        // Launch split-K kernel
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false},
            ck_tile::make_kernel<kBlockPerCu>(
                KernelV5{},
                kGridSize,
                kBlockSize,
                KernelV5::GetSmemSize(),
                static_cast<XDataType*>(d_x_mem.GetDeviceBuffer()),
                static_cast<PhiDataType*>(d_phi_mem.GetDeviceBuffer()),
                static_cast<ComputeDataType*>(d_workspace_mem.GetDeviceBuffer()),
                static_cast<ComputeDataType*>(d_partial_norms_mem.GetDeviceBuffer()),
                B,
                nC,
                output_dim,
                n,
                r,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias));

        // Launch reduction kernel
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false},
            ck_tile::make_kernel<kBlockPerCu>(
                ReductionKernel{},
                reduction_blocks,
                reduction_threads,
                0,
                static_cast<ComputeDataType*>(d_workspace_mem.GetDeviceBuffer()),
                static_cast<ComputeDataType*>(d_partial_norms_mem.GetDeviceBuffer()),
                static_cast<YDataType*>(d_output_mem.GetDeviceBuffer()),
                B,
                nC,
                output_dim,
                n,
                grid_k,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias));
    };

    // Warmup
    for(int i = 0; i < warmup; ++i)
    {
        launch_combined();
    }

    // Benchmark with manual timing
    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);

    (void)hipEventRecord(start);
    for(int i = 0; i < repeat; ++i)
    {
        launch_combined();
    }
    (void)hipEventRecord(stop);
    (void)hipEventSynchronize(stop);

    float total_time = 0;
    (void)hipEventElapsedTime(&total_time, start, stop);
    float ave_time = total_time / repeat;

    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    // Calculate performance metrics
    std::size_t num_bytes = sizeof(XDataType) * B * nC +            // Input x
                            sizeof(PhiDataType) * nC * output_dim + // Weights phi
                            sizeof(YDataType) * B * output_dim;     // Output

    float gb_per_sec = num_bytes / 1.E6 / ave_time;

    // Calculate FLOPs: B * output_dim * (2*nC - 1) for GEMM + additional ops
    std::size_t num_flops = static_cast<std::size_t>(B) * output_dim * (2 * nC);
    float tflops          = num_flops / 1.E9 / ave_time;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Average time: " << ave_time << " ms" << std::endl;
    std::cout << "  Bandwidth: " << gb_per_sec << " GB/s" << std::endl;
    std::cout << "  Throughput: " << tflops << " TFLOPS" << std::endl;
    std::cout << "========================================" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        std::cout << "\nRunning validation..." << std::endl;

        d_output_mem.FromDevice(h_output.data());

        // Compute reference
        ck_tile::HostTensor<YDataType> h_output_ref({B, output_dim});
        h_output_ref.SetZero();

        ck_tile::reference_mhc<XDataType, PhiDataType, YDataType, ComputeDataType, ActivationFunc>(
            h_x,
            h_phi,
            h_output_ref,
            n,
            C,
            r,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            ActivationFunc{});

        // Validate with appropriate tolerance for bf16
        float rtol = std::is_same_v<XDataType, ck_tile::bf16_t> ? 1e-2f : 1e-3f;
        float atol = std::is_same_v<XDataType, ck_tile::bf16_t> ? 1e-2f : 1e-3f;

        pass = ck_tile::check_err(
            h_output, h_output_ref, "Error: MHC V5 kernel output mismatch!", rtol, atol);

        std::cout << "Validation: " << (pass ? "PASS" : "FAIL") << std::endl;
    }

    return pass;
}

// Runtime dispatch wrapper for adaptive tile selection
template <typename XDataType,
          typename PhiDataType,
          typename YDataType,
          typename ComputeDataType,
          typename ActivationFunc = ck_tile::element_wise::Sigmoid>
bool run_mhc_benchmark(const ck_tile::ArgParser& arg_parser)
{
    const int B = arg_parser.get_int("B");

    // Adaptive tile selection based on batch size
    // Currently using M=16 (1 warp) for all batch sizes
    // Note: Multi-warp configurations (M=64, M=128) have validation issues
    // that need further investigation. The tile distribution appears correct
    // mathematically but produces incorrect results for B≥4096.
    // TODO: Debug multi-warp tile distribution for large batch sizes
    std::cout << "[Adaptive] Using M=16 tile (1 warp) for batch (B=" << B << ")" << std::endl;
    return run_mhc_benchmark_impl<XDataType,
                                  PhiDataType,
                                  YDataType,
                                  ComputeDataType,
                                  ActivationFunc,
                                  16>(arg_parser);
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cout << "Failed to parse arguments!" << std::endl;
        return -1;
    }

    // Run with BF16 inputs, float output and compute
    // Adaptive tile selection happens inside run_mhc_benchmark
    bool pass = run_mhc_benchmark<ck_tile::bf16_t, // XDataType
                                  ck_tile::bf16_t, // PhiDataType
                                  float,           // YDataType
                                  float,           // ComputeDataType
                                  ck_tile::element_wise::Sigmoid>(arg_parser);

    return pass ? 0 : -2;
}
