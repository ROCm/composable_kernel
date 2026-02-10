// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file pooling_benchmark_single.cpp
 * @brief Single-kernel benchmark for pooling operations.
 *
 * This benchmark includes the generated kernel header via -include flag
 * and runs the pooling kernel with specified problem sizes.
 */

#include <iostream>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "tile_engine/ops/pooling/pooling_common.hpp"
#include "tile_engine/ops/pooling/pooling_benchmark.hpp"

// The kernel header is included via compile command line with -include flag
// It defines: SelectedKernel, KERNEL_NAME, InDataType, OutDataType, etc.

static ck_tile::ArgParser create_args()
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("n", "1", "Batch size (N)")
        .insert("h", "16", "Input height (H)")
        .insert("w", "16", "Input width (W)")
        .insert("c", "32", "Channels (C)")
        .insert("wy", "2", "Window height (Y)")
        .insert("wx", "2", "Window width (X)")
        .insert("sy", "2", "Window stride height")
        .insert("sx", "2", "Window stride width")
        .insert("dy", "1", "Window dilation height")
        .insert("dx", "1", "Window dilation width")
        .insert("phy", "0", "Padding height left")
        .insert("phyr", "0", "Padding height right")
        .insert("pwx", "0", "Padding width left")
        .insert("pwxr", "0", "Padding width right")
        .insert("verify", "1", "Verify results (0/1)")
        .insert("warmup", "5", "Warmup iterations")
        .insert("repeat", "20", "Repeat iterations")
        .insert("log", "1", "Log level");
    return arg_parser;
}

int benchmark_pooling_single(int argc, char* argv[])
{
    auto arg_parser = create_args();
    bool result     = arg_parser.parse(argc, argv);
    if(!result)
        return -1;

    // Parse problem dimensions
    ck_tile::index_t N       = arg_parser.get_int("n");
    ck_tile::index_t H       = arg_parser.get_int("h");
    ck_tile::index_t W       = arg_parser.get_int("w");
    ck_tile::index_t C       = arg_parser.get_int("c");
    ck_tile::index_t Y       = arg_parser.get_int("wy");
    ck_tile::index_t X       = arg_parser.get_int("wx");
    ck_tile::index_t Sy      = arg_parser.get_int("sy");
    ck_tile::index_t Sx      = arg_parser.get_int("sx");
    ck_tile::index_t Dy      = arg_parser.get_int("dy");
    ck_tile::index_t Dx      = arg_parser.get_int("dx");
    ck_tile::index_t LeftPy  = arg_parser.get_int("phy");
    ck_tile::index_t RightPy = arg_parser.get_int("phyr");
    ck_tile::index_t LeftPx  = arg_parser.get_int("pwx");
    ck_tile::index_t RightPx = arg_parser.get_int("pwxr");

    bool verify   = arg_parser.get_int("verify") != 0;
    int warmup    = arg_parser.get_int("warmup");
    int repeat    = arg_parser.get_int("repeat");
    int log_level = arg_parser.get_int("log");

    // Calculate output dimensions
    ck_tile::index_t Ys = (Y - 1) * Dy + 1;
    ck_tile::index_t Xs = (X - 1) * Dx + 1;
    ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
    ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

    std::cout << "Pooling benchmark: " << KERNEL_NAME << std::endl;
    std::cout << "  Input:  NHWC = " << N << "x" << H << "x" << W << "x" << C << std::endl;
    std::cout << "  Output: NHWC = " << N << "x" << Ho << "x" << Wo << "x" << C << std::endl;
    std::cout << "  Window: " << Y << "x" << X << ", stride: " << Sy << "x" << Sx
              << ", dilation: " << Dy << "x" << Dx << std::endl;

    // Create host tensors
    ck_tile::HostTensor<InDataType> h_in({N, H, W, C});
    ck_tile::HostTensor<OutDataType> h_out({N, Ho, Wo, C});
    ck_tile::HostTensor<OutDataType> h_out_ref({N, Ho, Wo, C});
    ck_tile::HostTensor<IndexDataType> h_out_index({N, Ho, Wo, C});
    ck_tile::HostTensor<IndexDataType> h_out_ref_index({N, Ho, Wo, C});

    ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);

    // Device memory
    ck_tile::DeviceMem d_in(h_in.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_out(h_out.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_out_index(h_out_index.get_element_space_size_in_bytes());

    d_in.ToDevice(h_in.data());
    d_out.SetZero();
    d_out_index.SetZero();

    // Build host args
    const auto input_shape      = ck_tile::make_tuple(N, H, W, C);
    const auto output_shape     = ck_tile::make_tuple(N, Ho, Wo, C);
    const auto input_strides    = ck_tile::make_tuple(H * W * C, W * C, C, 1);
    const auto output_strides   = ck_tile::make_tuple(Ho * Wo * C, Wo * C, C, 1);
    const auto window_lengths   = ck_tile::make_tuple(Y, X);
    const auto window_strides   = ck_tile::make_tuple(Sy, Sx);
    const auto window_dilations = ck_tile::make_tuple(Dy, Dx);
    const auto input_left_pads  = ck_tile::make_tuple(LeftPy, LeftPx);
    const auto input_right_pads = ck_tile::make_tuple(RightPy, RightPx);

    auto host_args = ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_lengths)>{
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

    // Stream configuration
    ck_tile::stream_config stream{nullptr, true, log_level, warmup, repeat};

    // Launch kernel
    float latency = 0;
    try
    {
        latency = SelectedKernel::launch(host_args, stream);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Kernel launch failed: " << e.what() << std::endl;
        return -1;
    }

    // Calculate bandwidth
    size_t bytes_read    = static_cast<size_t>(N) * H * W * C * sizeof(InDataType);
    size_t bytes_written = static_cast<size_t>(N) * Ho * Wo * C * sizeof(OutDataType);
    float bandwidth      = (bytes_read + bytes_written) / (latency * 1e-3f) / 1e9f;

    std::cout << "  Latency: " << latency << " ms" << std::endl;
    std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Verify if requested
    if(verify)
    {
        d_out.FromDevice(h_out.data());
        d_out_index.FromDevice(h_out_index.data());

        auto kernel_args = ck_tile::PoolKernelArgs<decltype(input_shape), decltype(window_lengths)>{
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

        ck_tile::reference_pool2d<InDataType,
                                  ComputeDataType,
                                  OutDataType,
                                  IndexDataType,
                                  ReduceOpType,
                                  decltype(input_shape),
                                  decltype(window_lengths),
                                  SelectedKernel::kOutputIndex>(
            h_in, h_out_ref, h_out_ref_index, kernel_args, ReduceOpType{});

        bool pass_value =
            ck_tile::check_err(h_out, h_out_ref, "Error: Incorrect values!", 1e-3, 1e-3);
        std::cout << "  Verification: " << (pass_value ? "PASS" : "FAIL") << std::endl;

        if(SelectedKernel::kOutputIndex)
        {
            bool pass_index =
                ck_tile::check_err(h_out_index, h_out_ref_index, "Error: Incorrect indices!", 0, 0);
            std::cout << "  Index verification: " << (pass_index ? "PASS" : "FAIL") << std::endl;
        }
    }

    return 0;
}

int main(int argc, char* argv[]) { return benchmark_pooling_single(argc, argv); }
