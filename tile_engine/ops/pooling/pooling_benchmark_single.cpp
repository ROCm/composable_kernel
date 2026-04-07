// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file pooling_benchmark_single.cpp
 * @brief Single-kernel benchmark for pooling operations (2D and 3D).
 *
 * This benchmark includes the generated kernel header via -include flag
 * and runs the pooling kernel with specified problem sizes.
 *
 * The generated header provides:
 *   - SelectedKernel   (struct with ::launch())
 *   - KERNEL_NAME      (constexpr const char*)
 *   - POOLING_DIM      (constexpr int, 2 or 3)
 *   - InDataType, OutDataType, ComputeDataType, IndexDataType, ReduceOpType
 *   - TensorShape, WindowShape
 */

#include <iostream>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "pooling_common.hpp"
#include "pooling_benchmark.hpp"

// The kernel header is included via compile command line with -include flag.

// --------------------------------------------------------------------------
// Benchmark implementation — templated on pooling dimension so that only
// the matching branch is instantiated (2D or 3D).
// --------------------------------------------------------------------------

template <typename HostArgs>
static float launch_selected_kernel(HostArgs& args, const ck_tile::stream_config& stream)
{
    return SelectedKernel::launch(args, stream);
}

template <int PoolDim>
static int benchmark_pooling(int argc, char* argv[])
{
    if constexpr(PoolDim == 2)
    {
        // ---- 2D argument parser ----
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

        if(!arg_parser.parse(argc, argv))
            return -1;

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

        ck_tile::index_t Ys = (Y - 1) * Dy + 1;
        ck_tile::index_t Xs = (X - 1) * Dx + 1;
        ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
        ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

        std::cout << "Pooling 2D benchmark: " << KERNEL_NAME << std::endl;
        std::cout << "  Input:  NHWC = " << N << "x" << H << "x" << W << "x" << C << std::endl;
        std::cout << "  Output: NHWC = " << N << "x" << Ho << "x" << Wo << "x" << C << std::endl;
        std::cout << "  Window: " << Y << "x" << X << ", stride: " << Sy << "x" << Sx
                  << ", dilation: " << Dy << "x" << Dx << std::endl;

        ck_tile::HostTensor<InDataType> h_in({N, H, W, C});
        ck_tile::HostTensor<OutDataType> h_out({N, Ho, Wo, C});
        ck_tile::HostTensor<OutDataType> h_out_ref({N, Ho, Wo, C});
        ck_tile::HostTensor<IndexDataType> h_out_index({N, Ho, Wo, C});
        ck_tile::HostTensor<IndexDataType> h_out_ref_index({N, Ho, Wo, C});

        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);

        ck_tile::DeviceMem d_in(h_in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out(h_out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out_index(h_out_index.get_element_space_size_in_bytes());

        d_in.ToDevice(h_in.data());
        d_out.SetZero();
        d_out_index.SetZero();

        auto input_shape      = ck_tile::make_tuple(N, H, W, C);
        auto output_shape     = ck_tile::make_tuple(N, Ho, Wo, C);
        auto input_strides    = ck_tile::make_tuple(H * W * C, W * C, C, ck_tile::index_t{1});
        auto output_strides   = ck_tile::make_tuple(Ho * Wo * C, Wo * C, C, ck_tile::index_t{1});
        auto window_lengths   = ck_tile::make_tuple(Y, X);
        auto window_strides   = ck_tile::make_tuple(Sy, Sx);
        auto window_dilations = ck_tile::make_tuple(Dy, Dx);
        auto input_left_pads  = ck_tile::make_tuple(LeftPy, LeftPx);
        auto input_right_pads = ck_tile::make_tuple(RightPy, RightPx);

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

        ck_tile::stream_config stream{nullptr, true, log_level, warmup, repeat};

        float latency = 0;
        try
        {
            latency = launch_selected_kernel(host_args, stream);
        }
        catch(const std::exception& e)
        {
            std::cerr << "Kernel launch failed: " << e.what() << std::endl;
            return -1;
        }

        size_t bytes_read    = static_cast<size_t>(N) * H * W * C * sizeof(InDataType);
        size_t bytes_written = static_cast<size_t>(N) * Ho * Wo * C * sizeof(OutDataType);
        float bandwidth      = (bytes_read + bytes_written) / (latency * 1e-3f) / 1e9f;

        std::cout << "  Latency: " << latency << " ms" << std::endl;
        std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

        if(verify)
        {
            d_out.FromDevice(h_out.data());
            d_out_index.FromDevice(h_out_index.data());

            auto kernel_args =
                ck_tile::PoolKernelArgs<decltype(input_shape), decltype(window_lengths)>{
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
                bool pass_index = ck_tile::check_err(
                    h_out_index, h_out_ref_index, "Error: Incorrect indices!", 0, 0);
                std::cout << "  Index verification: " << (pass_index ? "PASS" : "FAIL")
                          << std::endl;
            }
        }

        return 0;
    }
    else // PoolDim == 3
    {
        // ---- 3D argument parser ----
        ck_tile::ArgParser arg_parser;
        arg_parser.insert("n", "1", "Batch size (N)")
            .insert("d", "4", "Input depth (D)")
            .insert("h", "16", "Input height (H)")
            .insert("w", "16", "Input width (W)")
            .insert("c", "32", "Channels (C)")
            .insert("wz", "2", "Window depth (Z)")
            .insert("wy", "2", "Window height (Y)")
            .insert("wx", "2", "Window width (X)")
            .insert("sz", "2", "Window stride depth")
            .insert("sy", "2", "Window stride height")
            .insert("sx", "2", "Window stride width")
            .insert("dz", "1", "Window dilation depth")
            .insert("dy", "1", "Window dilation height")
            .insert("dx", "1", "Window dilation width")
            .insert("pdz", "0", "Padding depth left")
            .insert("pdzr", "0", "Padding depth right")
            .insert("phy", "0", "Padding height left")
            .insert("phyr", "0", "Padding height right")
            .insert("pwx", "0", "Padding width left")
            .insert("pwxr", "0", "Padding width right")
            .insert("verify", "1", "Verify results (0/1)")
            .insert("warmup", "5", "Warmup iterations")
            .insert("repeat", "20", "Repeat iterations")
            .insert("log", "1", "Log level");

        if(!arg_parser.parse(argc, argv))
            return -1;

        ck_tile::index_t N       = arg_parser.get_int("n");
        ck_tile::index_t D       = arg_parser.get_int("d");
        ck_tile::index_t H       = arg_parser.get_int("h");
        ck_tile::index_t W       = arg_parser.get_int("w");
        ck_tile::index_t C       = arg_parser.get_int("c");
        ck_tile::index_t Z       = arg_parser.get_int("wz");
        ck_tile::index_t Y       = arg_parser.get_int("wy");
        ck_tile::index_t X       = arg_parser.get_int("wx");
        ck_tile::index_t Sz      = arg_parser.get_int("sz");
        ck_tile::index_t Sy      = arg_parser.get_int("sy");
        ck_tile::index_t Sx      = arg_parser.get_int("sx");
        ck_tile::index_t Dz      = arg_parser.get_int("dz");
        ck_tile::index_t Dy      = arg_parser.get_int("dy");
        ck_tile::index_t Dx      = arg_parser.get_int("dx");
        ck_tile::index_t LeftPz  = arg_parser.get_int("pdz");
        ck_tile::index_t RightPz = arg_parser.get_int("pdzr");
        ck_tile::index_t LeftPy  = arg_parser.get_int("phy");
        ck_tile::index_t RightPy = arg_parser.get_int("phyr");
        ck_tile::index_t LeftPx  = arg_parser.get_int("pwx");
        ck_tile::index_t RightPx = arg_parser.get_int("pwxr");

        bool verify   = arg_parser.get_int("verify") != 0;
        int warmup    = arg_parser.get_int("warmup");
        int repeat    = arg_parser.get_int("repeat");
        int log_level = arg_parser.get_int("log");

        ck_tile::index_t Zs = (Z - 1) * Dz + 1;
        ck_tile::index_t Ys = (Y - 1) * Dy + 1;
        ck_tile::index_t Xs = (X - 1) * Dx + 1;
        ck_tile::index_t Do = (D + LeftPz + RightPz - Zs) / Sz + 1;
        ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
        ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

        std::cout << "Pooling 3D benchmark: " << KERNEL_NAME << std::endl;
        std::cout << "  Input:  NDHWC = " << N << "x" << D << "x" << H << "x" << W << "x" << C
                  << std::endl;
        std::cout << "  Output: NDHWC = " << N << "x" << Do << "x" << Ho << "x" << Wo << "x" << C
                  << std::endl;
        std::cout << "  Window: " << Z << "x" << Y << "x" << X << ", stride: " << Sz << "x" << Sy
                  << "x" << Sx << ", dilation: " << Dz << "x" << Dy << "x" << Dx << std::endl;

        ck_tile::HostTensor<InDataType> h_in({N, D, H, W, C});
        ck_tile::HostTensor<OutDataType> h_out({N, Do, Ho, Wo, C});
        ck_tile::HostTensor<OutDataType> h_out_ref({N, Do, Ho, Wo, C});
        ck_tile::HostTensor<IndexDataType> h_out_index({N, Do, Ho, Wo, C});
        ck_tile::HostTensor<IndexDataType> h_out_ref_index({N, Do, Ho, Wo, C});

        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(h_in);

        ck_tile::DeviceMem d_in(h_in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out(h_out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_out_index(h_out_index.get_element_space_size_in_bytes());

        d_in.ToDevice(h_in.data());
        d_out.SetZero();
        d_out_index.SetZero();

        auto input_shape  = ck_tile::make_tuple(N, D, H, W, C);
        auto output_shape = ck_tile::make_tuple(N, Do, Ho, Wo, C);
        auto input_strides =
            ck_tile::make_tuple(D * H * W * C, H * W * C, W * C, C, ck_tile::index_t{1});
        auto output_strides =
            ck_tile::make_tuple(Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, ck_tile::index_t{1});
        auto window_lengths   = ck_tile::make_tuple(Z, Y, X);
        auto window_strides   = ck_tile::make_tuple(Sz, Sy, Sx);
        auto window_dilations = ck_tile::make_tuple(Dz, Dy, Dx);
        auto input_left_pads  = ck_tile::make_tuple(LeftPz, LeftPy, LeftPx);
        auto input_right_pads = ck_tile::make_tuple(RightPz, RightPy, RightPx);

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

        ck_tile::stream_config stream{nullptr, true, log_level, warmup, repeat};

        float latency = 0;
        try
        {
            latency = launch_selected_kernel(host_args, stream);
        }
        catch(const std::exception& e)
        {
            std::cerr << "Kernel launch failed: " << e.what() << std::endl;
            return -1;
        }

        size_t bytes_read    = static_cast<size_t>(N) * D * H * W * C * sizeof(InDataType);
        size_t bytes_written = static_cast<size_t>(N) * Do * Ho * Wo * C * sizeof(OutDataType);
        float bandwidth      = (bytes_read + bytes_written) / (latency * 1e-3f) / 1e9f;

        std::cout << "  Latency: " << latency << " ms" << std::endl;
        std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

        if(verify)
        {
            d_out.FromDevice(h_out.data());
            d_out_index.FromDevice(h_out_index.data());

            auto kernel_args =
                ck_tile::PoolKernelArgs<decltype(input_shape), decltype(window_lengths)>{
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

            ck_tile::reference_pool3d<InDataType,
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
                bool pass_index = ck_tile::check_err(
                    h_out_index, h_out_ref_index, "Error: Incorrect indices!", 0, 0);
                std::cout << "  Index verification: " << (pass_index ? "PASS" : "FAIL")
                          << std::endl;
            }
        }

        return 0;
    }
}

int main(int argc, char* argv[]) { return benchmark_pooling<POOLING_DIM>(argc, argv); }
