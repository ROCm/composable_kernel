// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <functional>
#include <tuple>
#include <exception>
#include <sstream>
#include <vector>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "pool_benchmark.hpp"
#include "pool_common.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines: InDataType, OutDataType, ComputeDataType, IndexDataType,
//             ReduceOpType, Kernel, Problem, OUTPUT_INDEX, PROPAGATE_NAN,
//             KERNEL_NAME, BLOCK_SHAPE_NAME, REDUCE_OP_NAME

// Create argument parser
inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser
        .insert("N", "2", "Batch size N dimension. Default is 2.")
        .insert("D", "30", "Depth D dimension (for 3D pooling). Default is 30.")
        .insert("H", "30", "Height H dimension. Default is 30.")
        .insert("W", "30", "Width W dimension. Default is 30.")
        .insert("C", "32", "Channel C dimension. Default is 32.")
        .insert("Z", "2", "Window depth Z dimension. Default is 2.")
        .insert("Y", "2", "Window height Y dimension. Default is 2.")
        .insert("X", "2", "Window width X dimension. Default is 2.")
        .insert("Sz", "2", "Window stride depth. Default is 2.")
        .insert("Sy", "2", "Window stride height. Default is 2.")
        .insert("Sx", "2", "Window stride width. Default is 2.")
        .insert("Dz", "1", "Window dilation depth. Default is 1.")
        .insert("Dy", "1", "Window dilation height. Default is 1.")
        .insert("Dx", "1", "Window dilation width. Default is 1.")
        .insert("LeftPz", "0", "Left padding depth. Default is 0.")
        .insert("LeftPy", "0", "Left padding height. Default is 0.")
        .insert("LeftPx", "0", "Left padding width. Default is 0.")
        .insert("RightPz", "0", "Right padding depth. Default is 0.")
        .insert("RightPy", "0", "Right padding height. Default is 0.")
        .insert("RightPx", "0", "Right padding width. Default is 0.")
        .insert("verify",
                "0",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU. "
                "Default is 0.")
        .insert("log",
                "false",
                "Whether output kernel instance information or not. Default is false")
        .insert("warmup", "20", "The number of warmup iterations. Default is 20.")
        .insert("repeat", "100", "The number of benchmark iterations. Default is 100.")
        .insert("timer", "true", "Whether to use GPU timer. Default is true.")
        .insert("init",
                "0",
                "The method of tensor initialization. 0=random, 1=linear, 2=constant(1). Default is 0.")
        .insert("json_output",
                "false",
                "Whether to output results in JSON format only. Default is false");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <bool IsPool3D>
void run_benchmark(const ck_tile::ArgParser& arg_parser)
{
    const ck_tile::index_t N = arg_parser.get_int("N");
    const ck_tile::index_t H = arg_parser.get_int("H");
    const ck_tile::index_t W = arg_parser.get_int("W");
    const ck_tile::index_t C = arg_parser.get_int("C");

    const ck_tile::index_t Y = arg_parser.get_int("Y");
    const ck_tile::index_t X = arg_parser.get_int("X");

    const ck_tile::index_t Sy = arg_parser.get_int("Sy");
    const ck_tile::index_t Sx = arg_parser.get_int("Sx");

    const ck_tile::index_t Dy = arg_parser.get_int("Dy");
    const ck_tile::index_t Dx = arg_parser.get_int("Dx");

    const ck_tile::index_t LeftPy  = arg_parser.get_int("LeftPy");
    const ck_tile::index_t LeftPx  = arg_parser.get_int("LeftPx");
    const ck_tile::index_t RightPy = arg_parser.get_int("RightPy");
    const ck_tile::index_t RightPx = arg_parser.get_int("RightPx");

    const int warmup        = arg_parser.get_int("warmup");
    const int repeat        = arg_parser.get_int("repeat");
    const int do_validation = arg_parser.get_int("verify");
    const int init_method   = arg_parser.get_int("init");
    const bool log          = arg_parser.get_bool("log");
    const bool json_output  = arg_parser.get_bool("json_output");

    if constexpr(IsPool3D)
    {
        // 3D Pooling (NDHWC layout)
        const ck_tile::index_t D = arg_parser.get_int("D");
        const ck_tile::index_t Z = arg_parser.get_int("Z");

        const ck_tile::index_t Sz = arg_parser.get_int("Sz");
        const ck_tile::index_t Dz = arg_parser.get_int("Dz");

        const ck_tile::index_t LeftPz  = arg_parser.get_int("LeftPz");
        const ck_tile::index_t RightPz = arg_parser.get_int("RightPz");

        // Calculate effective window sizes
        const ck_tile::index_t Zs = (Z - 1) * Dz + 1;
        const ck_tile::index_t Ys = (Y - 1) * Dy + 1;
        const ck_tile::index_t Xs = (X - 1) * Dx + 1;

        // Calculate output dimensions
        const ck_tile::index_t Do = (D + LeftPz + RightPz - Zs) / Sz + 1;
        const ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
        const ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

        if(log)
        {
            std::cout << "3D Pooling: N=" << N << ", D=" << D << ", H=" << H << ", W=" << W
                      << ", C=" << C << std::endl;
            std::cout << "Window: Z=" << Z << ", Y=" << Y << ", X=" << X << std::endl;
            std::cout << "Stride: Sz=" << Sz << ", Sy=" << Sy << ", Sx=" << Sx << std::endl;
            std::cout << "Output: Do=" << Do << ", Ho=" << Ho << ", Wo=" << Wo << std::endl;
        }

        // Create shapes using ck_tile::make_tuple
        const auto input_shape    = ck_tile::make_tuple(N, D, H, W, C);
        const auto output_shape   = ck_tile::make_tuple(N, Do, Ho, Wo, C);
        const auto input_strides  = ck_tile::make_tuple(D * H * W * C, H * W * C, W * C, C, 1);
        const auto output_strides = ck_tile::make_tuple(Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, 1);
        const auto window_lengths = ck_tile::make_tuple(Z, Y, X);
        const auto window_strides = ck_tile::make_tuple(Sz, Sy, Sx);
        const auto window_dilations = ck_tile::make_tuple(Dz, Dy, Dx);
        const auto input_left_pads  = ck_tile::make_tuple(LeftPz, LeftPy, LeftPx);
        const auto input_right_pads = ck_tile::make_tuple(RightPz, RightPy, RightPx);

        // Allocate host tensors
        ck_tile::HostTensor<InDataType> in({N, D, H, W, C}, {D * H * W * C, H * W * C, W * C, C, 1});
        ck_tile::HostTensor<OutDataType> out({N, Do, Ho, Wo, C},
                                             {Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, 1});
        ck_tile::HostTensor<IndexDataType> out_index(
            OUTPUT_INDEX ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                                    static_cast<std::size_t>(Do),
                                                    static_cast<std::size_t>(Ho),
                                                    static_cast<std::size_t>(Wo),
                                                    static_cast<std::size_t>(C)}
                         : std::vector<std::size_t>{1});

        // Initialize input
        if(init_method == 0)
        {
            ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(in);
        }
        else if(init_method == 1)
        {
            ck_tile::FillMonotonicSeq<InDataType>{}(in);
        }
        else
        {
            ck_tile::FillConstant<InDataType>{static_cast<InDataType>(1)}(in);
        }

        // Allocate device memory
        ck_tile::DeviceMem in_buf(in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_buf(out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_index_buf(OUTPUT_INDEX ? out_index.get_element_space_size_in_bytes()
                                                      : 0);

        in_buf.ToDevice(in.data());

        // Create host arguments
        auto host_args =
            ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_lengths)>{
                static_cast<InDataType*>(in_buf.GetDeviceBuffer()),
                static_cast<OutDataType*>(out_buf.GetDeviceBuffer()),
                OUTPUT_INDEX ? static_cast<IndexDataType*>(out_index_buf.GetDeviceBuffer())
                             : nullptr,
                input_shape,
                output_shape,
                input_strides,
                output_strides,
                window_lengths,
                window_strides,
                window_dilations,
                input_left_pads,
                input_right_pads};

        auto kernel_args = Kernel::MakeKernelArgs(host_args);

        // Validate arguments
        if(!Kernel::IsSupportedArgument(kernel_args))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping pooling kernel!");
        }

        constexpr ck_tile::index_t kBlockPerCu = 1;
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        const ck_tile::index_t kGridSize       = Kernel::CalculateGridSize(kernel_args);

        if(log)
        {
            std::cout << "Launching kernel: " << KERNEL_NAME << std::endl;
            std::cout << "Grid size: " << kGridSize << ", Block size: " << kBlockSize << std::endl;
        }

        // Launch kernel
        float ave_time = ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, true, log ? 1 : 0, warmup, repeat},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, kGridSize, kBlockSize, 0, kernel_args));

        // Calculate performance metrics
        std::size_t num_bytes =
            sizeof(InDataType) * N * D * H * W * C + sizeof(OutDataType) * N * Do * Ho * Wo * C;
        float gb_per_sec = num_bytes / 1.E6 / ave_time;

        // Output results
        if(json_output)
        {
            std::cout << "{\n"
                      << " \"name\": \"" << KERNEL_NAME << "\",\n"
                      << " \"problem\": {\n"
                      << "    \"N\": " << N << ",\n"
                      << "    \"D\": " << D << ",\n"
                      << "    \"H\": " << H << ",\n"
                      << "    \"W\": " << W << ",\n"
                      << "    \"C\": " << C << ",\n"
                      << "    \"windowZ\": " << Z << ",\n"
                      << "    \"windowY\": " << Y << ",\n"
                      << "    \"windowX\": " << X << "\n"
                      << " },\n"
                      << " \"perf_result\": {\n"
                      << "   \"latency(ms)\": " << ave_time << ",\n"
                      << "   \"bandwidth(GB/s)\": " << gb_per_sec << "\n"
                      << " }\n"
                      << "}" << std::endl;
        }
        else
        {
            std::cout << "Kernel: " << KERNEL_NAME << std::endl;
            std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
        }

        // Verification (if requested)
        if(do_validation)
        {
            out_buf.FromDevice(out.data());

            ck_tile::HostTensor<OutDataType> out_ref({N, Do, Ho, Wo, C},
                                                     {Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, 1});
            ck_tile::HostTensor<IndexDataType> out_ref_index(
                OUTPUT_INDEX ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                                        static_cast<std::size_t>(Do),
                                                        static_cast<std::size_t>(Ho),
                                                        static_cast<std::size_t>(Wo),
                                                        static_cast<std::size_t>(C)}
                             : std::vector<std::size_t>{1});

            ck_tile::reference_pool3d<InDataType,
                                      ComputeDataType,
                                      OutDataType,
                                      IndexDataType,
                                      ReduceOpType,
                                      decltype(input_shape),
                                      decltype(window_lengths),
                                      OUTPUT_INDEX>(
                in, out_ref, out_ref_index, kernel_args, ReduceOpType{});

            bool pass = ck_tile::check_err(out, out_ref);
            if(OUTPUT_INDEX)
            {
                out_index_buf.FromDevice(out_index.data());
                pass = pass && ck_tile::check_err(out_index, out_ref_index);
            }

            std::cout << "Verification: " << (pass ? "PASSED" : "FAILED") << std::endl;
        }
    }
    else
    {
        // 2D Pooling (NHWC layout)
        const ck_tile::index_t Ys = (Y - 1) * Dy + 1;
        const ck_tile::index_t Xs = (X - 1) * Dx + 1;

        const ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
        const ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

        if(log)
        {
            std::cout << "2D Pooling: N=" << N << ", H=" << H << ", W=" << W << ", C=" << C
                      << std::endl;
            std::cout << "Window: Y=" << Y << ", X=" << X << std::endl;
            std::cout << "Stride: Sy=" << Sy << ", Sx=" << Sx << std::endl;
            std::cout << "Output: Ho=" << Ho << ", Wo=" << Wo << std::endl;
        }

        const auto input_shape      = ck_tile::make_tuple(N, H, W, C);
        const auto output_shape     = ck_tile::make_tuple(N, Ho, Wo, C);
        const auto input_strides    = ck_tile::make_tuple(H * W * C, W * C, C, 1);
        const auto output_strides   = ck_tile::make_tuple(Ho * Wo * C, Wo * C, C, 1);
        const auto window_lengths   = ck_tile::make_tuple(Y, X);
        const auto window_strides   = ck_tile::make_tuple(Sy, Sx);
        const auto window_dilations = ck_tile::make_tuple(Dy, Dx);
        const auto input_left_pads  = ck_tile::make_tuple(LeftPy, LeftPx);
        const auto input_right_pads = ck_tile::make_tuple(RightPy, RightPx);

        ck_tile::HostTensor<InDataType> in({N, H, W, C}, {H * W * C, W * C, C, 1});
        ck_tile::HostTensor<OutDataType> out({N, Ho, Wo, C}, {Ho * Wo * C, Wo * C, C, 1});
        ck_tile::HostTensor<IndexDataType> out_index(
            OUTPUT_INDEX ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                                    static_cast<std::size_t>(Ho),
                                                    static_cast<std::size_t>(Wo),
                                                    static_cast<std::size_t>(C)}
                         : std::vector<std::size_t>{1});

        if(init_method == 0)
        {
            ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(in);
        }
        else if(init_method == 1)
        {
            ck_tile::FillMonotonicSeq<InDataType>{}(in);
        }
        else
        {
            ck_tile::FillConstant<InDataType>{static_cast<InDataType>(1)}(in);
        }

        ck_tile::DeviceMem in_buf(in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_buf(out.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_index_buf(OUTPUT_INDEX ? out_index.get_element_space_size_in_bytes()
                                                      : 0);

        in_buf.ToDevice(in.data());

        auto host_args =
            ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_lengths)>{
                static_cast<InDataType*>(in_buf.GetDeviceBuffer()),
                static_cast<OutDataType*>(out_buf.GetDeviceBuffer()),
                OUTPUT_INDEX ? static_cast<IndexDataType*>(out_index_buf.GetDeviceBuffer())
                             : nullptr,
                input_shape,
                output_shape,
                input_strides,
                output_strides,
                window_lengths,
                window_strides,
                window_dilations,
                input_left_pads,
                input_right_pads};

        auto kernel_args = Kernel::MakeKernelArgs(host_args);

        if(!Kernel::IsSupportedArgument(kernel_args))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping pooling kernel!");
        }

        constexpr ck_tile::index_t kBlockPerCu = 1;
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        const ck_tile::index_t kGridSize       = Kernel::CalculateGridSize(kernel_args);

        if(log)
        {
            std::cout << "Launching kernel: " << KERNEL_NAME << std::endl;
            std::cout << "Grid size: " << kGridSize << ", Block size: " << kBlockSize << std::endl;
        }

        float ave_time = ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, true, log ? 1 : 0, warmup, repeat},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, kGridSize, kBlockSize, 0, kernel_args));

        std::size_t num_bytes =
            sizeof(InDataType) * N * H * W * C + sizeof(OutDataType) * N * Ho * Wo * C;
        float gb_per_sec = num_bytes / 1.E6 / ave_time;

        if(json_output)
        {
            std::cout << "{\n"
                      << " \"name\": \"" << KERNEL_NAME << "\",\n"
                      << " \"problem\": {\n"
                      << "    \"N\": " << N << ",\n"
                      << "    \"H\": " << H << ",\n"
                      << "    \"W\": " << W << ",\n"
                      << "    \"C\": " << C << ",\n"
                      << "    \"windowY\": " << Y << ",\n"
                      << "    \"windowX\": " << X << "\n"
                      << " },\n"
                      << " \"perf_result\": {\n"
                      << "   \"latency(ms)\": " << ave_time << ",\n"
                      << "   \"bandwidth(GB/s)\": " << gb_per_sec << "\n"
                      << " }\n"
                      << "}" << std::endl;
        }
        else
        {
            std::cout << "Kernel: " << KERNEL_NAME << std::endl;
            std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
        }

        if(do_validation)
        {
            out_buf.FromDevice(out.data());

            ck_tile::HostTensor<OutDataType> out_ref({N, Ho, Wo, C}, {Ho * Wo * C, Wo * C, C, 1});
            ck_tile::HostTensor<IndexDataType> out_ref_index(
                OUTPUT_INDEX ? std::vector<std::size_t>{static_cast<std::size_t>(N),
                                                        static_cast<std::size_t>(Ho),
                                                        static_cast<std::size_t>(Wo),
                                                        static_cast<std::size_t>(C)}
                             : std::vector<std::size_t>{1});

            ck_tile::reference_pool2d<InDataType,
                                      ComputeDataType,
                                      OutDataType,
                                      IndexDataType,
                                      ReduceOpType,
                                      decltype(input_shape),
                                      decltype(window_lengths),
                                      OUTPUT_INDEX>(
                in, out_ref, out_ref_index, kernel_args, ReduceOpType{});

            bool pass = ck_tile::check_err(out, out_ref);
            if(OUTPUT_INDEX)
            {
                out_index_buf.FromDevice(out_index.data());
                pass = pass && ck_tile::check_err(out_index, out_ref_index);
            }

            std::cout << "Verification: " << (pass ? "PASSED" : "FAILED") << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv);
        if(!result)
            return EXIT_FAILURE;

        // POOL_DIM is defined in the generated header (2 or 3)
        if constexpr(POOL_DIM == 3)
        {
            run_benchmark<true>(parser);
        }
        else
        {
            run_benchmark<false>(parser);
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
