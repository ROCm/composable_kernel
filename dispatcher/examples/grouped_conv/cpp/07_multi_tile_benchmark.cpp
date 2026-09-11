// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Example 07: Multi-Tile Benchmark
//
// Benchmarks multiple tile configurations across ResNet-like problem sizes.
// Exposes warmup, repeat, and init method as CLI args (matching CK Tile
// example 20 patterns).
//
// Build: cd dispatcher/build && cmake .. && make grouped_conv_07_benchmark

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

#include "ck_tile/dispatcher/grouped_conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_utils;
using GroupedConvSig  = grouped_conv_decl::GroupedConvSignature;
using GroupedConvAlgo = grouped_conv_decl::GroupedConvAlgorithm;

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;

// Multiple tile configurations for benchmarking
DECL_GROUPED_CONV_KERNEL_SET(
    benchmark_tiles,
    // Small tile - compv3
    .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
         GroupedConvAlgo()
             .tile(1, 64, 64)
             .wave(1, 4, 1)
             .warp(16, 16, 32)
             .pipeline("compv3")
             .scheduler("intrawave")
             .epilogue("cshuffle")
             .vector_sizes(4, 8, 8)
             .block_per_cu(1),
         "gfx950")
        // Medium tile - compv3
        .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
             GroupedConvAlgo()
                 .tile(1, 128, 128)
                 .wave(2, 2, 1)
                 .warp(32, 32, 16)
                 .pipeline("compv3")
                 .scheduler("intrawave")
                 .epilogue("cshuffle")
                 .vector_sizes(4, 8, 8)
                 .block_per_cu(1),
             "gfx950")
        // Large tile - compv4 with double smem buffer
        .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
             GroupedConvAlgo()
                 .tile(1, 256, 256)
                 .wave(2, 2, 1)
                 .warp(32, 32, 16)
                 .pipeline("compv4")
                 .scheduler("intrawave")
                 .epilogue("cshuffle")
                 .vector_sizes(4, 8, 8)
                 .block_per_cu(1),
             "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 07: Multi-Tile Benchmark",
                            "Multiple tiles across ResNet-like problem sizes");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--warmup", "5", "Warmup iterations (passed to stream_config)");
    args.add_option("--repeat", "20", "Benchmark iterations (passed to stream_config)");
    args.add_option("--init", "0", "Init method: 0=random, 1=linear, 2=constant(1)");

    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 07: Multi-Tile Benchmark");

    std::string gfx_arch = args.get("--arch", "gfx950");
    int warmup           = args.get_int("--warmup", 5);
    int repeat           = args.get_int("--repeat", 20);
    int init_method      = args.get_int("--init", 0);

    std::cout << "\n  Config: warmup=" << warmup << " repeat=" << repeat << " init=" << init_method
              << "\n";

    GroupedConvRegistry registry;
    registry.set_name("benchmark");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    GroupedConvDispatcher dispatcher(&registry);

    // ResNet-like problem sizes
    struct BenchProblem
    {
        const char* label;
        int N, C, K, Hi, Wi, Y, X;
    };

    BenchProblem problems[] = {
        {"ResNet-stage2", 1, 64, 64, 56, 56, 3, 3},
        {"ResNet-stage3", 1, 128, 128, 28, 28, 3, 3},
        {"ResNet-stage4", 1, 256, 256, 14, 14, 3, 3},
        {"ResNet-stage5", 1, 512, 512, 7, 7, 3, 3},
        {"Pointwise-1x1", 1, 256, 256, 56, 56, 1, 1},
        {"Batch-8", 8, 64, 128, 56, 56, 3, 3},
    };

    std::cout << "\n  " << std::left << std::setw(16) << "Problem" << std::right << std::setw(5)
              << "N" << std::setw(5) << "C" << std::setw(5) << "K" << std::setw(5) << "H"
              << std::setw(5) << "W" << std::setw(4) << "F" << std::setw(10) << "Time(ms)"
              << std::setw(10) << "TFLOPS" << std::setw(10) << "Status" << "\n";
    std::cout << "  " << std::string(74, '-') << "\n";

    bool all_pass = true;
    for(const auto& bp : problems)
    {
        auto problem =
            create_grouped_conv2d_problem(bp.N, bp.C, bp.K, bp.Hi, bp.Wi, bp.Y, bp.X, 1, 1);
        problem.op = GroupedConvOp::Forward;

        ck_tile::conv::ConvParam conv_param{
            2,
            static_cast<ck_tile::index_t>(1),
            static_cast<ck_tile::index_t>(bp.N),
            static_cast<ck_tile::index_t>(bp.K),
            static_cast<ck_tile::index_t>(bp.C),
            {static_cast<ck_tile::index_t>(bp.Y), static_cast<ck_tile::index_t>(bp.X)},
            {static_cast<ck_tile::index_t>(bp.Hi), static_cast<ck_tile::index_t>(bp.Wi)},
            {1, 1},
            {1, 1},
            {1, 1},
            {1, 1}};

        using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
        using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;
        using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;

        auto in_desc =
            ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
        auto wei_desc =
            ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);
        auto out_desc =
            ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        ck_tile::HostTensor<InDataType> input(in_desc);
        ck_tile::HostTensor<WeiDataType> weight(wei_desc);
        ck_tile::HostTensor<OutDataType> output(out_desc);

        switch(init_method)
        {
        case 1:
            ck_tile::FillMonotonicSeq<InDataType>{0.0f, 0.001f}(input);
            ck_tile::FillMonotonicSeq<WeiDataType>{0.0f, 0.001f}(weight);
            break;
        case 2:
            ck_tile::FillConstant<InDataType>{1.0f}(input);
            ck_tile::FillConstant<WeiDataType>{1.0f}(weight);
            break;
        default:
            ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
            ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
            break;
        }
        ck_tile::DeviceMem in_dev(input.get_element_space_size_in_bytes());
        ck_tile::DeviceMem wei_dev(weight.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_dev(output.get_element_space_size_in_bytes());

        in_dev.ToDevice(input.data());
        wei_dev.ToDevice(weight.data());

        float time_ms = 0;
        bool ok       = false;
        try
        {
            time_ms = dispatcher.run(in_dev.GetDeviceBuffer(),
                                     wei_dev.GetDeviceBuffer(),
                                     out_dev.GetDeviceBuffer(),
                                     problem,
                                     nullptr);

            out_dev.FromDevice(output.data());
            size_t nz = 0;
            for(size_t j = 0; j < output.get_element_space_size(); ++j)
                if(static_cast<float>(output.data()[j]) != 0.0f)
                    ++nz;
            ok = nz > 0;
        }
        catch(const std::exception&)
        {
            ok = false;
        }

        double tflops = (time_ms > 0) ? calculate_conv_tflops(problem, time_ms) : 0;

        std::string filter_str = std::to_string(bp.Y) + "x" + std::to_string(bp.X);
        std::cout << "  " << std::left << std::setw(16) << bp.label << std::right << std::setw(5)
                  << bp.N << std::setw(5) << bp.C << std::setw(5) << bp.K << std::setw(5) << bp.Hi
                  << std::setw(5) << bp.Wi << std::setw(4) << filter_str << std::fixed
                  << std::setprecision(4) << std::setw(10) << time_ms << std::setprecision(2)
                  << std::setw(10) << tflops << std::setw(10) << (ok ? "OK" : "FAIL") << "\n";
        if(!ok)
            all_pass = false;
    }

    utils::print_separator();
    std::cout << "  Warmup: " << warmup << ", Repeat: " << repeat << ", Init: " << init_method
              << "\n";
    std::cout << "  Status: " << (all_pass ? "PASS" : "FAIL") << "\n";
    utils::print_separator();

    return all_pass ? 0 : 1;
}
