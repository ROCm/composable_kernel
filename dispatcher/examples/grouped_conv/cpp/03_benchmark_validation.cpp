// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Example 03: Benchmark and CPU-Reference Validation
//
// Runs a 2D grouped conv forward kernel on the GPU via dispatcher.run()
// and compares against the CK Tile host reference implementation.
// Exposes warmup/repeat/log_level as CLI args (matches example 20 pattern).
//
// Build: cd dispatcher/build && cmake .. && make grouped_conv_03_bench_val

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_fwd.hpp"

#include "ck_tile/dispatcher/grouped_conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_utils;
using GroupedConvSig  = grouped_conv_decl::GroupedConvSignature;
using GroupedConvAlgo = grouped_conv_decl::GroupedConvAlgorithm;

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;
using AccDataType = float;

DECL_GROUPED_CONV_KERNEL_SET(
    bench_kernels,
    .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv4").vector_sizes(4, 8, 8),
         "gfx950")
        .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
             GroupedConvAlgo().tile(1, 64, 64).pipeline("compv3").vector_sizes(4, 8, 8),
             "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 03: Benchmark & Validation",
                            "GPU execution with CPU reference validation");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-g", "1", "Groups G");
    args.add_option("-c", "64", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("--size", "14", "Spatial size (H=W)");
    args.add_option("--warmup", "3", "Warmup iterations");
    args.add_option("--repeat", "10", "Benchmark iterations");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_flag("--no-verify", "Skip CPU validation");

    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 03: Grouped Conv Benchmark & Validation");

    int N  = args.get_int("-n", 1);
    int G  = args.get_int("-g", 1);
    int C  = args.get_int("-c", 64);
    int K  = args.get_int("-k", 128);
    int Hi = args.get_int("--size", 14);
    int Wi = Hi;
    int Y = 3, X = 3;
    int warmup           = args.get_int("--warmup", 3);
    int repeat           = args.get_int("--repeat", 10);
    bool verify          = !args.has("--no-verify");
    std::string gfx_arch = args.get("--arch", "gfx950");

    std::cout << "\nProblem: N=" << N << " G=" << G << " C=" << C << " K=" << K << " Hi=" << Hi
              << " Wi=" << Wi << " Y=" << Y << " X=" << X << "\n";
    std::cout << "Benchmark: warmup=" << warmup << " repeat=" << repeat << "\n";

    // Step 1: Setup tensors using CK Tile descriptors
    std::cout << "\nStep 1: Setup tensors\n";

    ck_tile::conv::ConvParam conv_param{
        2,
        static_cast<ck_tile::index_t>(G),
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(C),
        {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},
        {static_cast<ck_tile::index_t>(Hi), static_cast<ck_tile::index_t>(Wi)},
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
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    auto out_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    ck_tile::HostTensor<InDataType> input(in_desc);
    ck_tile::HostTensor<WeiDataType> weight(wei_desc);
    ck_tile::HostTensor<OutDataType> output_gpu(out_desc);
    ck_tile::HostTensor<OutDataType> output_cpu(out_desc);

    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
    output_cpu.SetZero();

    std::cout << "  Input:  " << input.get_element_space_size() << " elements\n";
    std::cout << "  Weight: " << weight.get_element_space_size() << " elements\n";
    std::cout << "  Output: " << output_gpu.get_element_space_size() << " elements\n";

    // Step 2: CPU reference
    if(verify)
    {
        std::cout << "\nStep 2: CPU Reference\n";

        std::vector<ck_tile::long_index_t> strides_v    = {1, 1};
        std::vector<ck_tile::long_index_t> dilations_v  = {1, 1};
        std::vector<ck_tile::long_index_t> left_pads_v  = {1, 1};
        std::vector<ck_tile::long_index_t> right_pads_v = {1, 1};

        ck_tile::reference_grouped_conv_fwd<2, InDataType, WeiDataType, OutDataType>(
            input, weight, output_cpu, strides_v, dilations_v, left_pads_v, right_pads_v);

        std::cout << "  CPU ref[0..7]: ";
        for(int i = 0; i < std::min(8, static_cast<int>(output_cpu.get_element_space_size())); ++i)
            std::cout << std::fixed << std::setprecision(4)
                      << static_cast<float>(output_cpu.data()[i]) << " ";
        std::cout << "\n";
    }

    // Step 3: GPU execution via dispatcher
    std::cout << "\nStep 3: GPU Execution (via dispatcher.run)\n";

    GroupedConvRegistry registry;
    registry.set_name("bench_val");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    GroupedConvDispatcher dispatcher(&registry);

    auto problem = create_grouped_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1);
    problem.op   = GroupedConvOp::Forward;

    auto* selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cerr << "  ERROR: No kernel found!\n";
        return 1;
    }
    std::cout << "  Selected: " << selected->name() << "\n";

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output_gpu.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());

    float elapsed_ms = dispatcher.run(input_dev.GetDeviceBuffer(),
                                      weight_dev.GetDeviceBuffer(),
                                      output_dev.GetDeviceBuffer(),
                                      problem,
                                      nullptr);

    output_dev.FromDevice(output_gpu.data());

    size_t total = output_gpu.get_element_space_size();
    std::cout << "  GPU out[0..7]: ";
    for(int i = 0; i < std::min(8, static_cast<int>(total)); ++i)
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(output_gpu.data()[i])
                  << " ";
    std::cout << "\n";

    size_t nonzero_gpu = 0;
    double gpu_sum     = 0.0;
    for(size_t i = 0; i < total; ++i)
    {
        float v = static_cast<float>(output_gpu.data()[i]);
        if(v != 0.0f)
            ++nonzero_gpu;
        gpu_sum += v;
    }
    std::cout << "  GPU checksum:  " << std::fixed << std::setprecision(6) << gpu_sum << "\n";
    std::cout << "  GPU non-zero:  " << nonzero_gpu << "/" << total
              << (nonzero_gpu > 0 ? " (kernel produced output)" : " WARNING: all zeros!") << "\n";

    int Ho        = static_cast<int>(problem.Ho());
    int Wo        = static_cast<int>(problem.Wo());
    double flops  = 2.0 * G * N * K * C * Y * X * Ho * Wo;
    double tflops = flops / (elapsed_ms * 1e9);

    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";

    // Step 4: Validation
    bool passed = true;
    if(verify)
    {
        std::cout << "\nStep 4: Validation (GPU vs CPU)\n";

        constexpr float rtol = 1e-2f;
        constexpr float atol = 1e-2f;

        float max_diff      = 0.0f;
        float max_rel       = 0.0f;
        size_t max_diff_idx = 0;
        size_t num_elements = output_gpu.get_element_space_size();
        size_t mismatches   = 0;

        for(size_t i = 0; i < num_elements; ++i)
        {
            float gpu_val = static_cast<float>(output_gpu.data()[i]);
            float cpu_val = static_cast<float>(output_cpu.data()[i]);
            float diff    = std::abs(gpu_val - cpu_val);
            float tol     = atol + rtol * std::abs(cpu_val);
            float rel     = diff / (std::abs(cpu_val) + 1e-6f);
            if(diff > max_diff)
            {
                max_diff     = diff;
                max_diff_idx = i;
            }
            max_rel = std::max(max_rel, rel);
            if(diff > tol)
                ++mismatches;
        }

        passed = (mismatches == 0);

        std::cout << "  Side-by-side at worst element [" << max_diff_idx << "]:\n";
        std::cout << "    GPU: " << std::fixed << std::setprecision(6)
                  << static_cast<float>(output_gpu.data()[max_diff_idx])
                  << "  CPU: " << static_cast<float>(output_cpu.data()[max_diff_idx])
                  << "  diff: " << std::scientific << max_diff << "\n";
        std::cout << "  Elements:     " << num_elements << "\n";
        std::cout << "  Mismatches:   " << mismatches << "/" << num_elements << "\n";
        std::cout << "  Max abs diff: " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff: " << std::scientific << max_rel << "\n";
        std::cout << "  Status:       " << (passed ? "PASSED" : "FAILED") << "\n";
    }

    utils::print_separator();
    std::cout << "BENCHMARK & VALIDATION:\n";
    std::cout << "  GPU kernel:     " << (selected ? selected->name() : "none") << "\n";
    std::cout << "  Performance:    " << std::fixed << std::setprecision(2) << tflops
              << " TFLOPS\n";
    std::cout << "  CPU reference:  reference_grouped_conv_fwd<2>()\n";
    std::cout << "  Validation:     " << (passed ? "PASS" : "FAIL") << "\n";
    utils::print_separator();

    return passed ? 0 : 1;
}
