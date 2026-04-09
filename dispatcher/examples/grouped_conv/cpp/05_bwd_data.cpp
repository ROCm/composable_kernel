// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Example 05: Backward Data with CPU Reference Validation
//
// Computes dX = ConvBwdData(dY, W) on GPU via dispatcher.run()
// and validates against ck_tile::reference_grouped_conv_bwd_data.
//
// Build: cd dispatcher/build && cmake .. && make grouped_conv_05_bwd_data

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_bwd_data.hpp"

#include "ck_tile/dispatcher/grouped_conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_utils;
using GroupedConvSig  = grouped_conv_decl::GroupedConvSignature;
using GroupedConvAlgo = grouped_conv_decl::GroupedConvAlgorithm;

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;

DECL_GROUPED_CONV_KERNEL_SET(
    bwd_data_kernels,
    .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("bwd_data").dims(2),
         GroupedConvAlgo()
             .tile(1, 128, 128)
             .pipeline("compv3")
             .scheduler("intrawave")
             .vector_sizes(4, 8, 8),
         "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 05: Backward Data Validation",
                            "dX = ConvBwdData(dY, W) with CPU reference");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("-n", "1", "Batch size");
    args.add_option("-c", "64", "Input channels");
    args.add_option("-k", "128", "Output channels");
    args.add_option("--size", "14", "Spatial size (H=W)");

    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 05: Backward Data with CPU Validation");

    std::string gfx_arch = args.get("--arch", "gfx950");
    int N = args.get_int("-n", 1), G = 1;
    int C = args.get_int("-c", 64), K = args.get_int("-k", 128);
    int Hi = args.get_int("--size", 14), Wi = Hi, Y = 3, X = 3;

    // Setup
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

    // dY (gradient from next layer) and W (weight) are inputs; dX is output
    ck_tile::HostTensor<OutDataType> dy(out_desc);
    ck_tile::HostTensor<WeiDataType> weight(wei_desc);
    ck_tile::HostTensor<InDataType> dx_gpu(in_desc);
    ck_tile::HostTensor<InDataType> dx_cpu(in_desc);

    ck_tile::FillUniformDistribution<OutDataType>{-0.5f, 0.5f}(dy);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
    dx_cpu.SetZero();

    // CPU reference
    std::cout << "\nStep 1: CPU Reference (bwd_data)\n";
    std::vector<ck_tile::long_index_t> strides_v    = {1, 1};
    std::vector<ck_tile::long_index_t> dilations_v  = {1, 1};
    std::vector<ck_tile::long_index_t> left_pads_v  = {1, 1};
    std::vector<ck_tile::long_index_t> right_pads_v = {1, 1};

    ck_tile::reference_grouped_conv_bwd_data<2, InDataType, WeiDataType, OutDataType>(
        dx_cpu, weight, dy, strides_v, dilations_v, left_pads_v, right_pads_v);
    std::cout << "  CPU complete\n";

    // GPU execution via dispatcher
    std::cout << "\nStep 2: GPU Execution (via dispatcher.run)\n";

    GroupedConvRegistry registry;
    registry.set_name("bwd_data");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);

    GroupedConvDispatcher dispatcher(&registry);

    auto problem =
        create_grouped_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1, GroupedConvOp::BackwardData);

    auto* selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cerr << "  ERROR: No bwd_data kernel found!\n";
        return 1;
    }
    std::cout << "  Selected: " << selected->name() << "\n";

    ck_tile::DeviceMem dy_dev(dy.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dx_dev(dx_gpu.get_element_space_size_in_bytes());

    dy_dev.ToDevice(dy.data());
    wei_dev.ToDevice(weight.data());

    // dispatcher.run(dY, W, dX, problem) for bwd_data
    float time_ms = dispatcher.run(dy_dev.GetDeviceBuffer(),
                                   wei_dev.GetDeviceBuffer(),
                                   dx_dev.GetDeviceBuffer(),
                                   problem,
                                   nullptr);

    dx_dev.FromDevice(dx_gpu.data());

    double tflops = (time_ms > 0) ? calculate_conv_tflops(problem, time_ms) : 0;
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";

    // Validation
    std::cout << "\nStep 3: Validation (GPU vs CPU)\n";

    size_t num_elements = dx_gpu.get_element_space_size();
    float max_abs = 0, max_rel = 0;
    size_t mismatches    = 0;
    constexpr float rtol = 5e-2f, atol = 5e-2f;

    for(size_t i = 0; i < num_elements; ++i)
    {
        float gv = static_cast<float>(dx_gpu.data()[i]);
        float cv = static_cast<float>(dx_cpu.data()[i]);
        float d  = std::abs(gv - cv);
        float r  = d / (std::abs(cv) + 1e-6f);
        max_abs  = std::max(max_abs, d);
        max_rel  = std::max(max_rel, r);
        if(d > atol + rtol * std::abs(cv))
            ++mismatches;
    }

    bool passed = (mismatches == 0);
    std::cout << "  Elements:     " << num_elements << "\n";
    std::cout << "  Mismatches:   " << mismatches << "\n";
    std::cout << "  Max abs diff: " << std::scientific << max_abs << "\n";
    std::cout << "  Max rel diff: " << std::scientific << max_rel << "\n";

    utils::print_separator();
    std::cout << "  dX = ConvBwdData(dY, W)\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";
    utils::print_separator();

    return passed ? 0 : 1;
}
