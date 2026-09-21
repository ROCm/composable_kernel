// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Example 04: Heuristic Selection + JSON Export
//
// Demonstrates runtime kernel selection with heuristic ranking,
// GPU execution, and JSON registry export.
//
// Build: cd dispatcher/build && cmake .. && make grouped_conv_04_registry_json

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

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

// Two tile configs for heuristic selection
DECL_GROUPED_CONV_KERNEL_SET(
    heuristic_kernels,
    .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv4").vector_sizes(4, 8, 8),
         "gfx950")
        .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
             GroupedConvAlgo().tile(1, 64, 64).pipeline("compv3").vector_sizes(4, 8, 8),
             "gfx950"));

std::vector<std::string> conv_heuristic(const GroupedConvProblem& problem)
{
    int64_t spatial = problem.Ho() * problem.Wo();
    if(spatial > 400)
        return {"128x128", "64x64"};
    return {"64x64", "128x128"};
}

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 04: Heuristic + JSON",
                            "Runtime kernel selection and JSON export");
    args.add_option("--arch", "gfx950", "GPU architecture");

    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 04: Heuristic Selection + JSON Export");

    std::string gfx_arch = args.get("--arch", "gfx950");

    // Step 1: Register
    std::cout << "\nStep 1: Register Kernels" << std::endl;
    GroupedConvRegistry registry;
    registry.set_name("heuristic_conv");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)" << std::endl;

    // Step 2: Heuristic dispatcher
    std::cout << "\nStep 2: Heuristic Dispatcher" << std::endl;
    GroupedConvDispatcher dispatcher(&registry);
    dispatcher.set_strategy(GroupedConvDispatcher::SelectionStrategy::Heuristic);
    dispatcher.set_heuristic(conv_heuristic);

    // Step 3: Select kernels (no GPU yet)
    std::cout << "\nStep 3: Kernel Selection" << std::endl;

    auto problem = create_grouped_conv2d_problem(1, 64, 128, 14, 14, 3, 3, 1, 1);

    auto* selected = dispatcher.select_kernel(problem);
    std::cout << "  Selected: " << (selected ? selected->name() : "none") << std::endl;

    // Step 4: GPU execution
    std::cout << "\nStep 4: GPU Execution" << std::endl;

    ck_tile::conv::ConvParam cp{
        2,
        static_cast<ck_tile::index_t>(1),
        static_cast<ck_tile::index_t>(1),
        static_cast<ck_tile::index_t>(128),
        static_cast<ck_tile::index_t>(64),
        {static_cast<ck_tile::index_t>(3), static_cast<ck_tile::index_t>(3)},
        {static_cast<ck_tile::index_t>(14), static_cast<ck_tile::index_t>(14)},
        {1, 1},
        {1, 1},
        {1, 1},
        {1, 1}};

    using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;
    using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;

    std::cout << "  Creating tensors..." << std::endl;
    auto in_d  = ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(cp);
    auto wei_d = ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(cp);
    auto out_d = ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(cp);

    ck_tile::HostTensor<InDataType> input(in_d);
    ck_tile::HostTensor<WeiDataType> weight(wei_d);
    ck_tile::HostTensor<OutDataType> output(out_d);
    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);

    std::cout << "  Allocating device memory..." << std::endl;
    ck_tile::DeviceMem in_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_dev(output.get_element_space_size_in_bytes());
    in_dev.ToDevice(input.data());
    wei_dev.ToDevice(weight.data());

    std::cout << "  Launching kernel..." << std::endl;
    float time_ms = dispatcher.run(in_dev.GetDeviceBuffer(),
                                   wei_dev.GetDeviceBuffer(),
                                   out_dev.GetDeviceBuffer(),
                                   problem,
                                   nullptr);

    std::cout << "  Reading back..." << std::endl;
    out_dev.FromDevice(output.data());
    size_t nz = 0;
    for(size_t i = 0; i < output.get_element_space_size(); ++i)
        if(static_cast<float>(output.data()[i]) != 0.0f)
            ++nz;

    std::cout << "  Time:    " << std::fixed << std::setprecision(4) << time_ms << " ms"
              << std::endl;
    std::cout << "  TFLOPS:  " << std::setprecision(2) << calculate_conv_tflops(problem, time_ms)
              << std::endl;
    std::cout << "  NonZero: " << nz << "/" << output.get_element_space_size() << std::endl;

    // Step 5: JSON export
    std::cout << "\nStep 5: JSON Export" << std::endl;
    std::string json = registry.export_json(false);
    std::cout << "  JSON size: " << json.size() << " bytes" << std::endl;

    bool passed = nz > 0;
    utils::print_separator();
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";
    utils::print_separator();

    return passed ? 0 : 1;
}
