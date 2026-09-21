// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Example 02: All Convolution Directions
//
// Forward, backward-data, and backward-weight for 2D convolution,
// each executed on GPU with non-zero verification.
//
// Build: cd dispatcher/build && cmake .. && make grouped_conv_02_all_dirs

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

DECL_GROUPED_CONV_KERNEL_SET(
    conv_fwd_2d,
    .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv4").vector_sizes(4, 8, 8),
         "gfx950"));

DECL_GROUPED_CONV_KERNEL_SET(
    conv_bwdd_2d,
    .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("bwd_data").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv3").vector_sizes(4, 8, 8),
         "gfx950"));

DECL_GROUPED_CONV_KERNEL_SET(
    conv_bwdw_2d,
    .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("bwd_weight").dims(2),
         GroupedConvAlgo()
             .tile(1, 128, 128)
             .pipeline("compv3")
             .memory_op("atomic_add")
             .vector_sizes(4, 8, 8),
         "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 02: All Convolution Directions",
                            "Forward/BwdData/BwdWeight with GPU execution and verification");
    args.add_option("--arch", "gfx950", "GPU architecture");

    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 02: All Convolution Directions");

    std::string gfx_arch = args.get("--arch", "gfx950");

    GroupedConvRegistry registry;
    registry.set_name("all_directions");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    GroupedConvDispatcher dispatcher(&registry);

    const int N = 1, G = 1, C = 64, K = 128, Hi = 14, Wi = 14, Y = 3, X = 3;

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
    ck_tile::HostTensor<OutDataType> output(out_desc);

    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());

    std::cout << "\n  " << std::left << std::setw(12) << "Direction" << std::right << std::setw(10)
              << "Time(ms)" << std::setw(10) << "TFLOPS" << std::setw(14) << "NonZero"
              << std::setw(10) << "Status" << "\n";
    std::cout << "  " << std::string(56, '-') << "\n";

    bool all_pass = true;

    auto print_result =
        [](const char* label, float time_ms, double tflops, size_t nz, size_t total, bool ok) {
            std::cout << "  " << std::left << std::setw(12) << label << std::right << std::fixed
                      << std::setprecision(4) << std::setw(10) << time_ms << std::setprecision(2)
                      << std::setw(10) << tflops << std::setw(14)
                      << (std::to_string(nz) + "/" + std::to_string(total)) << std::setw(10)
                      << (ok ? "OK" : "FAIL") << "\n";
        };

    // Forward: run(X, W, Y)
    {
        auto problem =
            create_grouped_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1, GroupedConvOp::Forward);
        float time_ms = dispatcher.run(input_dev.GetDeviceBuffer(),
                                       weight_dev.GetDeviceBuffer(),
                                       output_dev.GetDeviceBuffer(),
                                       problem,
                                       nullptr);
        output_dev.FromDevice(output.data());
        size_t nz = 0;
        for(size_t i = 0; i < output.get_element_space_size(); ++i)
            if(static_cast<float>(output.data()[i]) != 0.0f)
                ++nz;
        bool ok = nz > 0;
        print_result("forward",
                     time_ms,
                     calculate_conv_tflops(problem, time_ms),
                     nz,
                     output.get_element_space_size(),
                     ok);
        if(!ok)
            all_pass = false;
    }

    // Backward Data: run(dY, W, dX)
    {
        auto problem =
            create_grouped_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1, GroupedConvOp::BackwardData);
        ck_tile::HostTensor<InDataType> dx_host(in_desc);
        ck_tile::DeviceMem dx_dev(dx_host.get_element_space_size_in_bytes());
        float time_ms = dispatcher.run(output_dev.GetDeviceBuffer(), // dY (from forward pass)
                                       weight_dev.GetDeviceBuffer(), // W
                                       dx_dev.GetDeviceBuffer(),     // dX (output)
                                       problem,
                                       nullptr);
        dx_dev.FromDevice(dx_host.data());
        size_t nz = 0;
        for(size_t i = 0; i < dx_host.get_element_space_size(); ++i)
            if(static_cast<float>(dx_host.data()[i]) != 0.0f)
                ++nz;
        bool ok = nz > 0;
        print_result("bwd_data",
                     time_ms,
                     calculate_conv_tflops(problem, time_ms),
                     nz,
                     dx_host.get_element_space_size(),
                     ok);
        if(!ok)
            all_pass = false;
    }

    // Backward Weight: run(X, dY, dW)
    {
        auto problem = create_grouped_conv2d_problem(
            N, C, K, Hi, Wi, Y, X, 1, 1, GroupedConvOp::BackwardWeight);
        ck_tile::HostTensor<WeiDataType> dw_host(wei_desc);
        ck_tile::DeviceMem dw_dev(dw_host.get_element_space_size_in_bytes());
        float time_ms = dispatcher.run(input_dev.GetDeviceBuffer(),  // X
                                       output_dev.GetDeviceBuffer(), // dY
                                       dw_dev.GetDeviceBuffer(),     // dW (output)
                                       problem,
                                       nullptr);
        dw_dev.FromDevice(dw_host.data());
        size_t nz = 0;
        for(size_t i = 0; i < dw_host.get_element_space_size(); ++i)
            if(static_cast<float>(dw_host.data()[i]) != 0.0f)
                ++nz;
        bool ok = nz > 0;
        print_result("bwd_weight",
                     time_ms,
                     calculate_conv_tflops(problem, time_ms),
                     nz,
                     dw_host.get_element_space_size(),
                     ok);
        if(!ok)
            all_pass = false;
    }

    utils::print_separator();
    std::cout << "  Status: " << (all_pass ? "PASS" : "FAIL") << "\n";
    utils::print_separator();

    return all_pass ? 0 : 1;
}
