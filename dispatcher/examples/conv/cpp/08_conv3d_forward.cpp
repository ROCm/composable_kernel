// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 08: 3D Convolution Forward
 *
 * Demonstrates 3D convolution (for video or volumetric data).
 *
 * Build: cd dispatcher/build && cmake .. && make conv_08_conv3d
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL DECLARATIONS - 3D Forward
// =============================================================================

DECL_CONV_KERNEL_SET(conv3d_fwd_kernels,
                     .add(ConvSig().dtype("fp16").layout("ndhwgc").conv_type("forward").dims(3),
                          ConvAlgo()
                              .tile(1, 16, 64)
                              .wave(1, 4, 1)
                              .warp(16, 16, 32)
                              .pipeline("compv3")
                              .scheduler("intrawave")
                              .vector_sizes(4, 8, 8)
                              .block_per_cu(1),
                          "gfx942"));

// =============================================================================
// DATA TYPES
// =============================================================================

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 08: Conv3D Forward", "3D convolution for video/volumetric data");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "32", "Input channels C");
    args.add_option("-k", "64", "Output channels K");
    args.add_option("--depth", "8", "Depth D");
    args.add_option("--size", "16", "Spatial size (H=W)");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 08: 3D Convolution Forward\n";
    std::cout << "======================================================================\n\n";

    // Show declared kernels
    std::cout << "Step 1: Declared Kernels\n";
    std::cout << "------------------------\n";
    ConvKernelSetRegistry::instance().print();
    std::cout << "\n";

    // Problem setup
    int N  = args.get_int("-n", 1);
    int C  = args.get_int("-c", 32);
    int K  = args.get_int("-k", 64);
    int Di = args.get_int("--depth", 8);
    int Hi = args.get_int("--size", 16);
    int Wi = Hi;
    int Z = 3, Y = 3, X = 3;

    std::cout << "Step 2: Problem\n";
    std::cout << "---------------\n";
    std::cout << "  Input:  N=" << N << ", D=" << Di << ", H=" << Hi << ", W=" << Wi << ", C=" << C
              << "\n";
    std::cout << "  Filter: Z=" << Z << ", Y=" << Y << ", X=" << X << ", K=" << K << "\n\n";

    // Create 3D conv param
    ck_tile::conv::ConvParam conv_param{3,
                                        1,
                                        static_cast<ck_tile::index_t>(N),
                                        static_cast<ck_tile::index_t>(K),
                                        static_cast<ck_tile::index_t>(C),
                                        {static_cast<ck_tile::index_t>(Z),
                                         static_cast<ck_tile::index_t>(Y),
                                         static_cast<ck_tile::index_t>(X)},
                                        {static_cast<ck_tile::index_t>(Di),
                                         static_cast<ck_tile::index_t>(Hi),
                                         static_cast<ck_tile::index_t>(Wi)},
                                        {1, 1, 1},
                                        {1, 1, 1},
                                        {1, 1, 1},
                                        {1, 1, 1}};

    using InLayout  = ck_tile::tensor_layout::convolution::NDHWGC;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKZYXC;
    using OutLayout = ck_tile::tensor_layout::convolution::NDHWGK;

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
    output.SetZero();

    std::cout << "Step 3: GPU Execution\n";
    std::cout << "---------------------\n";

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());
    output_dev.SetZero();

    ck_tile::GroupedConvFwdHostArgs<> kernel_args(conv_param,
                                                  input_dev.GetDeviceBuffer(),
                                                  weight_dev.GetDeviceBuffer(),
                                                  {},
                                                  output_dev.GetDeviceBuffer(),
                                                  1);

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 3, 10};

    using Launcher   = generated::FirstKernelLauncher;
    float elapsed_ms = Launcher::launch(kernel_args, stream_cfg);

    double flops  = 2.0 * N * K * C * Z * Y * X * Di * Hi * Wi;
    double tflops = flops / (elapsed_ms * 1e9);

    // Basic output check
    output_dev.FromDevice(output.data());
    size_t non_zero = 0;
    for(size_t i = 0; i < output.get_element_space_size(); ++i)
        if(std::abs(static_cast<float>(output.data()[i])) > 1e-6f)
            ++non_zero;
    bool passed = (non_zero > 0);

    std::cout << "  Input:  " << input.get_element_space_size() << " elements\n";
    std::cout << "  Weight: " << weight.get_element_space_size() << " elements\n";
    std::cout << "  Output: " << output.get_element_space_size() << " elements\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";

    std::cout << "\n======================================================================\n";
    std::cout << "3D Convolution: Used for video, medical imaging, volumetric data\n";
    std::cout << "======================================================================\n";

    return passed ? 0 : 1;
}
