// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/builder/types.hpp"
#include "utils/ckb_conv_test_configs.hpp"
#include "ck/library/utility/device_memory.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace {

using namespace ck_tile::builder;
using namespace ck_tile::builder::test_utils;

TEST(ReferenceValidation, Can_Call_Reference_Run_Method)
{
    // Test: Can we call Run() method on reference kernel?

    constexpr ConvSignature sig{.spatial_dim           = 2,
                                .direction             = ConvDirection::FORWARD,
                                .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
                                .data_type             = DataType::FP16,
                                .elementwise_operation = ElementwiseOperation::PASS_THROUGH};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    // Simple dimensions
    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    // Allocate minimal device memory (just to test API)
    const size_t in_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_dev(out_size);

    in_dev.SetZero();
    wei_dev.SetZero();
    out_dev.SetZero();

    // Prepare parameters for Run()
    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    // Test: Can we call Run()?
    RefKernel ref_kernel;

    // This should compile and execute:
    ref_kernel.Run(reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
                   reinterpret_cast<ck::half_t*>(out_dev.GetDeviceBuffer()),
                   G,
                   N,
                   K,
                   C,
                   input_spatial,
                   filter_spatial,
                   output_spatial,
                   strides,
                   dilations,
                   left_pads);

    // If we get here, Run() worked!
    std::cout << "✓ Reference Run() method callable through Builder!" << std::endl;
    EXPECT_TRUE(true); // Test passes if no crash
}

TEST(ReferenceValidation, Compare_Reference_vs_Optimized_Forward_2D_FP16)
{
    // Test: Can we compare reference vs optimized kernel?

    // Define problem
    constexpr ConvSignature sig{.spatial_dim           = 2,
                                .direction             = ConvDirection::FORWARD,
                                .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
                                .data_type             = DataType::FP16,
                                .elementwise_operation = ElementwiseOperation::PASS_THROUGH};

    // Reference algorithm
    constexpr auto ref_alg = ConvAlgorithm_Reference{};

    // Optimized algorithm (simple XDL config)
    constexpr auto opt_alg =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::Default)
            .with_block_gemm(BlockGemmDesc_v3_intrawave);

    using RefKernel = ConvBuilder<sig, ref_alg>::Instance;
    using OptKernel = ConvBuilder<sig, opt_alg>::Instance;

    // For now, just test both can instantiate
    RefKernel ref;
    OptKernel opt;

    std::cout << "Reference type: " << ref.GetTypeString() << std::endl;
    std::cout << "Optimized type: " << opt.GetTypeString() << std::endl;

    // Both types exist!
    EXPECT_TRUE(true);

    std::cout << "✓ Can instantiate both Reference and Optimized through Builder!" << std::endl;
}

} // namespace
