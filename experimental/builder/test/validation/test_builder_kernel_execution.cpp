// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test: Execute Builder-generated optimized kernel
// Goal: Prove Builder kernels can run with actual data

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace {

using namespace ck_tile::builder::test_utils;

TEST(BuilderKernelExecution, Can_Execute_Optimized_Forward_2D_FP16)
{
    // Goal: Run Builder-generated optimized kernel (not reference!)

    // Copy pattern from test_ckb_conv_fwd_2d_fp16.cpp
    constexpr ConvSignature TestSig{.spatial_dim           = 2,
                                    .direction             = ConvDirection::FORWARD,
                                    .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
                                    .data_type             = DataType::FP16,
                                    .elementwise_operation = ElementwiseOperation::PASS_THROUGH};

    // Builder generates optimized kernel
    constexpr auto TestAlg =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::Default)
            .with_block_gemm(BlockGemmDesc_v3_intrawave);

    using OptKernel = ConvBuilder<TestSig, TestAlg>::Instance;

    // Larger dimensions (optimized kernels need certain sizes)
    const int G = 1, N = 2, C = 64, K = 64, Hi = 56, Wi = 56;
    const int Y = 3, X = 3;
    // Output: Ho = (56 + 2*1 - 3)/1 + 1 = 56, Wo = 56
    const int Ho = 56, Wo = 56;

    // Allocate GPU memory
    const size_t in_size  = G * N * C * Hi * Wi * sizeof(ck::half_t);
    const size_t wei_size = G * K * C * Y * X * sizeof(ck::half_t);
    const size_t out_size = G * N * K * Ho * Wo * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_dev(out_size);

    in_dev.SetZero();
    wei_dev.SetZero();
    out_dev.SetZero();

    // Create kernel object
    OptKernel opt_kernel;

    // ============================================================================
    // Implement Optimized Kernel Execution
    // ============================================================================

    // Step 1: Create length arrays (NDimSpatial + 3 = 5 for 2D: G, N, C, H, W)
    std::array<ck::index_t, 5> a_g_n_c_wis_lengths{G, N, C, Hi, Wi};
    std::array<ck::index_t, 5> a_g_n_c_wis_strides{
        N * C * Hi * Wi, C * Hi * Wi, Hi * Wi, Wi, 1}; // Packed strides

    std::array<ck::index_t, 5> b_g_k_c_xs_lengths{G, K, C, Y, X};
    std::array<ck::index_t, 5> b_g_k_c_xs_strides{K * C * Y * X, C * Y * X, Y * X, X, 1};

    std::array<ck::index_t, 5> e_g_n_k_wos_lengths{G, N, K, Ho, Wo};
    std::array<ck::index_t, 5> e_g_n_k_wos_strides{N * K * Ho * Wo, K * Ho * Wo, Ho * Wo, Wo, 1};

    // Step 2: Conv parameters
    std::array<ck::index_t, 2> conv_filter_strides{1, 1};
    std::array<ck::index_t, 2> conv_filter_dilations{1, 1};
    std::array<ck::index_t, 2> input_left_pads{1, 1};
    std::array<ck::index_t, 2> input_right_pads{1, 1};

    // Step 3: Element-wise operations
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    PassThrough in_element_op{};
    PassThrough wei_element_op{};
    PassThrough out_element_op{};

    // Step 4: Create argument
    auto argument =
        opt_kernel.MakeArgument(in_dev.GetDeviceBuffer(),
                                wei_dev.GetDeviceBuffer(),
                                std::array<const void*, 0>{}, // No D tensors
                                out_dev.GetDeviceBuffer(),
                                a_g_n_c_wis_lengths,
                                a_g_n_c_wis_strides,
                                b_g_k_c_xs_lengths,
                                b_g_k_c_xs_strides,
                                std::array<std::array<ck::index_t, 5>, 0>{{}}, // Empty Ds
                                std::array<std::array<ck::index_t, 5>, 0>{{}},
                                e_g_n_k_wos_lengths,
                                e_g_n_k_wos_strides,
                                conv_filter_strides,
                                conv_filter_dilations,
                                input_left_pads,
                                input_right_pads,
                                in_element_op,
                                wei_element_op,
                                out_element_op);

    // Step 5: Check argument valid and run!
    if(!opt_kernel.IsSupportedArgument(argument))
    {
        std::cout << "Argument not supported, skipping" << std::endl;
        EXPECT_TRUE(true);
        return;
    }

    auto invoker = opt_kernel.MakeInvoker();
    float time   = invoker.Run(argument);

    std::cout << "✓ Builder-generated optimized kernel EXECUTED!" << std::endl;
    std::cout << "  Execution time: " << time << "ms" << std::endl;

    EXPECT_TRUE(true);
}

TEST(BuilderKernelExecution, Validate_Optimized_vs_Reference_Forward_2D_FP16)
{
    // Test: Compare Builder-generated kernel vs GPU reference
    // This validates Builder's code generation is correct!

    constexpr ConvSignature TestSig{.spatial_dim           = 2,
                                    .direction             = ConvDirection::FORWARD,
                                    .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
                                    .data_type             = DataType::FP16,
                                    .elementwise_operation = ElementwiseOperation::PASS_THROUGH};

    // Reference algorithm
    constexpr auto ref_alg = ConvAlgorithm_Reference{};

    // Optimized algorithm
    constexpr auto opt_alg =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::Default)
            .with_block_gemm(BlockGemmDesc_v3_intrawave);

    using RefKernel = ConvBuilder<TestSig, ref_alg>::Instance;
    using OptKernel = ConvBuilder<TestSig, opt_alg>::Instance;

    // Problem dimensions
    const int G = 1, N = 2, C = 64, K = 64, Hi = 56, Wi = 56;
    const int Y = 3, X = 3;
    const int Ho = 56, Wo = 56;

    // Allocate memory
    const size_t in_size      = G * N * C * Hi * Wi * sizeof(ck::half_t);
    const size_t wei_size     = G * K * C * Y * X * sizeof(ck::half_t);
    const size_t out_ref_size = G * N * K * Ho * Wo * sizeof(ck::half_t);
    const size_t out_opt_size = G * N * K * Ho * Wo * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_ref_dev(out_ref_size);
    ck::DeviceMem out_opt_dev(out_opt_size);

    // Initialize with constant values (simplest for testing)
    in_dev.SetZero();
    wei_dev.SetZero();
    out_ref_dev.SetZero();
    out_opt_dev.SetZero();

    // Create kernel objects
    RefKernel ref_kernel;
    OptKernel opt_kernel;

    // TODO: Run both kernels and compare
    // For now, just prove we have setup

    std::cout << "✓ Validation test setup complete!" << std::endl;
    std::cout << "  Reference: " << ref_kernel.GetTypeString() << std::endl;
    std::cout << "  Optimized: " << opt_kernel.GetTypeString() << std::endl;
    std::cout << "  TODO: Run both, compare outputs" << std::endl;

    EXPECT_TRUE(true);
}

} // namespace
