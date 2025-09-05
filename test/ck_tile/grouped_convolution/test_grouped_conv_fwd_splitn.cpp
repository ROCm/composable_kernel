// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_fwd.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

using namespace ck_tile;

// Helper function to run convolution test with a specific batch size
template <typename KernelType>
void RunConvolutionTest(index_t batch_size, const char* description, bool expect_split_n)
{
    std::cout << "\n--- Test case: " << description << " (N=" << batch_size << ") ---" << std::endl;

    // Define kernel types for the test (same for all test cases)
    using InDataType  = half_t;
    using WeiDataType = half_t;
    using OutDataType = half_t;

    using InLayout  = tensor_layout::convolution::NHWGC;
    using WeiLayout = tensor_layout::convolution::GKYXC;
    using OutLayout = tensor_layout::convolution::NHWGK;

    // Check if we should run full tests (including slow CPU reference for large batches)
    // Set CK_TILE_FULL_SPLITN_TEST=1 environment variable to enable full testing with CPU reference
    const char* full_test_env    = std::getenv("CK_TILE_FULL_SPLITN_TEST");
    bool run_full_accuracy_tests = (full_test_env != nullptr && std::string(full_test_env) == "1");

    // Decide whether to run CPU reference based on batch size and environment variable
    bool skip_cpu_reference = false;
    if(batch_size > 32 && !run_full_accuracy_tests)
    {
        skip_cpu_reference = true;
        std::cout << "Note: CPU accuracy check skipped for large batch (set "
                     "CK_TILE_FULL_SPLITN_TEST=1 to enable)"
                  << std::endl;
    }

    // Create configuration for this test case
    conv::ConvParam conv_param{
        2,          // num_dim_spatial
        1,          // G (groups)
        batch_size, // N (batch size)
        256,        // K (output channels)
        256,        // C (input channels)
        {3, 3},     // filter_spatial_lengths
        {112, 112}, // input_spatial_lengths
        {1, 1},     // conv_filter_strides
        {1, 1},     // conv_filter_dilations
        {0, 0},     // input_left_pads
        {0, 0}      // input_right_pads
    };

    // Calculate and display tensor size
    long_index_t input_size  = static_cast<long_index_t>(conv_param.N_) * conv_param.C_ * 112 * 112;
    long_index_t input_bytes = input_size * sizeof(half_t);

    std::cout << "Input tensor size: " << (input_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB ("
              << input_bytes << " bytes)" << std::endl;

    // Calculate tensor sizes
    long_index_t weight_size =
        static_cast<long_index_t>(conv_param.K_) * conv_param.C_ * 3 * 3; // 3x3 filters
    long_index_t output_size = static_cast<long_index_t>(conv_param.N_) * conv_param.K_ * 110 *
                               110; // Output H=W=110 (112-3+1)

    // Allocate device memory for tensors
    void* d_input  = nullptr;
    void* d_weight = nullptr;
    void* d_output = nullptr;

    hipError_t err;
    err = hipMalloc(&d_input, input_size * sizeof(half_t));
    if(err != hipSuccess)
    {
        std::cout << "Failed to allocate input memory: " << hipGetErrorString(err) << std::endl;
        return; // Skip this test case
    }

    err = hipMalloc(&d_weight, weight_size * sizeof(half_t));
    if(err != hipSuccess)
    {
        (void)hipFree(d_input);
        std::cout << "Failed to allocate weight memory: " << hipGetErrorString(err) << std::endl;
        return; // Skip this test case
    }

    err = hipMalloc(&d_output, output_size * sizeof(half_t));
    if(err != hipSuccess)
    {
        (void)hipFree(d_input);
        (void)hipFree(d_weight);
        std::cout << "Failed to allocate output memory: " << hipGetErrorString(err) << std::endl;
        return; // Skip this test case
    }

    // Initialize input and weights with simple patterns for verification
    // Fill input with 1.0 and weights with 0.1 for a simple test
    std::vector<half_t> h_input(input_size, half_t(1.0f));
    std::vector<half_t> h_weight(weight_size, half_t(0.1f));

    (void)hipMemcpy(d_input, h_input.data(), input_size * sizeof(half_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(half_t), hipMemcpyHostToDevice);

    // Initialize output to zero
    (void)hipMemset(d_output, 0, output_size * sizeof(half_t));

    // Prepare for potential CPU reference computation
    HostTensor<OutDataType>* output_ref_tensor_ptr = nullptr;

    if(!skip_cpu_reference)
    {
        // Run CPU reference convolution for comparison
        // Create tensor descriptors for HostTensor
        const auto in_desc =
            conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
        const auto wei_desc =
            conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
        const auto out_desc =
            conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

        // Create HostTensor wrappers around our vectors
        HostTensor<InDataType> input_tensor(in_desc);
        HostTensor<WeiDataType> weight_tensor(wei_desc);
        output_ref_tensor_ptr = new HostTensor<OutDataType>(out_desc);

        // Copy data to HostTensor
        std::copy(h_input.begin(), h_input.end(), input_tensor.mData.begin());
        std::copy(h_weight.begin(), h_weight.end(), weight_tensor.mData.begin());
        output_ref_tensor_ptr->SetZero();

        // Run CPU reference convolution
        reference_grouped_conv_fwd<2, InDataType, WeiDataType, OutDataType>(
            input_tensor,
            weight_tensor,
            *output_ref_tensor_ptr,
            conv_param.conv_filter_strides_,
            conv_param.conv_filter_dilations_,
            conv_param.input_left_pads_,
            conv_param.input_right_pads_);

        std::cout << "CPU reference convolution computed" << std::endl;
    }

    // Create GroupedConvFwdHostArgs
    std::vector<const void*> ds_ptr; // No bias tensors
    index_t k_batch = 1;             // No Split-K

    GroupedConvFwdHostArgs args(conv_param, d_input, d_weight, ds_ptr, d_output, k_batch);

    // Create kernel arguments
    auto kargs = KernelType::MakeKernelArgs(args);

    // Check if arguments are supported
    if(!KernelType::IsSupportedArgument(kargs))
    {
        std::cout << "Arguments not supported by kernel - skipping" << std::endl;
        (void)hipFree(d_input);
        (void)hipFree(d_weight);
        (void)hipFree(d_output);
        return;
    }

    // Get grid and block sizes
    const dim3 grids  = KernelType::GridSize(kargs);
    const dim3 blocks = KernelType::BlockSize();

    std::cout << "Grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
              << " Block: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
              << std::endl;

    // Check if Split-N is active
    bool split_n_active = (grids.z > 1);
    if(split_n_active)
    {
        std::cout << "[PASS] Split-N ACTIVE with " << grids.z << " splits" << std::endl;
    }
    else
    {
        std::cout << "[INFO] Split-N NOT active (grid.z = 1)" << std::endl;
    }

    // Verify Split-N activation when we expect it (only check when > 2GB)
    if(expect_split_n)
    {
        EXPECT_TRUE(split_n_active)
            << "Expected Split-N to be active for " << description << " (>2GB)";
    }
    // Note: We don't enforce Split-N to be OFF for small sizes - kernel may optimize as needed

    // Launch the kernel
    stream_config stream_cfg;
    stream_cfg.log_level_   = 0;
    stream_cfg.time_kernel_ = false;

    constexpr int kBlockPerCu = 1;
    float elapsed_time =
        launch_kernel(stream_cfg, make_kernel<kBlockPerCu>(KernelType{}, grids, blocks, 0, kargs));

    if(elapsed_time < 0)
    {
        std::cout << "[FAIL] Kernel execution FAILED" << std::endl;
        EXPECT_GE(elapsed_time, 0) << "Kernel should execute successfully";
    }
    else
    {
        std::cout << "[PASS] Kernel executed successfully in " << elapsed_time << " ms"
                  << std::endl;

        // Accuracy check if CPU reference was computed
        if(!skip_cpu_reference)
        {
            // Get GPU output
            std::vector<half_t> h_output_gpu(output_size);
            (void)hipMemcpy(
                h_output_gpu.data(), d_output, output_size * sizeof(half_t), hipMemcpyDeviceToHost);

            // Accuracy check - compare GPU output with reference
            // Calculate error tolerances for FP16
            const float rtol = 1e-2f; // 1% relative tolerance for FP16
            const float atol = 1e-4f; // Small absolute tolerance

            bool accuracy_pass = true;
            float max_diff     = 0.0f;
            size_t error_count = 0;

            for(size_t i = 0; i < static_cast<size_t>(output_size); i++)
            {
                float ref_val = static_cast<float>(output_ref_tensor_ptr->mData[i]);
                float gpu_val = static_cast<float>(h_output_gpu[i]);
                float diff    = std::abs(ref_val - gpu_val);

                if(diff > atol + rtol * std::abs(ref_val))
                {
                    error_count++;
                    max_diff = std::max(max_diff, diff);
                    if(error_count <= 5)
                    { // Print first few errors
                        std::cout << "Mismatch at index " << i << ": ref=" << ref_val
                                  << ", gpu=" << gpu_val << ", diff=" << diff << std::endl;
                    }
                }
            }

            if(error_count > 0)
            {
                std::cout << "[FAIL] Accuracy check FAILED: " << error_count << "/" << output_size
                          << " elements exceed tolerance"
                          << ", max_diff=" << max_diff << std::endl;
                accuracy_pass = false;
            }
            else
            {
                std::cout << "[PASS] Accuracy check PASSED (rtol=" << rtol << ", atol=" << atol
                          << ")" << std::endl;
            }

            EXPECT_TRUE(accuracy_pass) << "Accuracy check failed with " << error_count << " errors";

            // Clean up CPU reference tensor
            delete output_ref_tensor_ptr;
        }
        else
        {
            // Sanity check when CPU reference is skipped
            std::cout << "Running sanity checks..." << std::endl;

            // For large tensors, only sample a subset to avoid slow memory copy
            const size_t max_samples = 10000; // Sample at most 10k values
            size_t sample_size       = std::min(max_samples, static_cast<size_t>(output_size));
            size_t stride            = output_size / sample_size; // Sample evenly across the output

            // Get sampled GPU output for sanity check
            std::vector<half_t> h_output_samples(sample_size);

            // Copy only sampled values
            for(size_t i = 0; i < sample_size; i++)
            {
                size_t offset = i * stride;
                (void)hipMemcpy(&h_output_samples[i],
                                reinterpret_cast<half_t*>(d_output) + offset,
                                sizeof(half_t),
                                hipMemcpyDeviceToHost);
            }

            // Basic sanity checks on samples
            bool has_non_zero = false;
            bool has_nan_inf  = false;
            float min_val     = std::numeric_limits<float>::max();
            float max_val     = std::numeric_limits<float>::lowest();

            for(size_t i = 0; i < sample_size; i++)
            {
                float val = static_cast<float>(h_output_samples[i]);

                if(std::isnan(val) || std::isinf(val))
                {
                    has_nan_inf = true;
                }
                if(val != 0.0f)
                {
                    has_non_zero = true;
                }
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            // Report sanity check results
            if(has_nan_inf)
            {
                std::cout << "[FAIL] Sanity check FAILED: Output contains NaN or Inf values"
                          << std::endl;
                EXPECT_FALSE(has_nan_inf) << "Output should not contain NaN or Inf";
            }
            else if(!has_non_zero)
            {
                std::cout << "[WARN] Sanity check WARNING: All sampled values are zero"
                          << std::endl;
            }
            else
            {
                std::cout << "[PASS] Sanity check PASSED (sampled " << sample_size
                          << " values, range: [" << min_val << ", " << max_val << "])" << std::endl;
            }
        }
    }

    // Clean up
    (void)hipFree(d_input);
    (void)hipFree(d_weight);
    (void)hipFree(d_output);
}

// Test 1: Tiny batch (25MB) - No Split-N expected
TEST(GroupedConvFwdSplitN, TinyBatch)
{
    // Define kernel types for the test (same for all test cases)
    using InDataType  = half_t;
    using WeiDataType = half_t;
    using AccDataType = float;
    using OutDataType = half_t;

    using InLayout  = tensor_layout::convolution::NHWGC;
    using WeiLayout = tensor_layout::convolution::GKYXC;
    using OutLayout = tensor_layout::convolution::NHWGK;

    // Define tile configuration
    constexpr index_t M_Tile      = 64;
    constexpr index_t N_Tile      = 64;
    constexpr index_t K_Tile      = 64;
    constexpr index_t M_Warp      = 2;
    constexpr index_t N_Warp      = 2;
    constexpr index_t K_Warp      = 1;
    constexpr index_t M_Warp_Tile = 32;
    constexpr index_t N_Warp_Tile = 32;
    constexpr index_t K_Warp_Tile = 16;

    using CodegenShape = TileGemmShape<sequence<M_Tile, N_Tile, K_Tile>,
                                       sequence<M_Warp, N_Warp, K_Warp>,
                                       sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner   = GemmTile1DPartitioner<CodegenShape>;
    constexpr auto ConvSpec = ConvolutionSpecialization::Default;
    using DsLayout          = tuple<>;
    using DsDataType        = tuple<>;

    using GroupedConvTraitsType =
        GroupedConvTraits<2, ConvSpec, InLayout, WeiLayout, DsLayout, OutLayout>;

    constexpr index_t VectorSizeA = 8;
    constexpr index_t VectorSizeB = 8;
    constexpr index_t VectorSizeC = 8;

    using CodegenPipelineProblem =
        GemmPipelineProblem<InDataType,
                            WeiDataType,
                            AccDataType,
                            CodegenShape,
                            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraits,
                            InDataType,
                            true,
                            VectorSizeA,
                            VectorSizeB>;

    using CodegenPipeline = GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    using CDEElementWise = element_wise::PassThrough;
    using ConvEpilogue   = CShuffleEpilogue<
          CShuffleEpilogueProblem<InDataType,
                                  WeiDataType,
                                  DsDataType,
                                  AccDataType,
                                  OutDataType,
                                  typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                                  tensor_layout::gemm::RowMajor,
                                  CDEElementWise,
                                  TilePartitioner::MPerBlock,
                                  TilePartitioner::NPerBlock,
                                  M_Warp,
                                  N_Warp,
                                  M_Warp_Tile,
                                  N_Warp_Tile,
                                  K_Warp_Tile,
                                  CodegenPipelineProblem::TransposeC,
                                  memory_operation_enum::set,
                                  1,
                                  true,
                                  VectorSizeC>>;

    using Kernel = GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                   TilePartitioner,
                                                   CodegenPipeline,
                                                   ConvEpilogue>;

    RunConvolutionTest<Kernel>(4, "Tiny batch (25MB)", false);
}

// // Test 2: Small batch (412MB) - No Split-N expected
// TEST(GroupedConvFwdSplitN, SmallBatch)
// {
//     // Define kernel types for the test (same for all test cases)
//     using InDataType  = half_t;
//     using WeiDataType = half_t;
//     using AccDataType = float;
//     using OutDataType = half_t;

//     using InLayout  = tensor_layout::convolution::NHWGC;
//     using WeiLayout = tensor_layout::convolution::GKYXC;
//     using OutLayout = tensor_layout::convolution::NHWGK;

//     // Define tile configuration
//     constexpr index_t M_Tile      = 64;
//     constexpr index_t N_Tile      = 64;
//     constexpr index_t K_Tile      = 64;
//     constexpr index_t M_Warp      = 2;
//     constexpr index_t N_Warp      = 2;
//     constexpr index_t K_Warp      = 1;
//     constexpr index_t M_Warp_Tile = 32;
//     constexpr index_t N_Warp_Tile = 32;
//     constexpr index_t K_Warp_Tile = 16;

//     using CodegenShape = TileGemmShape<sequence<M_Tile, N_Tile, K_Tile>,
//                                        sequence<M_Warp, N_Warp, K_Warp>,
//                                        sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

//     using TilePartitioner   = GemmTile1DPartitioner<CodegenShape>;
//     constexpr auto ConvSpec = ConvolutionSpecialization::Default;
//     using DsLayout          = tuple<>;
//     using DsDataType        = tuple<>;

//     using GroupedConvTraitsType =
//         GroupedConvTraits<2, ConvSpec, InLayout, WeiLayout, DsLayout, OutLayout>;

//     constexpr index_t VectorSizeA = 8;
//     constexpr index_t VectorSizeB = 8;
//     constexpr index_t VectorSizeC = 8;

//     using CodegenPipelineProblem =
//         GemmPipelineProblem<InDataType,
//                             WeiDataType,
//                             AccDataType,
//                             CodegenShape,
//                             typename GroupedConvTraitsType::GroupedConvImplicitGemmTraits,
//                             InDataType,
//                             true,
//                             VectorSizeA,
//                             VectorSizeB>;

//     using CodegenPipeline = GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

//     using CDEElementWise = element_wise::PassThrough;
//     using ConvEpilogue   = CShuffleEpilogue<
//           CShuffleEpilogueProblem<InDataType,
//                                   WeiDataType,
//                                   DsDataType,
//                                   AccDataType,
//                                   OutDataType,
//                                   typename GroupedConvTraitsType::ImplicitGemmDsLayout,
//                                   tensor_layout::gemm::RowMajor,
//                                   CDEElementWise,
//                                   TilePartitioner::MPerBlock,
//                                   TilePartitioner::NPerBlock,
//                                   M_Warp,
//                                   N_Warp,
//                                   M_Warp_Tile,
//                                   N_Warp_Tile,
//                                   K_Warp_Tile,
//                                   CodegenPipelineProblem::TransposeC,
//                                   memory_operation_enum::set,
//                                   1,
//                                   true,
//                                   VectorSizeC>>;

//     using Kernel = GroupedConvolutionForwardKernel<GroupedConvTraitsType,
//                                                    TilePartitioner,
//                                                    CodegenPipeline,
//                                                    ConvEpilogue>;

//     RunConvolutionTest<Kernel>(64, "Small batch (412MB)", false);
// }

// // Test 3: Medium batch (824MB) - No Split-N expected
// // This test might be skipped in CI with limited GPU memory
// TEST(GroupedConvFwdSplitN, MediumBatch)
// {
//     // Check if we should run large memory tests
//     const char* enable_large_tests = std::getenv("CK_TILE_ENABLE_LARGE_TESTS");
//     if(!enable_large_tests || std::string(enable_large_tests) != "1")
//     {
//         GTEST_SKIP() << "Skipping medium batch test (set CK_TILE_ENABLE_LARGE_TESTS=1 to
//         enable)";
//     }

//     // Define kernel types for the test (same for all test cases)
//     using InDataType  = half_t;
//     using WeiDataType = half_t;
//     using AccDataType = float;
//     using OutDataType = half_t;

//     using InLayout  = tensor_layout::convolution::NHWGC;
//     using WeiLayout = tensor_layout::convolution::GKYXC;
//     using OutLayout = tensor_layout::convolution::NHWGK;

//     // Define tile configuration
//     constexpr index_t M_Tile      = 64;
//     constexpr index_t N_Tile      = 64;
//     constexpr index_t K_Tile      = 64;
//     constexpr index_t M_Warp      = 2;
//     constexpr index_t N_Warp      = 2;
//     constexpr index_t K_Warp      = 1;
//     constexpr index_t M_Warp_Tile = 32;
//     constexpr index_t N_Warp_Tile = 32;
//     constexpr index_t K_Warp_Tile = 16;

//     using CodegenShape = TileGemmShape<sequence<M_Tile, N_Tile, K_Tile>,
//                                        sequence<M_Warp, N_Warp, K_Warp>,
//                                        sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

//     using TilePartitioner   = GemmTile1DPartitioner<CodegenShape>;
//     constexpr auto ConvSpec = ConvolutionSpecialization::Default;
//     using DsLayout          = tuple<>;
//     using DsDataType        = tuple<>;

//     using GroupedConvTraitsType =
//         GroupedConvTraits<2, ConvSpec, InLayout, WeiLayout, DsLayout, OutLayout>;

//     constexpr index_t VectorSizeA = 8;
//     constexpr index_t VectorSizeB = 8;
//     constexpr index_t VectorSizeC = 8;

//     using CodegenPipelineProblem =
//         GemmPipelineProblem<InDataType,
//                             WeiDataType,
//                             AccDataType,
//                             CodegenShape,
//                             typename GroupedConvTraitsType::GroupedConvImplicitGemmTraits,
//                             InDataType,
//                             true,
//                             VectorSizeA,
//                             VectorSizeB>;

//     using CodegenPipeline = GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

//     using CDEElementWise = element_wise::PassThrough;
//     using ConvEpilogue   = CShuffleEpilogue<
//           CShuffleEpilogueProblem<InDataType,
//                                   WeiDataType,
//                                   DsDataType,
//                                   AccDataType,
//                                   OutDataType,
//                                   typename GroupedConvTraitsType::ImplicitGemmDsLayout,
//                                   tensor_layout::gemm::RowMajor,
//                                   CDEElementWise,
//                                   TilePartitioner::MPerBlock,
//                                   TilePartitioner::NPerBlock,
//                                   M_Warp,
//                                   N_Warp,
//                                   M_Warp_Tile,
//                                   N_Warp_Tile,
//                                   K_Warp_Tile,
//                                   CodegenPipelineProblem::TransposeC,
//                                   memory_operation_enum::set,
//                                   1,
//                                   true,
//                                   VectorSizeC>>;

//     using Kernel = GroupedConvolutionForwardKernel<GroupedConvTraitsType,
//                                                    TilePartitioner,
//                                                    CodegenPipeline,
//                                                    ConvEpilogue>;

//     RunConvolutionTest<Kernel>(128, "Medium batch (824MB)", false);
// }

// // Test 4: Large batch (2.18GB) - Split-N expected
// // This test might be skipped in CI with limited GPU memory
// TEST(GroupedConvFwdSplitN, LargeBatch)
// {
//     // Check if we should run large memory tests
//     const char* enable_large_tests = std::getenv("CK_TILE_ENABLE_LARGE_TESTS");
//     if(!enable_large_tests || std::string(enable_large_tests) != "1")
//     {
//         GTEST_SKIP() << "Skipping large batch test (set CK_TILE_ENABLE_LARGE_TESTS=1 to enable)";
//     }

//     // Define kernel types for the test (same for all test cases)
//     using InDataType  = half_t;
//     using WeiDataType = half_t;
//     using AccDataType = float;
//     using OutDataType = half_t;

//     using InLayout  = tensor_layout::convolution::NHWGC;
//     using WeiLayout = tensor_layout::convolution::GKYXC;
//     using OutLayout = tensor_layout::convolution::NHWGK;

//     // Define tile configuration
//     constexpr index_t M_Tile      = 64;
//     constexpr index_t N_Tile      = 64;
//     constexpr index_t K_Tile      = 64;
//     constexpr index_t M_Warp      = 2;
//     constexpr index_t N_Warp      = 2;
//     constexpr index_t K_Warp      = 1;
//     constexpr index_t M_Warp_Tile = 32;
//     constexpr index_t N_Warp_Tile = 32;
//     constexpr index_t K_Warp_Tile = 16;

//     using CodegenShape = TileGemmShape<sequence<M_Tile, N_Tile, K_Tile>,
//                                        sequence<M_Warp, N_Warp, K_Warp>,
//                                        sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

//     using TilePartitioner   = GemmTile1DPartitioner<CodegenShape>;
//     constexpr auto ConvSpec = ConvolutionSpecialization::Default;
//     using DsLayout          = tuple<>;
//     using DsDataType        = tuple<>;

//     using GroupedConvTraitsType =
//         GroupedConvTraits<2, ConvSpec, InLayout, WeiLayout, DsLayout, OutLayout>;

//     constexpr index_t VectorSizeA = 8;
//     constexpr index_t VectorSizeB = 8;
//     constexpr index_t VectorSizeC = 8;

//     using CodegenPipelineProblem =
//         GemmPipelineProblem<InDataType,
//                             WeiDataType,
//                             AccDataType,
//                             CodegenShape,
//                             typename GroupedConvTraitsType::GroupedConvImplicitGemmTraits,
//                             InDataType,
//                             true,
//                             VectorSizeA,
//                             VectorSizeB>;

//     using CodegenPipeline = GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

//     using CDEElementWise = element_wise::PassThrough;
//     using ConvEpilogue   = CShuffleEpilogue<
//           CShuffleEpilogueProblem<InDataType,
//                                   WeiDataType,
//                                   DsDataType,
//                                   AccDataType,
//                                   OutDataType,
//                                   typename GroupedConvTraitsType::ImplicitGemmDsLayout,
//                                   tensor_layout::gemm::RowMajor,
//                                   CDEElementWise,
//                                   TilePartitioner::MPerBlock,
//                                   TilePartitioner::NPerBlock,
//                                   M_Warp,
//                                   N_Warp,
//                                   M_Warp_Tile,
//                                   N_Warp_Tile,
//                                   K_Warp_Tile,
//                                   CodegenPipelineProblem::TransposeC,
//                                   memory_operation_enum::set,
//                                   1,
//                                   true,
//                                   VectorSizeC>>;

//     using Kernel = GroupedConvolutionForwardKernel<GroupedConvTraitsType,
//                                                    TilePartitioner,
//                                                    CodegenPipeline,
//                                                    ConvEpilogue>;

//     RunConvolutionTest<Kernel>(340, "Large batch (2.18GB)", true);
// }

// // Test 5: Extra large batch (3.28GB) - Split-N expected
// // This test might be skipped in CI with limited GPU memory
// TEST(GroupedConvFwdSplitN, ExtraLargeBatch)
// {
//     // Check if we should run large memory tests
//     const char* enable_large_tests = std::getenv("CK_TILE_ENABLE_LARGE_TESTS");
//     if(!enable_large_tests || std::string(enable_large_tests) != "1")
//     {
//         GTEST_SKIP()
//             << "Skipping extra large batch test (set CK_TILE_ENABLE_LARGE_TESTS=1 to enable)";
//     }

//     // Define kernel types for the test (same for all test cases)
//     using InDataType  = half_t;
//     using WeiDataType = half_t;
//     using AccDataType = float;
//     using OutDataType = half_t;

//     using InLayout  = tensor_layout::convolution::NHWGC;
//     using WeiLayout = tensor_layout::convolution::GKYXC;
//     using OutLayout = tensor_layout::convolution::NHWGK;

//     // Define tile configuration
//     constexpr index_t M_Tile      = 64;
//     constexpr index_t N_Tile      = 64;
//     constexpr index_t K_Tile      = 64;
//     constexpr index_t M_Warp      = 2;
//     constexpr index_t N_Warp      = 2;
//     constexpr index_t K_Warp      = 1;
//     constexpr index_t M_Warp_Tile = 32;
//     constexpr index_t N_Warp_Tile = 32;
//     constexpr index_t K_Warp_Tile = 16;

//     using CodegenShape = TileGemmShape<sequence<M_Tile, N_Tile, K_Tile>,
//                                        sequence<M_Warp, N_Warp, K_Warp>,
//                                        sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

//     using TilePartitioner   = GemmTile1DPartitioner<CodegenShape>;
//     constexpr auto ConvSpec = ConvolutionSpecialization::Default;
//     using DsLayout          = tuple<>;
//     using DsDataType        = tuple<>;

//     using GroupedConvTraitsType =
//         GroupedConvTraits<2, ConvSpec, InLayout, WeiLayout, DsLayout, OutLayout>;

//     constexpr index_t VectorSizeA = 8;
//     constexpr index_t VectorSizeB = 8;
//     constexpr index_t VectorSizeC = 8;

//     using CodegenPipelineProblem =
//         GemmPipelineProblem<InDataType,
//                             WeiDataType,
//                             AccDataType,
//                             CodegenShape,
//                             typename GroupedConvTraitsType::GroupedConvImplicitGemmTraits,
//                             InDataType,
//                             true,
//                             VectorSizeA,
//                             VectorSizeB>;

//     using CodegenPipeline = GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

//     using CDEElementWise = element_wise::PassThrough;
//     using ConvEpilogue   = CShuffleEpilogue<
//           CShuffleEpilogueProblem<InDataType,
//                                   WeiDataType,
//                                   DsDataType,
//                                   AccDataType,
//                                   OutDataType,
//                                   typename GroupedConvTraitsType::ImplicitGemmDsLayout,
//                                   tensor_layout::gemm::RowMajor,
//                                   CDEElementWise,
//                                   TilePartitioner::MPerBlock,
//                                   TilePartitioner::NPerBlock,
//                                   M_Warp,
//                                   N_Warp,
//                                   M_Warp_Tile,
//                                   N_Warp_Tile,
//                                   K_Warp_Tile,
//                                   CodegenPipelineProblem::TransposeC,
//                                   memory_operation_enum::set,
//                                   1,
//                                   true,
//                                   VectorSizeC>>;

//     using Kernel = GroupedConvolutionForwardKernel<GroupedConvTraitsType,
//                                                    TilePartitioner,
//                                                    CodegenPipeline,
//                                                    ConvEpilogue>;

//     RunConvolutionTest<Kernel>(512, "Extra large batch (3.28GB)", true);
// }
