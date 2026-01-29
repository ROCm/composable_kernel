// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/mhc.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_mhc.hpp"
#include "ck_tile/host/check_err.hpp"

template <typename Tuple>
class TestCkTileMHC : public ::testing::Test
{
    public:
    // protected:
    // using XDataType               = std::tuple_element_t<0, Tuple>;
    // using ComputeDataType         = std::tuple_element_t<1, Tuple>;
    // using YDataType               = std::tuple_element_t<2, Tuple>;
    // using ReduceOpsType           = std::tuple_element_t<3, Tuple>;
    // using ElementwiseOpsType      = std::tuple_element_t<4, Tuple>;
    // using AccumulatorOpsType      = std::tuple_element_t<5, Tuple>;
    // using InterBlockReduceOpsType = std::tuple_element_t<6, Tuple>;
    // using BlockWarps_             = std::tuple_element_t<7, Tuple>;
    // using BlockTile_              = std::tuple_element_t<8, Tuple>;
    // using WarpTile_               = std::tuple_element_t<9, Tuple>;
    // using ThreadTile_             = std::tuple_element_t<10, Tuple>;

    // using TestReduce2dShape =
    //     ck_tile::Reduce2dShape<BlockWarps_, BlockTile_, WarpTile_, ThreadTile_>;

    // template <std::size_t InputDim, typename KeptDimSeq, typename ReduceDimSeq>
    // void RunGenericTest(const std::vector<ck_tile::index_t>& input_shape,
    //                     const std::vector<ck_tile::index_t>& input_strides,
    //                     const std::vector<ck_tile::index_t>& output_shape,
    //                     const std::vector<ck_tile::index_t>& output_strides,
    //                     ck_tile::index_t kept_dim_len_prod,
    //                     ck_tile::index_t total_reduce_elements,
    //                     KeptDimSeq kept_dims,
    //                     ReduceDimSeq reduce_dims)
    void RunGenericTest()
    {

        // Test parameters
        const int B  = 8;     // Batch size
        const int n  = 4;     // Expansion rate (aka streams)
        const int C  = 64;    // Output layer dim (reduced to avoid shared memory overflow)
        const int nC = n * C; // Total input dimension

        const int output_dim = 2 * n + n * n; // 2n + n^2 = 8 + 16 = 24 for n=4

        // Allocate host tensors
        ck_tile::HostTensor<float> h_x({B, nC});              // Input [B, nC]
        ck_tile::HostTensor<float> h_phi({nC, output_dim});   // Weights [nC, 2n+n^2]
        ck_tile::HostTensor<float> h_output({B, output_dim}); // Output [B, 2n+n^2]

        // Initialize with random data
        ck_tile::FillUniformDistribution<float>{-1.0f, 1.0f}(h_x);
        ck_tile::FillUniformDistribution<float>{-0.5f, 0.5f}(h_phi);
        h_output.SetZero();

        // Allocate device memory
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_phi_mem(h_phi.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_output_mem(h_output.get_element_space_size_in_bytes());

        // Copy data to device
        d_x_mem.ToDevice(h_x.data());
        d_phi_mem.ToDevice(h_phi.data());
        d_output_mem.ToDevice(h_output.data());

        // DEBUG: Print first few values of h_x to compare with x_lds
        std::cout << "DEBUG h_x[0, 0:4]: " << h_x(0, 0) << ", " << h_x(0, 1) << ", " << h_x(0, 2)
                  << ", " << h_x(0, 3) << std::endl;
        std::cout << "DEBUG h_x[1, 0:4]: " << h_x(1, 0) << ", " << h_x(1, 1) << ", " << h_x(1, 2)
                  << ", " << h_x(1, 3) << std::endl;

        // DEBUG: Print first few values of h_phi column 0 (stream_id=0)
        std::cout << "DEBUG h_phi[0:4, 0]: " << h_phi(0, 0) << ", " << h_phi(1, 0) << ", "
                  << h_phi(2, 0) << ", " << h_phi(3, 0) << std::endl;

        // Define block shape for the kernel
        // For simplicity, we use a basic configuration
        using BlockShape =
            ck_tile::Generic2dBlockShape<ck_tile::sequence<1, 256>, // Block tile size [M, N] - 1
                                                                    // row, 256 columns
                                         ck_tile::sequence<1, 256>, // Threads per block [M, N]
                                         ck_tile::sequence<1, 1>    // Vector size [M, N]
                                         >;

        // Define the Problem type
        using Problem = ck_tile::MHCProblem<float,     // XDataType
                                            float,     // ComputeDataType
                                            float,     // YDataType
                                            BlockShape // BlockShape
                                            >;

        // Define the Kernel type with default policy (naive version)
        using Kernel =
            ck_tile::ManifoldConstrainedHyperConnection<Problem, ck_tile::MHCDefaultPolicy>;

        // Define the CK Tile version kernel (v2 with proper tiling)
        // Use compile-time parameters for B, n, C
        using KernelCKTile = ck_tile::
            ManifoldConstrainedHyperConnectionTiled<Problem, ck_tile::MHCDefaultPolicy, B, n, C>;

        // Kernel launch configuration
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        const ck_tile::index_t kGridSize       = B; // One block per batch element
        constexpr ck_tile::index_t kBlockPerCu = 1;

        std::cout << "Launching MHC kernel (naive version) with:" << std::endl;
        std::cout << "  Batch size (B): " << B << std::endl;
        std::cout << "  Expansion factor (n): " << n << std::endl;
        std::cout << "  Channels per stream (C): " << C << std::endl;
        std::cout << "  Input dimension (nC): " << nC << std::endl;
        std::cout << "  Output dimension (2n+n²): " << output_dim << std::endl;
        std::cout << "  Grid size: " << kGridSize << std::endl;
        std::cout << "  Block size: " << kBlockSize << std::endl;

        // Get shared memory size
        const ck_tile::index_t smem_size = Kernel::GetSmemSize();
        std::cout << "  Shared memory size: " << smem_size << " bytes" << std::endl;

        // Kernel parameters
        const float r          = 1.0f;
        const float alpha_pre  = 1.0f;
        const float alpha_post = 1.0f;
        const float alpha_res  = 1.0f;
        const float bias       = 0.0f;

        // Kernel launch
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                              kGridSize,
                                              kBlockSize,
                                              smem_size,
                                              static_cast<float*>(d_x_mem.GetDeviceBuffer()),
                                              static_cast<float*>(d_phi_mem.GetDeviceBuffer()),
                                              static_cast<float*>(d_output_mem.GetDeviceBuffer()),
                                              B,
                                              n,
                                              C,
                                              r,
                                              alpha_pre,
                                              alpha_post,
                                              alpha_res,
                                              bias));

        // Copy results back to host
        d_output_mem.FromDevice(h_output.data());

        std::cout << "Kernel launched successfully!" << std::endl;

        // Print output to verify kernel actually modified the tensor
        std::cout << "\nOutput tensor (first 2 batches, all " << output_dim
                  << " elements):" << std::endl;
        for(int b = 0; b < std::min(2, B); b++)
        {
            std::cout << "Batch " << b << ": [";
            for(int i = 0; i < output_dim; i++)
            {
                std::cout << h_output(b, i);
                if(i < output_dim - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Verify that output is not all zeros (kernel actually ran)
        bool has_nonzero = false;
        for(int b = 0; b < B && !has_nonzero; b++)
        {
            for(int i = 0; i < output_dim && !has_nonzero; i++)
            {
                if(std::abs(h_output(b, i)) > 1e-6f)
                {
                    has_nonzero = true;
                }
            }
        }

        std::cout << "\nNaive kernel output verification: "
                  << (has_nonzero ? "PASS (non-zero values found)" : "FAIL (all zeros)")
                  << std::endl;

        // Test CK Tile version
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing CK Tile version kernel..." << std::endl;
        std::cout << "========================================" << std::endl;

        ck_tile::HostTensor<float> h_output_cktile({B, output_dim});
        h_output_cktile.SetZero();

        ck_tile::DeviceMem d_output_cktile_mem(h_output_cktile.get_element_space_size_in_bytes());
        d_output_cktile_mem.ToDevice(h_output_cktile.data());

        // Launch CK Tile kernel (B, n, C are template parameters)
        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                               ck_tile::make_kernel<kBlockPerCu>(
                                   KernelCKTile{},
                                   kGridSize,
                                   kBlockSize,
                                   smem_size,
                                   static_cast<float*>(d_x_mem.GetDeviceBuffer()),
                                   static_cast<float*>(d_phi_mem.GetDeviceBuffer()),
                                   static_cast<float*>(d_output_cktile_mem.GetDeviceBuffer()),
                                   r,
                                   alpha_pre,
                                   alpha_post,
                                   alpha_res,
                                   bias));

        d_output_cktile_mem.FromDevice(h_output_cktile.data());

        std::cout << "\nCK Tile kernel output (first 2 batches):" << std::endl;
        for(int b = 0; b < std::min(2, B); b++)
        {
            std::cout << "Batch " << b << ": [";
            for(int i = 0; i < output_dim; i++)
            {
                std::cout << h_output_cktile(b, i);
                if(i < output_dim - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Compute reference result
        ck_tile::HostTensor<float> h_output_ref({B, output_dim});
        h_output_ref.SetZero();

        std::cout << "\nComputing reference result..." << std::endl;
        ck_tile::reference_mhc<float, float, float, float>(
            h_x, h_phi, h_output_ref, n, C, r, alpha_pre, alpha_post, alpha_res, bias);

        std::cout << "\nReference output (first 2 batches):" << std::endl;
        for(int b = 0; b < std::min(2, B); b++)
        {
            std::cout << "Batch " << b << ": [";
            for(int i = 0; i < output_dim; i++)
            {
                std::cout << h_output_ref(b, i);
                if(i < output_dim - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Validate results
        const float rtol = 1e-3f; // Relative tolerance
        const float atol = 1e-3f; // Absolute tolerance

        bool pass_naive = ck_tile::check_err(
            h_output, h_output_ref, "Error: Naive MHC output mismatch!", rtol, atol);

        bool pass_cktile = ck_tile::check_err(
            h_output_cktile, h_output_ref, "Error: CK Tile MHC output mismatch!", rtol, atol);

        std::cout << "\n========================================" << std::endl;
        std::cout << "Final Results:" << std::endl;
        std::cout << "  Naive kernel:   " << (pass_naive ? "PASS" : "FAIL") << std::endl;
        std::cout << "  CK Tile kernel: " << (pass_cktile ? "PASS" : "FAIL") << std::endl;
        std::cout << "========================================" << std::endl;

        EXPECT_TRUE(pass_naive && pass_cktile);
    }

    // Test with specific batch size (template version with compile-time parameters)
    template <int B = 16, int n = 4, int C = 64>
    void RunBatchSizeTest()
    {
        const int nC         = n * C;         // Total input dimension
        const int output_dim = 2 * n + n * n; // 2n + n^2

        std::cout << "\n--- Testing batch size B=" << B << " (n=" << n << ", C=" << C << ") ---"
                  << std::endl;

        // Allocate host tensors
        ck_tile::HostTensor<float> h_x({B, nC});
        ck_tile::HostTensor<float> h_phi({nC, output_dim});
        ck_tile::HostTensor<float> h_output({B, output_dim});

        // Initialize with random data
        ck_tile::FillUniformDistribution<float>{-1.0f, 1.0f}(h_x);
        ck_tile::FillUniformDistribution<float>{-0.5f, 0.5f}(h_phi);
        h_output.SetZero();

        // Allocate device memory
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_phi_mem(h_phi.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_output_mem(h_output.get_element_space_size_in_bytes());

        // Copy data to device
        d_x_mem.ToDevice(h_x.data());
        d_phi_mem.ToDevice(h_phi.data());
        d_output_mem.ToDevice(h_output.data());

        // Define block shape
        using BlockShape = ck_tile::Generic2dBlockShape<ck_tile::sequence<1, 256>,
                                                        ck_tile::sequence<1, 256>,
                                                        ck_tile::sequence<1, 1>>;

        using Problem = ck_tile::MHCProblem<float, float, float, BlockShape>;

        // Use template parameters for B, n, C (compile-time)
        // This allows better optimization and proper use of store_tile
        using KernelExpansionParallel = ck_tile::
            ManifoldConstrainedHyperConnectionTiled<Problem, ck_tile::MHCDefaultPolicy, B, n, C>;

        const ck_tile::index_t kBlockSize      = KernelExpansionParallel::BlockSize();
        const ck_tile::index_t kGridSize       = (output_dim + 15) / 16;
        constexpr ck_tile::index_t kBlockPerCu = 1;

        // const float r = 1.0f, alpha_pre = 1.0f, alpha_post = 1.0f, alpha_res = 1.0f, bias = 0.0f;
        const float r = 2.0f, alpha_pre = 1.5f, alpha_post = 2.5f, alpha_res = 3.5f, bias = 1.5f;

        // Launch kernel (B, n, C are now template parameters, not runtime)
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(KernelExpansionParallel{},
                                              kGridSize,
                                              kBlockSize,
                                              0,
                                              static_cast<float*>(d_x_mem.GetDeviceBuffer()),
                                              static_cast<float*>(d_phi_mem.GetDeviceBuffer()),
                                              static_cast<float*>(d_output_mem.GetDeviceBuffer()),
                                              r,
                                              alpha_pre,
                                              alpha_post,
                                              alpha_res,
                                              bias));

        d_output_mem.FromDevice(h_output.data());

        // Compute reference
        ck_tile::HostTensor<float> h_output_ref({B, output_dim});
        h_output_ref.SetZero();
        ck_tile::reference_mhc<float, float, float, float>(
            h_x, h_phi, h_output_ref, n, C, r, alpha_pre, alpha_post, alpha_res, bias);

        // Validate
        bool pass =
            ck_tile::check_err(h_output, h_output_ref, "Error: Batch size mismatch!", 1e-3f, 1e-3f);

        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;

        if(!pass)
        {
            // Print first few values for debugging
            std::cout << "  First batch kernel output: [";
            for(int i = 0; i < std::min(4, output_dim); i++)
            {
                std::cout << h_output(0, i);
                if(i < std::min(4, output_dim) - 1)
                    std::cout << ", ";
            }
            std::cout << " ...]" << std::endl;

            std::cout << "  First batch reference: [";
            for(int i = 0; i < std::min(4, output_dim); i++)
            {
                std::cout << h_output_ref(0, i);
                if(i < std::min(4, output_dim) - 1)
                    std::cout << ", ";
            }
            std::cout << " ...]" << std::endl;
        }

        EXPECT_TRUE(pass);
    }

    // Test with specific batch size and custom activation function
    template <int B                   = 16,
              int n                   = 4,
              int C                   = 64,
              typename ActivationFunc = ck_tile::element_wise::Sigmoid>
    void RunBatchSizeTestWithActivation()
    {
        const int nC         = n * C;         // Total input dimension
        const int output_dim = 2 * n + n * n; // 2n + n^2

        std::cout << "\n--- Testing batch size B=" << B << " (n=" << n << ", C=" << C
                  << ") with activation: " << ActivationFunc::name << " ---" << std::endl;

        // Allocate host tensors
        ck_tile::HostTensor<float> h_x({B, nC});
        ck_tile::HostTensor<float> h_phi({nC, output_dim});
        ck_tile::HostTensor<float> h_output({B, output_dim});

        // Initialize with random data
        ck_tile::FillUniformDistribution<float>{-1.0f, 1.0f}(h_x);
        ck_tile::FillUniformDistribution<float>{-0.5f, 0.5f}(h_phi);
        h_output.SetZero();

        // Allocate device memory
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_phi_mem(h_phi.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_output_mem(h_output.get_element_space_size_in_bytes());

        // Copy data to device
        d_x_mem.ToDevice(h_x.data());
        d_phi_mem.ToDevice(h_phi.data());
        d_output_mem.ToDevice(h_output.data());

        // Define block shape
        using BlockShape = ck_tile::Generic2dBlockShape<ck_tile::sequence<1, 256>,
                                                        ck_tile::sequence<1, 256>,
                                                        ck_tile::sequence<1, 1>>;

        using Problem = ck_tile::MHCProblem<float, float, float, BlockShape>;

        // Use template parameters for B, n, C, and Activation (compile-time)
        using KernelExpansionParallel =
            ck_tile::ManifoldConstrainedHyperConnectionTiled<Problem,
                                                             ck_tile::MHCDefaultPolicy,
                                                             B,
                                                             n,
                                                             C,
                                                             256,
                                                             ActivationFunc>;

        const ck_tile::index_t kBlockSize      = KernelExpansionParallel::BlockSize();
        const ck_tile::index_t kGridSize       = (output_dim + 15) / 16;
        constexpr ck_tile::index_t kBlockPerCu = 1;

        const float r = 2.0f, alpha_pre = 1.5f, alpha_post = 2.5f, alpha_res = 3.5f, bias = 1.5f;

        // Launch kernel
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(KernelExpansionParallel{},
                                              kGridSize,
                                              kBlockSize,
                                              0,
                                              static_cast<float*>(d_x_mem.GetDeviceBuffer()),
                                              static_cast<float*>(d_phi_mem.GetDeviceBuffer()),
                                              static_cast<float*>(d_output_mem.GetDeviceBuffer()),
                                              r,
                                              alpha_pre,
                                              alpha_post,
                                              alpha_res,
                                              bias));

        d_output_mem.FromDevice(h_output.data());

        // Compute reference with the same activation function
        ck_tile::HostTensor<float> h_output_ref({B, output_dim});
        h_output_ref.SetZero();
        ck_tile::reference_mhc<float, float, float, float, ActivationFunc>(h_x,
                                                                           h_phi,
                                                                           h_output_ref,
                                                                           n,
                                                                           C,
                                                                           r,
                                                                           alpha_pre,
                                                                           alpha_post,
                                                                           alpha_res,
                                                                           bias,
                                                                           ActivationFunc{});

        // Validate
        bool pass = ck_tile::check_err(
            h_output, h_output_ref, "Error: Activation function mismatch!", 1e-3f, 1e-3f);

        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;

        if(!pass)
        {
            // Print first few values for debugging
            std::cout << "  First batch kernel output: [";
            for(int i = 0; i < std::min(8, output_dim); i++)
            {
                std::cout << h_output(0, i);
                if(i < std::min(8, output_dim) - 1)
                    std::cout << ", ";
            }
            std::cout << " ...]" << std::endl;

            std::cout << "  First batch reference: [";
            for(int i = 0; i < std::min(8, output_dim); i++)
            {
                std::cout << h_output_ref(0, i);
                if(i < std::min(8, output_dim) - 1)
                    std::cout << ", ";
            }
            std::cout << " ...]" << std::endl;
        }

        EXPECT_TRUE(pass);
    }

    // Test with multiple arbitrary batch sizes
    void RunArbitraryBatchSizeTest()
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing Arbitrary Batch Sizes..." << std::endl;
        std::cout << "  Expansion factor (n): 4" << std::endl;
        std::cout << "  Channels per stream (C): 64" << std::endl;
        std::cout << "  Output dimension: 24" << std::endl;
        std::cout << "========================================" << std::endl;

        // Call template versions with compile-time parameters
        RunBatchSizeTest<1, 4, 64>();
        RunBatchSizeTest<7, 4, 64>();
        RunBatchSizeTest<15, 4, 64>();
        RunBatchSizeTest<16, 4, 64>();
        RunBatchSizeTest<17, 4, 64>();
        RunBatchSizeTest<23, 4, 64>();
        RunBatchSizeTest<32, 4, 64>();
        RunBatchSizeTest<33, 4, 64>();
        RunBatchSizeTest<47, 4, 64>();
        RunBatchSizeTest<48, 4, 64>();
        RunBatchSizeTest<64, 4, 64>();

        std::cout << "\n========================================" << std::endl;
        std::cout << "Overall Result: ALL TESTS COMPLETED" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // New test: Parallelize by expansion factor (n) instead of batch
    void RunExpansionParallelTest()
    {
        // Test parameters - realistic sizes for BlockGemm
        const int B          = 16;            // Batch size (M dimension in GEMM)
        const int n          = 4;             // Expansion rate
        const int C          = 64;            // Output layer dim (smaller for testing)
        const int nC         = n * C;         // Total input dimension = 256 (K dimension in GEMM)
        const int output_dim = 2 * n + n * n; // 2n + n^2 = 24 for n=4

        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing Expansion-Parallel MHC kernel..." << std::endl;
        std::cout << "  Batch size (B): " << B << std::endl;
        std::cout << "  Expansion factor (n): " << n << std::endl;
        std::cout << "  Channels per stream (C): " << C << std::endl;
        std::cout << "  Grid size: " << output_dim << " (one block per expansion stream)"
                  << std::endl;
        std::cout << "========================================" << std::endl;

        // Allocate host tensors
        ck_tile::HostTensor<float> h_x({B, nC});
        ck_tile::HostTensor<float> h_phi({nC, output_dim});
        ck_tile::HostTensor<float> h_output({B, output_dim});

        // Initialize with random data
        ck_tile::FillUniformDistribution<float>{-1.0f, 1.0f}(h_x);
        ck_tile::FillUniformDistribution<float>{-0.5f, 0.5f}(h_phi);
        h_output.SetZero();

        // Allocate device memory
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_phi_mem(h_phi.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_output_mem(h_output.get_element_space_size_in_bytes());

        // Copy data to device
        d_x_mem.ToDevice(h_x.data());
        d_phi_mem.ToDevice(h_phi.data());
        d_output_mem.ToDevice(h_output.data());

        // Define block shape
        using BlockShape = ck_tile::Generic2dBlockShape<ck_tile::sequence<1, 256>,
                                                        ck_tile::sequence<1, 256>,
                                                        ck_tile::sequence<1, 1>>;

        using Problem                 = ck_tile::MHCProblem<float, float, float, BlockShape>;
        using KernelExpansionParallel = ck_tile::
            ManifoldConstrainedHyperConnectionTiled<Problem, ck_tile::MHCDefaultPolicy, B, n, C>;

        const ck_tile::index_t kBlockSize = KernelExpansionParallel::BlockSize();
        // Grid size: one block per 16 output columns (since BlockGemm processes N=16)
        const ck_tile::index_t kGridSize       = (output_dim + 15) / 16;
        constexpr ck_tile::index_t kBlockPerCu = 1;

        const float r = 1.0f, alpha_pre = 1.0f, alpha_post = 1.0f, alpha_res = 1.0f, bias = 0.0f;

        // Launch expansion-parallel kernel (B, n, C are template parameters)
        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(KernelExpansionParallel{},
                                              kGridSize,
                                              kBlockSize,
                                              0,
                                              static_cast<float*>(d_x_mem.GetDeviceBuffer()),
                                              static_cast<float*>(d_phi_mem.GetDeviceBuffer()),
                                              static_cast<float*>(d_output_mem.GetDeviceBuffer()),
                                              r,
                                              alpha_pre,
                                              alpha_post,
                                              alpha_res,
                                              bias));

        d_output_mem.FromDevice(h_output.data());

        // Print kernel output to debug
        std::cout << "\nKernel output (first 2 batches, first 8 elements):" << std::endl;
        for(int b = 0; b < std::min(2, B); b++)
        {
            std::cout << "Batch " << b << ": [";
            for(int i = 0; i < std::min(8, output_dim); i++)
            {
                std::cout << h_output(b, i);
                if(i < std::min(8, output_dim) - 1)
                    std::cout << ", ";
            }
            std::cout << " ...]" << std::endl;
        }

        // Compute reference
        ck_tile::HostTensor<float> h_output_ref({B, output_dim});
        h_output_ref.SetZero();
        ck_tile::reference_mhc<float, float, float, float>(
            h_x, h_phi, h_output_ref, n, C, r, alpha_pre, alpha_post, alpha_res, bias);

        // Print reference output
        std::cout << "\nReference output (first 2 batches, first 8 elements):" << std::endl;
        for(int b = 0; b < std::min(2, B); b++)
        {
            std::cout << "Batch " << b << ": [";
            for(int i = 0; i < std::min(8, output_dim); i++)
            {
                std::cout << h_output_ref(b, i);
                if(i < std::min(8, output_dim) - 1)
                    std::cout << ", ";
            }
            std::cout << " ...]" << std::endl;
        }

        // Validate
        bool pass = ck_tile::check_err(
            h_output, h_output_ref, "Error: Expansion-parallel MHC mismatch!", 1e-3f, 1e-3f);

        std::cout << "Expansion-parallel kernel: " << (pass ? "PASS" : "FAIL") << std::endl;
        EXPECT_TRUE(pass);
    }

    void RunGenericTestOld()
    {
        // auto h_ys = ck_tile::generate_tuple(
        //     [&output_shape, &output_strides](auto /*i*/) {
        //         return ck_tile::HostTensor<YDataType>(output_shape, output_strides);
        //     },
        //     ck_tile::number<number_operations>{});

        // auto h_ys_ref = ck_tile::generate_tuple(
        //     [&output_shape, &output_strides](auto /*i*/) {
        //         return ck_tile::HostTensor<YDataType>(output_shape, output_strides);
        //     },
        //     ck_tile::number<number_operations>{});

        // ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(h_x);

        // ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
        //     h_ys.template at<i>().SetZero();
        //     h_ys_ref.template at<i>().SetZero();
        // });

        // auto output_number_elements = [&output_shape]() {
        //     ck_tile::index_t prod = 1;
        //     for(auto len : output_shape)
        //         prod *= len;
        //     return prod;
        // }();

        // auto output_buffer_size =
        //     number_operations * h_ys.get(ck_tile::number<0>{}).get_element_space_size_in_bytes();
        // ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        // ck_tile::DeviceMem d_y_mem(output_buffer_size);

        // std::vector<YDataType> h(number_operations * output_number_elements);

        // // Init the output data with identity values respective to each reduce op
        // ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
        //     constexpr auto op       = ReduceOpsType{}.at(i);
        //     const auto identity_val = op.template GetIdentityValue<YDataType>();
        //     std::fill(h.begin() + i * output_number_elements,
        //               h.begin() + (i + 1) * output_number_elements,
        //               identity_val);
        // });

        // d_x_mem.ToDevice(h_x.data());
        // d_y_mem.ToDevice(h.data());

        // using Problem = ck_tile::Reduce2dProblem<XDataType,
        //                                          ComputeDataType,
        //                                          YDataType,
        //                                          TestReduce2dShape,
        //                                          ReduceOpsType,
        //                                          KeptDimSeq,
        //                                          ReduceDimSeq,
        //                                          InputDim>;

        // using Kernel = ck_tile::MultiReduceMultiblock<Problem>;

        // // Launch configuration
        // const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        // constexpr ck_tile::index_t kBlockPerCu = 1;

        // auto elementwise_ops =
        //     make_elementwise_ops_tuple(total_reduce_elements, ElementwiseOpsType{});
        // auto accumulator_ops =
        //     make_elementwise_ops_tuple(total_reduce_elements, AccumulatorOpsType{});

        // auto [num_block_tile_iterations, block_group_size] =
        //     typename Kernel::TilePartitioner{total_reduce_elements}.GetBlockGroupParams();

        // std::cout << "Block group size: " << block_group_size
        //           << ", Num block tile iterations: " << num_block_tile_iterations
        //           << ", Reduce total length: " << total_reduce_elements << std::endl;

        // ck_tile::index_t kGridSize =
        //     ((kept_dim_len_prod + TestReduce2dShape::Block_M - 1) / TestReduce2dShape::Block_M) *
        //     block_group_size;

        // // Generic helper to create tuple from vector based on compile-time size
        // auto make_shape_tuple = []<std::size_t N>(const std::vector<ck_tile::index_t>& vec) {
        //     return [&vec]<std::size_t... I>(std::index_sequence<I...>) {
        //         return ck_tile::make_tuple(vec[I]...);
        //     }(std::make_index_sequence<N>{});
        // };

        // auto input_shape_tuple   = make_shape_tuple.template operator()<InputDim>(input_shape);
        // auto input_strides_tuple = make_shape_tuple.template operator()<InputDim>(input_strides);

        // if(!Kernel::IsSupportedArgument()) // TODO
        // {
        // }

        // ck_tile::launch_kernel(
        //     ck_tile::stream_config{nullptr, false, 0},
        //     ck_tile::make_kernel<kBlockPerCu>(Kernel{},
        //                                       kGridSize,
        //                                       kBlockSize,
        //                                       0,
        //                                       static_cast<XDataType*>(d_x_mem.GetDeviceBuffer()),
        //                                       static_cast<YDataType*>(d_y_mem.GetDeviceBuffer()),
        //                                       input_shape_tuple,
        //                                       input_strides_tuple,
        //                                       kept_dims,
        //                                       reduce_dims,
        //                                       output_number_elements,
        //                                       elementwise_ops,
        //                                       accumulator_ops,
        //                                       InterBlockReduceOpsType{}));

        // TODO: Reference computation + Transfer data back to host
        // EXPECT_TRUE(true);
    }
};
