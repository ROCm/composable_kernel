// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <unordered_map>
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

// Test problem dimensions for GEMM validation
struct GemmProblemSize
{
    int M, N, K, split_k;
};

// Default test problems for kernel validation
const std::vector<GemmProblemSize> kTestProblems = {
    {256, 256, 256, 1},
    {256, 256, 256, 4},
};

// Helper function to determine if a layout is row-major at compile time
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// Calculate relative and absolute tolerances for numerical validation based on data types and
// problem size
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    // Determine compute precision based on smaller data type
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    // Calculate tolerances for GEMM operation
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // Calculate additional tolerances for split-k operations
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);

    // Return the maximum of both tolerance sets
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

// Problem context holds pre-allocated test data and computed reference results for a specific
// problem configuration
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct ProblemContext
{
    // Host tensors for input matrices and results
    ck_tile::HostTensor<ADataType> a_m_k;
    ck_tile::HostTensor<BDataType> b_k_n;
    ck_tile::HostTensor<CDataType> c_m_n_host_result; // Reference result from CPU GEMM
    ck_tile::HostTensor<CDataType> c_m_n_dev_result;  // Result buffer for GPU kernels

    // Device memory buffers
    ck_tile::DeviceMem a_m_k_dev_buf;
    ck_tile::DeviceMem b_k_n_dev_buf;
    ck_tile::DeviceMem c_m_n_dev_buf;

    // Problem configuration
    int M, N, K, split_k;
    ck_tile::index_t stride_a, stride_b, stride_c;
    bool structured_sparsity;
    float max_accumulated_value; // Precomputed for tolerance calculations

    // Constructor: allocates memory and initializes all test data
    ProblemContext(int M_,
                   int N_,
                   int K_,
                   int split_k_,
                   bool structured_sparsity_,
                   ck_tile::index_t stride_a_,
                   ck_tile::index_t stride_b_,
                   ck_tile::index_t stride_c_)
        : a_m_k(ck_tile::host_tensor_descriptor(M_, K_, stride_a_, is_row_major(ALayout{}))),
          b_k_n(ck_tile::host_tensor_descriptor(K_, N_, stride_b_, is_row_major(BLayout{}))),
          c_m_n_host_result(
              ck_tile::host_tensor_descriptor(M_, N_, stride_c_, is_row_major(CLayout{}))),
          c_m_n_dev_result(
              ck_tile::host_tensor_descriptor(M_, N_, stride_c_, is_row_major(CLayout{}))),
          a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes()),
          b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes()),
          c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes()),
          M(M_),
          N(N_),
          K(K_),
          split_k(split_k_),
          stride_a(stride_a_),
          stride_b(stride_b_),
          stride_c(stride_c_),
          structured_sparsity(structured_sparsity_)
    {
        initialize_data();
    }

    private:
    // Initialize test data: generate random inputs, apply sparsity, transfer to device, compute
    // reference
    void initialize_data()
    {
        // Generate random input data
        ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);

        // Apply structured sparsity pattern if requested
        if(structured_sparsity)
        {
            ck_tile::AdjustToStructuredSparsity<ADataType>{}(a_m_k);
        }

        // Transfer matrix A to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());

        // Transfer matrix B to device with special handling for int4 data type
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
            permute_vectors_i4x4_b(b_k_n_dev); // Apply required permutation for int4
            b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
        }
        else
        {
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        // Compute reference result using CPU GEMM
        c_m_n_host_result.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_result);

        // Precompute maximum value for tolerance calculations
        max_accumulated_value =
            *std::max_element(c_m_n_host_result.mData.begin(), c_m_n_host_result.mData.end());
    }

    public:
    // Reset output buffer to zero for next kernel test
    void reset_output_buffer() { c_m_n_dev_buf.SetZero(); }
};

// Base class for GEMM kernel validation tests - provides common functionality for testing GEMM
// kernels
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename Dispatcher>
class GemmKernelValidationTestBase : public ::testing::Test
{
    protected:
    // Kernel function map organized by trait names - use raw pointer like profiler
    std::unordered_map<std::string,
                       std::vector<std::function<std::tuple<std::string, float>(
                           ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>>>* kernel_map_;

    // Initialize the dispatcher and populate kernel map
    void InitDispatcher(bool structured_sparsity)
    {
        // Initialize dispatcher with sparsity configuration
        Dispatcher::init(structured_sparsity);

        // Allocate kernel map
        kernel_map_ =
            new std::unordered_map<std::string,
                                   std::vector<std::function<std::tuple<std::string, float>(
                                       ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>>>();

        // Copy kernels from dispatcher to local map
        for(const auto& [trait_name, kernels] : Dispatcher::get_kernel_map())
            (*kernel_map_)[trait_name] = kernels;
    }

    // Clean up allocated resources
    virtual void TearDown() override
    {
        delete kernel_map_;
        kernel_map_ = nullptr;
    }

    // Create problem context with all pre-allocated data for efficient kernel testing
    std::unique_ptr<
        ProblemContext<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout>>
    create_problem_context(int M, int N, int K, int split_k, bool structured_sparsity)
    {
        // Define layouts for input and output tensors
        const ALayout layout_a{};
        const BLayout layout_b{};
        const CLayout layout_c{};

        // Calculate strides based on tensor dimensions and layout
        ck_tile::index_t stride_a = is_row_major(layout_a) ? K : M;
        ck_tile::index_t stride_b = is_row_major(layout_b) ? N : K;
        ck_tile::index_t stride_c = is_row_major(layout_c) ? N : M;

        // Create and return problem context with all data initialized
        return std::make_unique<ProblemContext<ADataType,
                                               BDataType,
                                               CDataType,
                                               AccDataType,
                                               ALayout,
                                               BLayout,
                                               CLayout>>(
            M, N, K, split_k, structured_sparsity, stride_a, stride_b, stride_c);
    }

    // Test a single kernel with pre-allocated problem context
    std::tuple<std::string, float, bool> test_single_kernel(
        const std::function<std::tuple<std::string, float>(
            ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>& kernel_func,
        ProblemContext<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout>&
            context)
    {
        // Reset output buffer for this kernel test
        context.reset_output_buffer();

        // Prepare kernel arguments structure
        ck_tile::GemmHostArgs gemm_args = {
            context.a_m_k_dev_buf.GetDeviceBuffer(),
            context.b_k_n_dev_buf.GetDeviceBuffer(),
            context.c_m_n_dev_buf.GetDeviceBuffer(),
            context.split_k,
            context.M,
            context.N,
            context.K,
            context.stride_a,
            context.stride_b,
            context.stride_c,
        };

        try
        {
            // Configure stream for kernel execution
            ck_tile::stream_config stream_cfg{nullptr, false, 0, 0, 1};

            // Execute kernel and measure performance
            auto [kernel_name, execution_time] = kernel_func(gemm_args, stream_cfg);

            // Copy result back from device
            context.c_m_n_dev_buf.FromDevice(context.c_m_n_dev_result.data());

            // Validate kernel output against reference
            bool verified_correct = verify_results(context.c_m_n_dev_result,
                                                   context.c_m_n_host_result,
                                                   kernel_name,
                                                   context.K,
                                                   context.split_k,
                                                   context.max_accumulated_value);

            return std::make_tuple(kernel_name, execution_time, verified_correct);
        }
        catch(const std::exception& e)
        {
            ADD_FAILURE() << "        ERROR: " << e.what();
            return std::make_tuple("", 0, false);
        }
    }

    // Verify kernel results against reference with appropriate tolerances
    bool verify_results(const ck_tile::HostTensor<CDataType>& device_result,
                        const ck_tile::HostTensor<CDataType>& host_reference,
                        const std::string& kernel_name,
                        int K,
                        int split_k,
                        float max_accumulated_value)
    {
        // Calculate appropriate tolerances based on problem characteristics
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, split_k, max_accumulated_value);

        // Perform numerical comparison with calculated tolerances
        bool pass = ck_tile::check_err(device_result,
                                       host_reference,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        // Report result and return validation status
        EXPECT_EQ(pass, true) << "Verification failed : " << kernel_name;
        return pass;
    }
};
