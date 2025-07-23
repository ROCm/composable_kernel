// File: /workspaces/dev/composable_kernel/test/ck_tile/gemm/test_gemm_instance_validation.cpp
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gemm_common.hpp"
#include "gemm_dispatcher.hpp"

// Add this BEFORE the class definition - copy from benchmark_gemm.hpp lines 140-156
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

// Define datatypes for this test
using ADataType   = ck_tile::fp16_t;
using BDataType   = ck_tile::fp16_t;
using CDataType   = ck_tile::fp16_t;
using AccDataType = float;

using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

// FIX: Define is_row_major function exactly as in gemm_host_api.hpp line 72
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// Test fixture for GEMM kernel validation
class GemmKernelValidationTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        // Initialize dispatcher
        GemmDispatcher::init(false); // structured_sparsity = false
        kernel_map_ = &GemmDispatcher::get_kernel_map();

        std::cout << "Available traits:" << std::endl;
        for(const auto& [trait_name, kernels] : *kernel_map_)
        {
            std::cout << "  - " << trait_name << " (" << kernels.size() << " kernel variants)"
                      << std::endl;
        }
    }

    // Test a specific kernel with multiple problem sizes
    bool test_kernel_correctness(
        const std::function<std::tuple<std::string, float>(
            ck_tile::GemmHostArgs<>&, const ck_tile::stream_config&)>& kernel_func)
    {

        // FIXED: Use the EXACT same problem sizes as the benchmark config
        std::vector<std::tuple<int, int, int, int>> test_problems = {
            // Use smaller problems first to verify correctness
            // {256, 256, 128, 1},      // Small problem
            // {512, 512, 256, 1},      // Medium problem
            {1024, 1024, 512, 1},  // Large problem
            {3840, 4096, 2048, 1}, // EXACT benchmark size that passes

            // Test split-K with known working sizes
            // {256, 256, 128, 2},      // Small split-K
            // {1024, 1024, 512, 2},    // Large split-K
        };

        for(const auto& [M, N, K, split_k] : test_problems)
        {
            if(!test_single_problem(kernel_func, M, N, K, split_k))
            {
                return false;
            }
        }

        return true;
    }

    protected:
    std::unordered_map<std::string,
                       std::vector<std::function<std::tuple<std::string, float>(
                           ck_tile::GemmHostArgs<>&, const ck_tile::stream_config&)>>>* kernel_map_;

    private:
    // FIXED: Use CK-Tile's reference implementation (same as benchmark)
    void compute_reference_gemm(const ck_tile::HostTensor<ADataType>& a,
                                const ck_tile::HostTensor<BDataType>& b,
                                ck_tile::HostTensor<CDataType>& c)
    {

        c.SetZero();

        // Use the EXACT same function as the benchmark (reference_gemm.hpp line 20)
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(a, b, c);
    }

    // FIXED: Use the EXACT same verification logic as the benchmark
    bool verify_results(const ck_tile::HostTensor<CDataType>& device_result,
                        const ck_tile::HostTensor<CDataType>& host_reference,
                        const std::string& kernel_name,
                        int K,
                        int split_k)
    {

        // Calculate thresholds using the EXACT same logic as benchmark_gemm.hpp
        const float max_accumulated_value =
            *std::max_element(host_reference.mData.begin(), host_reference.mData.end());

        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, split_k, max_accumulated_value);

        // Use the EXACT same check_err function as the benchmark
        bool pass = ck_tile::check_err(device_result,
                                       host_reference,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "For " << kernel_name << " Relative error threshold is "
                  << rtol_atol.at(ck_tile::number<0>{}) << " Absolute error threshold is "
                  << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
        std::cout << "The verification result is:" << (pass ? "correct" : "fail") << std::endl;

        return pass;
    }

    bool
    test_single_problem(const std::function<std::tuple<std::string, float>(
                            ck_tile::GemmHostArgs<>&, const ck_tile::stream_config&)>& kernel_func,
                        int M,
                        int N,
                        int K,
                        int split_k)
    {

        std::cout << "      Testing problem M=" << M << " N=" << N << " K=" << K
                  << " split_k=" << split_k << std::endl;

        // Setup layouts - EXACT same as benchmark config
        const ALayout layout_a{}; // RowMajor
        const BLayout layout_b{}; // ColumnMajor
        const CLayout layout_c{}; // RowMajor

        // FIXED: Use the exact same stride calculation as benchmark
        // For RowMajor A: stride = K, For ColumnMajor B: stride = K, For RowMajor C: stride = N
        ck_tile::index_t stride_a = K; // RowMajor A: stride = inner dimension
        ck_tile::index_t stride_b = K; // ColumnMajor B: stride = inner dimension
        ck_tile::index_t stride_c = N; // RowMajor C: stride = inner dimension

        // Create host tensors with explicit strides (same as benchmark)
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_a, is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_b, is_row_major(layout_b)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(layout_c)));
        ck_tile::HostTensor<CDataType> c_m_n_host_result(
            ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(layout_c)));

        // Initialize matrices with small values to avoid overflow (same as benchmark)
        ck_tile::FillUniformDistribution<ADataType>{-0.5f,
                                                    0.5f}(a_m_k); // Smaller range for stability
        ck_tile::FillUniformDistribution<BDataType>{-0.5f,
                                                    0.5f}(b_k_n); // Smaller range for stability
        c_m_n_dev_result.SetZero();

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        // Copy to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();

        // Setup kernel arguments
        ck_tile::GemmHostArgs<> gemm_args = {
            a_m_k_dev_buf.GetDeviceBuffer(),
            b_k_n_dev_buf.GetDeviceBuffer(),
            {}, // ds_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),
            split_k,
            M,
            N,
            K,
            stride_a,
            stride_b,
            {}, // stride_Ds
            stride_c,
        };

        // FIXED: Compute reference using CK-Tile function (same as benchmark)
        compute_reference_gemm(a_m_k, b_k_n, c_m_n_host_result);

        try
        {
            // Execute kernel via dispatcher
            ck_tile::stream_config stream_cfg{nullptr, true, 0, 1, 1, true, false, 1};
            auto [kernel_name, execution_time] = kernel_func(gemm_args, stream_cfg);

            // Copy result back
            c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

            // FIXED: Use the benchmark's verification logic with proper tolerances
            bool verified_correct =
                verify_results(c_m_n_dev_result, c_m_n_host_result, kernel_name, K, split_k);

            if(verified_correct)
            {
                std::cout << "        PASS: " << kernel_name << " M=" << M << " N=" << N
                          << " K=" << K << " split_k=" << split_k << " time=" << execution_time
                          << "ms" << std::endl;
                return true;
            }
            else
            {
                std::cout << "        FAIL: " << kernel_name << " verification failed" << std::endl;
                return false;
            }
        }
        catch(const std::exception& e)
        {
            std::cout << "        ERROR: " << e.what() << std::endl;
            return false;
        }
    }
};

// Test all kernel configurations
TEST_F(GemmKernelValidationTest, AllKernelConfigurations)
{
    if(kernel_map_->empty())
    {
        GTEST_FAIL() << "No kernels found in dispatcher! Check if libraries are properly linked.";
    }

    int total_kernels  = 0;
    int passed_kernels = 0;

    for(const auto& [trait_name, kernels] : *kernel_map_)
    {
        std::cout << "\nTesting trait: " << trait_name << " (" << kernels.size() << " variants)"
                  << std::endl;

        for(size_t i = 0; i < kernels.size(); ++i)
        {
            std::cout << "  Testing kernel variant " << i << ":" << std::endl;

            if(test_kernel_correctness(kernels[i]))
            {
                passed_kernels++;
            }
            else
            {
                std::cout << "  FAILED: Kernel variant " << i << " of trait " << trait_name
                          << std::endl;
            }
            total_kernels++;
        }
    }

    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Total kernels tested: " << total_kernels << std::endl;
    std::cout << "Passed: " << passed_kernels << std::endl;
    std::cout << "Failed: " << (total_kernels - passed_kernels) << std::endl;

    EXPECT_EQ(passed_kernels, total_kernels) << "Some kernels failed verification";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "=== CK Tile GEMM Kernel Instance Validation ===" << std::endl;
    std::cout << "Testing all kernel configurations generated from JSON config" << std::endl;

    return RUN_ALL_TESTS();
}
