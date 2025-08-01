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
    {3840, 4096, 2048, 1}, {1024, 1024, 1024, 1},
    // Add more test cases as needed
};

// Helper function to determine if a layout is row-major
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// Calculate relative and absolute tolerances for numerical validation
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

// Base class for GEMM kernel validation tests
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
        Dispatcher::init(structured_sparsity);
        kernel_map_ =
            new std::unordered_map<std::string,
                                   std::vector<std::function<std::tuple<std::string, float>(
                                       ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>>>();

        // Populate kernel map from dispatcher
        for(const auto& [trait_name, kernels] : Dispatcher::get_kernel_map())
            (*kernel_map_)[trait_name] = kernels;
    }

    // Clean up resources
    virtual void TearDown() override
    {
        delete kernel_map_;
        kernel_map_ = nullptr;
    }

    // Compute reference GEMM result on host for validation
    void compute_reference_gemm(const ck_tile::HostTensor<ADataType>& a,
                                const ck_tile::HostTensor<BDataType>& b,
                                ck_tile::HostTensor<CDataType>& c)
    {
        c.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(a, b, c);
    }

    // Verify kernel results against reference implementation
    bool verify_results(const ck_tile::HostTensor<CDataType>& device_result,
                        const ck_tile::HostTensor<CDataType>& host_reference,
                        const std::string& kernel_name,
                        int K,
                        int split_k)
    {
        const float max_accumulated_value =
            *std::max_element(host_reference.mData.begin(), host_reference.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, split_k, max_accumulated_value);
        bool pass = ck_tile::check_err(device_result,
                                       host_reference,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));
        EXPECT_EQ(pass, true) << "Verification failed : " << kernel_name;
        return pass;
    }

    // Test a single GEMM problem with given dimensions
    std::tuple<std::string, float, bool>
    test_single_problem(const std::function<std::tuple<std::string, float>(
                            ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>& kernel_func,
                        int M,
                        int N,
                        int K,
                        int split_k,
                        bool structured_sparsity = false)
    {
        // Define layouts for input and output tensors
        const ALayout layout_a{};
        const BLayout layout_b{};
        const CLayout layout_c{};

        // Calculate strides based on tensor dimensions and layout
        ck_tile::index_t stride_a = is_row_major(layout_a) ? K : M;
        ck_tile::index_t stride_b = is_row_major(layout_b) ? N : K;
        ck_tile::index_t stride_c = is_row_major(layout_c) ? N : M;

        // Create host tensors for test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_a, is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_b, is_row_major(layout_b)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(layout_c)));
        ck_tile::HostTensor<CDataType> c_m_n_host_result(
            ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(layout_c)));

        // Initialize input tensors with random data
        ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);
        c_m_n_dev_result.SetZero();

        // Apply structured sparsity pattern to matrix A if requested
        if(structured_sparsity)
        {
            ck_tile::AdjustToStructuredSparsity<ADataType>{}(a_m_k);
        }

        // Allocate device memory buffers
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        // Copy matrix A to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());

        // Handle special data type conversions for matrix B
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Apply int4 vector permutation for device compatibility
            ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
            permute_vectors_i4x4_b(b_k_n_dev);
            b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
        }
        else
        {
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        // Initialize output buffer to zero
        c_m_n_dev_buf.SetZero();

        // Prepare kernel arguments structure
        ck_tile::GemmHostArgs gemm_args = {
            a_m_k_dev_buf.GetDeviceBuffer(),
            b_k_n_dev_buf.GetDeviceBuffer(),
            c_m_n_dev_buf.GetDeviceBuffer(),
            split_k,
            M,
            N,
            K,
            stride_a,
            stride_b,
            stride_c,
        };

        // Compute reference result for validation
        compute_reference_gemm(a_m_k, b_k_n, c_m_n_host_result);

        try
        {
            // Configure stream for kernel execution
            ck_tile::stream_config stream_cfg{nullptr, true, 0, 1, 1, true, false, 1};

            // Execute kernel and measure performance
            auto [kernel_name, execution_time] = kernel_func(gemm_args, stream_cfg);

            // Copy result back from device
            c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

            // Validate kernel output against reference
            bool verified_correct =
                verify_results(c_m_n_dev_result, c_m_n_host_result, kernel_name, K, split_k);

            if(verified_correct)
            {
                return std::make_tuple(kernel_name, execution_time, true);
            }
            else
            {
                return std::make_tuple(kernel_name, execution_time, false);
            }
        }
        catch(const std::exception& e)
        {
            ADD_FAILURE() << "        ERROR: " << e.what();
            return std::make_tuple("", 0, false);
        }
    }
};
