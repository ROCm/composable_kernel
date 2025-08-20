// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_kernel_validation_base.hpp"
#include "gemm_common.hpp"
#include "gemm_dispatcher.hpp"

using DispatcherTypesBF16 = ::testing::Types<GemmDispatcher>;

// Base template class for BF16 GEMM validation
template <typename DispatcherType, bool StructuredSparsity>
class GemmKernelValidationTestBF16Base
    : public GemmKernelValidationTestBase<ck_tile::bf16_t,
                                          ck_tile::bf16_t,
                                          ck_tile::bf16_t,
                                          float,
                                          ck_tile::tensor_layout::gemm::RowMajor,
                                          ck_tile::tensor_layout::gemm::ColumnMajor,
                                          ck_tile::tensor_layout::gemm::RowMajor,
                                          DispatcherType>
{
    protected:
    static constexpr bool structured_sparsity = StructuredSparsity;

    void RunAllKernelConfigurations()
    {
        int total_kernels  = 0;
        int passed_kernels = 0;

        // Initialize dispatcher with compile-time sparsity mode
        this->InitDispatcher(structured_sparsity);

        auto* kernel_map_ = this->kernel_map_;
        ASSERT_FALSE(kernel_map_->empty())
            << "No kernels found in dispatcher! Check if libraries are properly linked.";

        // Test all problem configurations
        for(const auto& prob : kTestProblems)
        {
            SCOPED_TRACE("Problem: M=" + std::to_string(prob.M) + " N=" + std::to_string(prob.N) +
                         " K=" + std::to_string(prob.K) +
                         " split_k=" + std::to_string(prob.split_k));

            // Create problem context with pre-allocated data
            auto problem_context = this->create_problem_context(
                prob.M, prob.N, prob.K, prob.split_k, structured_sparsity);

            // Test all kernel variants for this problem
            for(const auto& [trait_name, kernels] : *kernel_map_)
            {
                SCOPED_TRACE("Testing trait: " + trait_name +
                             " structured_sparsity=" + (structured_sparsity ? "true" : "false"));

                // Test each kernel implementation
                for(size_t i = 0; i < kernels.size(); ++i)
                {
                    SCOPED_TRACE("Kernel index: " + std::to_string(i));

                    // Execute kernel and validate results
                    auto [kernel_name, elapsed_time, is_valid] =
                        this->test_single_kernel(kernels[i], *problem_context);

                    total_kernels++;

                    if(is_valid)
                    {
                        passed_kernels++;
                        // Test passed - GoogleTest will handle success reporting
                    }
                    else
                    {
                        // Fail immediately with detailed error information
                        FAIL() << "Kernel validation failed: " << kernel_name << " M=" << prob.M
                               << " N=" << prob.N << " K=" << prob.K << " split_k=" << prob.split_k
                               << " structured_sparsity="
                               << (structured_sparsity ? "true" : "false");
                    }
                }
            }
        }

        // Final assertions for test completion
        EXPECT_EQ(passed_kernels, total_kernels)
            << "Not all kernels passed validation. Passed: " << passed_kernels << "/"
            << total_kernels;
        EXPECT_GT(total_kernels, 0) << "No kernels were found to test";
    }
};

// Test class for dense GEMM kernels
template <typename DispatcherType>
class GemmKernelValidationTestBF16Dense
    : public GemmKernelValidationTestBF16Base<DispatcherType, false>
{
};

// Test class for structured sparse GEMM kernels
template <typename DispatcherType>
class GemmKernelValidationTestBF16Sparse
    : public GemmKernelValidationTestBF16Base<DispatcherType, true>
{
};

// Test suites for both dense and sparse variants
TYPED_TEST_SUITE(GemmKernelValidationTestBF16Dense, DispatcherTypesBF16);
TYPED_TEST_SUITE(GemmKernelValidationTestBF16Sparse, DispatcherTypesBF16);

// Test all dense GEMM kernel configurations
TYPED_TEST(GemmKernelValidationTestBF16Dense, AllKernelConfigurations)
{
    this->RunAllKernelConfigurations();
}

// Test all structured sparse GEMM kernel configurations
TYPED_TEST(GemmKernelValidationTestBF16Sparse, AllKernelConfigurations)
{
    this->RunAllKernelConfigurations();
}
