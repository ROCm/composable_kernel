// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_kernel_validation_base.hpp"
#include "gemm_common.hpp"
#include "gemm_dispatcher.hpp"

using DispatcherTypesFP16 = ::testing::Types<GemmDispatcher>;

template <typename DispatcherType>
class GemmKernelValidationTestFP16
    : public GemmKernelValidationTestBase<ck_tile::fp16_t,
                                          ck_tile::fp16_t,
                                          ck_tile::fp16_t,
                                          float,
                                          ck_tile::tensor_layout::gemm::RowMajor,
                                          ck_tile::tensor_layout::gemm::ColumnMajor,
                                          ck_tile::tensor_layout::gemm::RowMajor,
                                          DispatcherType>
{
};

TYPED_TEST_SUITE(GemmKernelValidationTestFP16, DispatcherTypesFP16);

TYPED_TEST(GemmKernelValidationTestFP16, AllKernelConfigurations)
{
    int total_kernels  = 0;
    int passed_kernels = 0;

    for(bool structured_sparsity : {false, true})
    {
        this->InitDispatcher(structured_sparsity);

        auto* kernel_map_ = this->kernel_map_;
        ASSERT_FALSE(kernel_map_->empty())
            << "No kernels found in dispatcher! Check if libraries are properly linked.";

        for(const auto& [trait_name, kernels] : *kernel_map_)
        {
            SCOPED_TRACE("Testing trait: " + trait_name +
                         " structured_sparsity=" + (structured_sparsity ? "true" : "false"));

            for(size_t i = 0; i < kernels.size(); ++i)
            {
                SCOPED_TRACE("Kernel index: " + std::to_string(i));

                for(const auto& prob : kTestProblems)
                {
                    SCOPED_TRACE("Problem: M=" + std::to_string(prob.M) +
                                 " N=" + std::to_string(prob.N) + " K=" + std::to_string(prob.K) +
                                 " split_k=" + std::to_string(prob.split_k));

                    auto [kernel_name, elapsed_time, is_valid] = this->test_single_problem(
                        kernels[i], prob.M, prob.N, prob.K, prob.split_k, structured_sparsity);

                    total_kernels++;

                    if(is_valid)
                    {
                        passed_kernels++;
                        std::cout << "PASS: " << kernel_name << " M=" << prob.M << " N=" << prob.N
                                  << " K=" << prob.K << " split_k=" << prob.split_k
                                  << " structured_sparsity="
                                  << (structured_sparsity ? "true" : "false")
                                  << " elapsed_time=" << elapsed_time << " ms" << std::endl;
                    }
                    else
                    {
                        std::cerr << "FAIL: " << kernel_name
                                  << " verification failed for M=" << prob.M << " N=" << prob.N
                                  << " K=" << prob.K << " split_k=" << prob.split_k
                                  << " structured_sparsity="
                                  << (structured_sparsity ? "true" : "false") << std::endl;

                        // Fail immediately - all kernels must pass
                        FAIL() << "Kernel validation failed: " << kernel_name << " M=" << prob.M
                               << " N=" << prob.N << " K=" << prob.K << " split_k=" << prob.split_k
                               << " structured_sparsity="
                               << (structured_sparsity ? "true" : "false");
                    }
                }
            }
        }
    }

    std::cout << "All kernels passed validation: " << passed_kernels << "/" << total_kernels
              << std::endl;

    // Require 100% pass rate - all kernels must work correctly
    EXPECT_EQ(passed_kernels, total_kernels)
        << "Not all kernels passed validation. Expected: " << total_kernels
        << ", Actual: " << passed_kernels;

    // Additional check to ensure we actually tested some kernels
    EXPECT_GT(total_kernels, 0) << "No kernels were found to test";
}
