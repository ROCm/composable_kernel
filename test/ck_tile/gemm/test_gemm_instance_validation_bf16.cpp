// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_kernel_validation_base.hpp"
#include "gemm_common_bf16.hpp"
#include "gemm_dispatcher_bf16.hpp"

using DispatcherTypesBF16 = ::testing::Types<GemmDispatcher>;

template <typename DispatcherType>
class GemmKernelValidationTestBF16
    : public GemmKernelValidationTestBase<ck_tile::bf16_t,
                                          ck_tile::bf16_t,
                                          ck_tile::bf16_t,
                                          float,
                                          ck_tile::tensor_layout::gemm::RowMajor,
                                          ck_tile::tensor_layout::gemm::ColumnMajor,
                                          ck_tile::tensor_layout::gemm::RowMajor,
                                          DispatcherType>
{
};

TYPED_TEST_SUITE(GemmKernelValidationTestBF16, DispatcherTypesBF16);

TYPED_TEST(GemmKernelValidationTestBF16, AllKernelConfigurations)
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
            for(size_t i = 0; i < kernels.size(); ++i)
            {
                for(const auto& prob : kTestProblems)
                {
                    if(this->test_single_problem(kernels[i], prob.M, prob.N, prob.K, prob.split_k))
                        passed_kernels++;
                    total_kernels++;
                }
            }
        }
        SCOPED_TRACE(::testing::Message() << "Passed Kernels: " << passed_kernels
                                          << ", Total Kernels: " << total_kernels);
        EXPECT_EQ(passed_kernels, total_kernels) << "Some BF16 kernels failed verification";
    }
}
