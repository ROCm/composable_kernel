// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_kernel_validation_base.hpp"
#include "gemm_common_fp16.hpp"
#include "gemm_dispatcher_fp16.hpp"

using FP16TestBase = GemmKernelValidationTestBase<ck_tile::fp16_t,
                                                  ck_tile::fp16_t,
                                                  ck_tile::fp16_t,
                                                  float,
                                                  ck_tile::tensor_layout::gemm::RowMajor,
                                                  ck_tile::tensor_layout::gemm::ColumnMajor,
                                                  ck_tile::tensor_layout::gemm::RowMajor,
                                                  GemmDispatcher>;

class GemmKernelValidationTestFP16 : public FP16TestBase
{
};

TEST_F(GemmKernelValidationTestFP16, AllKernelConfigurations)
{
    for(bool structured_sparsity : {false, true})
    {
        std::cout << "\n=== Testing FP16 kernels with structured_sparsity = " << structured_sparsity
                  << " ===\n";
        GemmKernelValidationTestFP16::InitDispatcher(structured_sparsity);

        auto* kernel_map_ = this->kernel_map_;
        if(kernel_map_->empty())
            GTEST_FAIL()
                << "No kernels found in dispatcher! Check if libraries are properly linked.";

        int total_kernels  = 0;
        int passed_kernels = 0;
        for(const auto& [trait_name, kernels] : *kernel_map_)
        {
            std::cout << "\nTesting trait: " << trait_name << " (" << kernels.size() << " variants)"
                      << std::endl;
            for(size_t i = 0; i < kernels.size(); ++i)
            {
                for(const auto& prob : kTestProblems)
                {
                    std::cout << "  Testing kernel variant " << i << " with M=" << prob.M
                              << " N=" << prob.N << " K=" << prob.K << " split_k=" << prob.split_k
                              << std::endl;
                    if(test_single_problem(kernels[i], prob.M, prob.N, prob.K, prob.split_k))
                        passed_kernels++;
                    total_kernels++;
                }
            }
        }
        std::cout << "\n=== FP16 SUMMARY ===\nTotal kernels tested: " << total_kernels
                  << "\nPassed: " << passed_kernels
                  << "\nFailed: " << (total_kernels - passed_kernels) << std::endl;
        EXPECT_EQ(passed_kernels, total_kernels) << "Some FP16 kernels failed verification";
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "=== CK Tile GEMM Kernel Instance Validation (FP16) ===" << std::endl;
    std::cout << "Testing all FP16 kernel configurations generated from JSON config" << std::endl;
    return RUN_ALL_TESTS();
}
