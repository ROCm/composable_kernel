// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "gtest/gtest.h"

TYPED_TEST_SUITE(TestCkTileGemmPipelineBasic, BasicTestTypes);

TYPED_TEST(TestCkTileGemmPipelineBasic, GemmTest)
{
    // Define possible values for each parameter
    std::vector<int> m_values = {128, 1024};
    std::vector<int> n_values = {128, 2048};
    std::vector<int> k_values = {64, 128};

    for(const auto& m : m_values)
    {
        for(const auto& n : n_values)
        {
            for(const auto& k : k_values)
            {
                this->run_gemm_combinations(m, n, k);
            }
        }
    }
}
