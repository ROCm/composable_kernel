// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "gtest/gtest.h"

TYPED_TEST_SUITE(TestCkTileGemmPipelineUniversal, UniversalTestTypes);

TYPED_TEST(TestCkTileGemmPipelineUniversal, GemmTest)
{
    // Define possible values for each parameter
    std::vector<int> m_values = {512, 1024};
    std::vector<int> n_values = {512, 2048};
    std::vector<int> k_values = {512, 1024};

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
