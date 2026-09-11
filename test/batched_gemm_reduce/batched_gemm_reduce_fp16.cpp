// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_batched_gemm_reduce_impl.hpp"

static ck::index_t param_mask = 0xffff;
struct GemmParams
{
    ck::index_t M;
    ck::index_t N;
    ck::index_t K;
    ck::index_t BatchCount;
};

class TestBatchedGemmReduce : public ::testing::Test
{
    protected:
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    std::vector<GemmParams> params;

    bool Run()
    {
        bool pass = true;
        for(size_t i = 0; i < params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            const auto& param     = params[i];
            const auto M          = param.M;
            const auto N          = param.N;
            const auto K          = param.K;
            const auto BatchCount = param.BatchCount;

            pass = pass && ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                                          ck::half_t,
                                                                          ck::half_t,
                                                                          float,
                                                                          Row,
                                                                          Row,
                                                                          Row>(
                               true, 1, false, false, M, N, K, K, N, N, BatchCount);

            pass = pass && ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                                          ck::half_t,
                                                                          ck::half_t,
                                                                          float,
                                                                          Row,
                                                                          Col,
                                                                          Row>(
                               true, 1, false, false, M, N, K, K, K, N, BatchCount);

            pass = pass && ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                                          ck::half_t,
                                                                          ck::half_t,
                                                                          float,
                                                                          Col,
                                                                          Row,
                                                                          Row>(
                               true, 1, false, false, M, N, K, M, N, N, BatchCount);

            pass = pass && ck::profiler::profile_batched_gemm_reduce_impl<ck::half_t,
                                                                          ck::half_t,
                                                                          ck::half_t,
                                                                          float,
                                                                          Col,
                                                                          Col,
                                                                          Row>(
                               true, 1, false, false, M, N, K, M, K, N, BatchCount);
        }
        return pass;
    }
};

#ifdef CK_ENABLE_FP16
TEST_F(TestBatchedGemmReduce, fp16)
{
    this->params.push_back({64, 64, 64, 2});
    this->params.push_back({64, 64, 64, 1});
    this->params.push_back({40, 40, 40, 2});
    this->params.push_back({256, 256, 128, 3});

    // Tests with larger MNK
    this->params.push_back({512, 256, 128, 1});
    this->params.push_back({256, 240, 192, 2});
    this->params.push_back({256, 256, 128, 3});
    this->params.push_back({240, 128, 128, 5});
    EXPECT_TRUE(this->Run());
}
#endif

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 2)
    {
        param_mask = strtol(argv[1], nullptr, 0);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1: param_mask " << std::endl;
    }
    return RUN_ALL_TESTS();
}
