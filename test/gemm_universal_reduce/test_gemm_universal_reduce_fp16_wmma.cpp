// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "gtest/gtest.h"
#include "profiler/profile_gemm_universal_reduce_impl.hpp"
static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;
TEST(GemmUniversalReduce, FP16)
{
    using Row = ck::tensor_layout::gemm::RowMajor;

    int M      = 512;
    int N      = 256;
    int K      = 128;
    int KBatch = 1;

    bool pass = true;

    pass = pass && ck::profiler::profile_gemm_universal_reduce_impl<ck::half_t,
                                                                    ck::half_t,
                                                                    ck::Tuple<>,
                                                                    float,
                                                                    ck::half_t,
                                                                    Row,
                                                                    Row,
                                                                    ck::Tuple<>,
                                                                    Row>(
                       true, 1, false, false, M, N, K, K, N, N, KBatch, 1, 10, 0, instance_index);
    EXPECT_TRUE(pass);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
