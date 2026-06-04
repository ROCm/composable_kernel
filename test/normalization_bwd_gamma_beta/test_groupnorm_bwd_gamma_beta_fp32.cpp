// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "profiler/profile_groupnorm_bwd_gamma_beta_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;
static ck::index_t param_mask [[maybe_unused]] = 0xffff;
static ck::index_t instance_index              = -1;
template <typename Tuple>
class TestgroupnormBwdGammaBeta : public ::testing::Test
{
    protected:
    using DYDataType         = std::tuple_element_t<0, Tuple>;
    using XDataType          = std::tuple_element_t<1, Tuple>;
    using MeanInvStdDataType = std::tuple_element_t<2, Tuple>;
    using ComputeDataType    = std::tuple_element_t<3, Tuple>;
    using DGammaDataType     = std::tuple_element_t<4, Tuple>;
    using DBetaDataType      = std::tuple_element_t<5, Tuple>;

    void Run()
    {
        // Bwd data: [N, H, W, G, C], reduce H, W, C
        std::vector<std::vector<ck::index_t>> lengths = {{1, 1, 1, 1, 1},
                                                         {1, 2, 3, 4, 5},
                                                         {256, 9, 9, 9, 9},
                                                         {1, 64, 64, 32, 10},
                                                         {1, 32, 32, 32, 20},
                                                         {1, 16, 16, 32, 40}};

        for(auto length : lengths)
        {
            bool success = ck::profiler::profile_groupnorm_bwd_gamma_beta_impl<DYDataType,
                                                                               XDataType,
                                                                               MeanInvStdDataType,
                                                                               ComputeDataType,
                                                                               DGammaDataType,
                                                                               DBetaDataType>(
                true, 2, false, false, length, instance_index);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<
    // DYDataType XDataType, MeanInvStdDataType, ComputeDataType, DGammaDataType, DBetaDataType>
    std::tuple<F32, F32, F32, F32, F32, F32>>;

TYPED_TEST_SUITE(TestgroupnormBwdGammaBeta, KernelTypes);
TYPED_TEST(TestgroupnormBwdGammaBeta, Test_FP32) { this->Run(); }

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
