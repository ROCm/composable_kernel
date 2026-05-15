// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "profiler/profile_batched_gemm_multiple_d_gemm_multiple_d_impl.hpp"

using ck::tensor_operation::device::GemmSpecialization;
static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;
template <ck::index_t N>
using I = ck::Number<N>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
struct BaseTestBatchedGemmMultipleDGemmMultipleD : public ::testing::Test
{
    using ADataType     = std::tuple_element_t<0, Tuple>;
    using B0DataType    = std::tuple_element_t<1, Tuple>;
    using D0sDataType   = std::tuple_element_t<2, Tuple>;
    using B1DataType    = std::tuple_element_t<3, Tuple>;
    using D1sDataType   = std::tuple_element_t<4, Tuple>;
    using EDataType     = std::tuple_element_t<5, Tuple>;
    using ALayout       = std::tuple_element_t<6, Tuple>;
    using B0Layout      = std::tuple_element_t<7, Tuple>;
    using D0sLayout     = std::tuple_element_t<8, Tuple>;
    using B1Layout      = std::tuple_element_t<9, Tuple>;
    using D1sLayout     = std::tuple_element_t<10, Tuple>;
    using ELayout       = std::tuple_element_t<11, Tuple>;
    using A0ElementOp   = std::tuple_element_t<12, Tuple>;
    using B0ElementOp   = std::tuple_element_t<13, Tuple>;
    using CDE0ElementOp = std::tuple_element_t<14, Tuple>;
    using B1ElementOp   = std::tuple_element_t<15, Tuple>;
    using CDE1ElementOp = std::tuple_element_t<16, Tuple>;

    std::vector<std::vector<int>> lengths_ = {
        {256, 256, 64, 64, 4},
        {256, 256, 128, 128, 4},
        {512, 512, 64, 64, 2},
        {512, 512, 128, 128, 2},
        {1024, 1024, 64, 64, 1},
        {1024, 1024, 128, 128, 1},
    };
    bool bench_  = false;
    bool verify_ = true;

    void RunSingle(int M, int N, int K, int O, int BatchCount)
    {
        // WMMA instances are setup to support all the test cases
        // XDL instances are not.
        bool fail_if_no_supported_instances = ck::is_gfx11_supported() || ck::is_gfx12_supported();

        bool pass =
            ck::profiler::profile_batched_gemm_multiple_d_gemm_multiple_d_impl<ALayout,
                                                                               B0Layout,
                                                                               D0sLayout,
                                                                               B1Layout,
                                                                               D1sLayout,
                                                                               ELayout,
                                                                               ADataType,
                                                                               B0DataType,
                                                                               D0sDataType,
                                                                               B1DataType,
                                                                               D1sDataType,
                                                                               EDataType,
                                                                               A0ElementOp,
                                                                               B0ElementOp,
                                                                               CDE0ElementOp,
                                                                               B1ElementOp,
                                                                               CDE1ElementOp>(
                verify_,
                1,
                false,
                bench_,
                M,
                N,
                K,
                O,
                BatchCount,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                fail_if_no_supported_instances,
                instance_index);

        EXPECT_TRUE(pass);
    }

    void Run()
    {
        for(auto lengths : this->lengths_)
        {
            int M          = lengths[0];
            int N          = lengths[1];
            int K          = lengths[2];
            int O          = lengths[3];
            int BatchCount = lengths[4];

            this->RunSingle(M, N, K, O, BatchCount);
        }
    }
};

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
