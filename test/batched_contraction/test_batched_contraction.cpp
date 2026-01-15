// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_batched_contraction_multiple_d_impl.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

using F16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

template <typename Tuple>
class TestBatchedContraction : public ::testing::Test
{
    using ADataType    = std::tuple_element_t<0, Tuple>;
    using BDataType    = std::tuple_element_t<1, Tuple>;
    using DsDataType   = std::tuple_element_t<2, Tuple>;
    using EDataType    = std::tuple_element_t<3, Tuple>;
    using AElementOp   = std::tuple_element_t<4, Tuple>;
    using BElementOp   = std::tuple_element_t<5, Tuple>;
    using CDEElementOp = std::tuple_element_t<6, Tuple>;

    static constexpr ck::index_t NumDimG = 1;
    static constexpr ck::index_t NumDimM = 2;
    static constexpr ck::index_t NumDimN = 3;
    static constexpr ck::index_t NumDimK = 1;

    protected:
    struct GemmParams
    {
        std::array<ck::index_t, NumDimG> Gs;
        std::array<ck::index_t, NumDimM> Ms;
        std::array<ck::index_t, NumDimN> Ns;
        std::array<ck::index_t, NumDimK> Ks;
    };

    bool bench_      = true;
    bool verify_     = true;
    bool do_log_     = true;
    int init_method_ = 1;

    std::vector<GemmParams> params;

    void Run()
    {
        bool pass = true;
        for(size_t i = 0; i < params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = params[i];

            pass = pass && ck::profiler::profile_batched_contraction_multiple_d_impl<NumDimG,
                                                                                     NumDimM,
                                                                                     NumDimN,
                                                                                     NumDimK,
                                                                                     ADataType,
                                                                                     BDataType,
                                                                                     DsDataType,
                                                                                     EDataType,
                                                                                     AElementOp,
                                                                                     BElementOp,
                                                                                     CDEElementOp>(
                               verify_,
                               init_method_,
                               do_log_,
                               bench_,
                               param.Gs,
                               param.Ms,
                               param.Ns,
                               param.Ks,
                               instance_index,
                               true);
        }
        EXPECT_TRUE(pass);
    }
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, ck::Tuple<F16>, F16, PassThrough, PassThrough, Add>
>;
// clang-format on

TYPED_TEST_SUITE(TestBatchedContraction, KernelTypes);

TYPED_TEST(TestBatchedContraction, BaseCase)
{
    this->params = std::vector<typename TestFixture::GemmParams>{
        // Gs, Ms, Ns, Ks
        {{1}, {4, 128}, {4, 16, 32}, {256}},
        {{4}, {4, 128}, {4, 16, 32}, {256}},
    };
    this->Run();
}
TYPED_TEST(TestBatchedContraction, TinyCases)
{
    this->params = std::vector<typename TestFixture::GemmParams>{
        // Gs, Ms, Ns, Ks
        {{1}, {1, 16}, {1, 1, 16}, {16}},
        {{2}, {4, 8}, {2, 2, 8}, {32}},
    };
    this->Run();
}
TYPED_TEST(TestBatchedContraction, PadM)
{
    this->params = std::vector<typename TestFixture::GemmParams>{
        // Gs, Ms, Ns, Ks
        {{1}, {1, 130}, {2, 4, 32}, {256}},
    };
    this->Run();
}

// Disabled:  Currently fails on the XDL instances
TYPED_TEST(TestBatchedContraction, DISABLED_PadN)
{
    this->params = std::vector<typename TestFixture::GemmParams>{
        // Gs, Ms, Ns, Ks
        {{1}, {1, 128}, {1, 1, 66}, {256}},
    };
    this->Run();
}

// Disabled: Currently fails on the WMMA and XDL instances
TYPED_TEST(TestBatchedContraction, DISABLED_PadK)
{
    this->params = std::vector<typename TestFixture::GemmParams>{
        // Gs, Ms, Ns, Ks
        {{1}, {1, 128}, {1, 1, 64}, {258}},
    };
    this->Run();
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
