// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/number.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_mx.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_mx.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "profiler/profile_gemm_mx_impl.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

namespace ck {
namespace test {

namespace {
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
} // namespace

template <typename Tuple>
class TestGemmMX : public testing::Test
{
    using Row       = ck::tensor_layout::gemm::RowMajor;
    using F32       = float;
    using ScaleType = e8m0_bexp_t;

    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = Row;
    using ADataType   = std::tuple_element_t<2, Tuple>;
    using BDataType   = std::tuple_element_t<3, Tuple>;
    using CDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = float;

    public:
    static constexpr index_t ScaleBlockSize = std::tuple_element_t<5, Tuple>{};
    static constexpr bool verify_           = true;
    static constexpr int init_method_       = 2; // decimal value initialization
    static constexpr bool log_              = false;
    static constexpr bool bench_            = false; // measure kernel performance
    std::vector<int> k_batches_;

    void SetUp() override { k_batches_ = {1}; }

    void Run(const int M,
             const int N,
             const int K,
             const int StrideA,
             const int StrideB,
             const int StrideC)
    {
        for(auto kb : k_batches_)
        {
            RunSingle(M, N, K, StrideA, StrideB, StrideC, kb);
        }
    }

    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   int kbatch   = 1,
                   int n_warmup = 10,
                   int n_iter   = 10)
    {
        bool pass = ck::profiler::profile_gemm_mx_impl<ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       ScaleBlockSize>(verify_,
                                                                       init_method_,
                                                                       log_,
                                                                       bench_,
                                                                       M,
                                                                       N,
                                                                       K,
                                                                       StrideA,
                                                                       StrideB,
                                                                       StrideC,
                                                                       kbatch,
                                                                       n_warmup,
                                                                       n_iter,
                                                                       0,
                                                                       instance_index);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck
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
