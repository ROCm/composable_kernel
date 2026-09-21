// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_blockscale_wp_impl.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

namespace ck {
namespace test {

using Row = ck::tensor_layout::gemm::RowMajor;
using F32 = float;

template <typename Tuple>
class TestGemmBlockscaleWPCommon : public ::testing::Test
{
    protected:
    using ALayout         = std::tuple_element_t<0, Tuple>;
    using BLayout         = std::tuple_element_t<1, Tuple>;
    using CLayout         = Row;
    using A0DataType      = std::tuple_element_t<2, Tuple>;
    using A1DataType      = std::tuple_element_t<3, Tuple>;
    using B0DataType      = std::tuple_element_t<4, Tuple>;
    using B1DataType      = std::tuple_element_t<5, Tuple>;
    using ComputeDataType = std::tuple_element_t<6, Tuple>;
    using CDataType       = std::tuple_element_t<7, Tuple>;

    public:
    static constexpr bool verify_        = true;
    static constexpr int init_method_    = 1;
    static constexpr bool log_           = false;
    static constexpr bool bench_         = false;
    static constexpr index_t ScaleBlockM = 1;
    static constexpr index_t ScaleBlockN = 128;
    static constexpr index_t ScaleBlockK = 128;

    void Run(const int M,
             const int N,
             const int K,
             int n_warmup          = 1,
             int n_iter            = 10,
             int determinism_check = 1,
             bool do_verification  = verify_)
    {
        bool all_success = true;

        int StrideA = std::is_same_v<ALayout, Row> ? K : M;
        int StrideB = std::is_same_v<BLayout, Row> ? N : K;
        int StrideC = std::is_same_v<CLayout, Row> ? N : M;

        all_success =
            all_success &
            ck::profiler::profile_gemm_blockscale_weightpreshuffle_impl<A0DataType,
                                                                        A1DataType,
                                                                        B0DataType,
                                                                        B1DataType,
                                                                        ComputeDataType,
                                                                        F32,
                                                                        CDataType,
                                                                        ScaleBlockM,
                                                                        ScaleBlockN,
                                                                        ScaleBlockK,
                                                                        ALayout,
                                                                        BLayout,
                                                                        CLayout>(do_verification,
                                                                                 init_method_,
                                                                                 log_,
                                                                                 bench_,
                                                                                 M,
                                                                                 N,
                                                                                 K,
                                                                                 StrideA,
                                                                                 StrideB,
                                                                                 StrideC,
                                                                                 n_warmup,
                                                                                 n_iter,
                                                                                 0,
                                                                                 determinism_check,
                                                                                 instance_index);

        EXPECT_TRUE(all_success);
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
