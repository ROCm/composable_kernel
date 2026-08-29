// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/utility/tuple.hpp"
#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "example/68_gemm_add/common.hpp"
#include "test_grouped_gemm_tile_loop_util.hpp"

ck::index_t param_mask     = 0xffffff;
ck::index_t instance_index = -1;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F8   = ck::f8_t;
using I8   = int8_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough         = ck::tensor_operation::element_wise::PassThrough;
using Multiply            = ck::tensor_operation::element_wise::Multiply;
using MultiplyAdd         = ck::tensor_operation::element_wise::MultiplyAdd;
using MultiplyAddFastGelu = ck::tensor_operation::element_wise::MultiplyAddFastGelu;
using MultiplyFastGelu    = ck::tensor_operation::element_wise::MultiplyFastGelu;

template <typename Tuple>
class TestGroupedGemmTileLoop : public ck::test::TestGroupedGemmTileLoop<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    ck::Tuple<Row, Row, ck::Tuple<Row>, Row, BF16, I8, ck::Tuple<BF16>, BF16, PassThrough, PassThrough, Multiply>,
    ck::Tuple<Row, Row, ck::Tuple<Row, Row>, Row, BF16, I8, ck::Tuple<BF16, BF16>, BF16, PassThrough, PassThrough, MultiplyAdd>,
    ck::Tuple<Row, Row, ck::Tuple<Row, Row>, Row, BF16, I8, ck::Tuple<BF16, BF16>, BF16, PassThrough, PassThrough, MultiplyAddFastGelu>,
    ck::Tuple<Row, Row, ck::Tuple<Row>, Row, BF16, I8, ck::Tuple<BF16>, BF16, PassThrough, PassThrough, MultiplyFastGelu>
    >;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemmTileLoop, KernelTypes);

#include "test_grouped_gemm_tile_loop_ut_cases.inc"
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
