// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "test_grouped_gemm_util.hpp"

ck::index_t param_mask     = 0xffffff;
ck::index_t instance_index = -1;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F8   = ck::f8_t;
using I8   = int8_t;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::PassThrough;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
class TestGroupedGemm : public ck::test::TestGroupedGemm<Tuple,false,ck::test::FixedNKGroupedGemmProfiler>
{
    public:
    void SetUp() override
    {
        ck::test::TestGroupedGemm<Tuple,false,ck::test::FixedNKGroupedGemmProfiler>::SetUp();

#if defined(CK_USE_WMMA)
        // The old XDL tests didn't fail if instances were not supported, so we want to keep that
        // behaviour When compiling WMMA instances and WMMA is supported, then we'll fail if a
        // specific case is not supported
        this->fail_if_no_supported_instances_ =
            ck::is_gfx11_supported() || ck::is_gfx12_supported();
#endif
    }
};


using KernelTypes = ::testing::Types<
    
#if defined(CK_USE_XDL) && defined(__gfx9__)
    // XDL only at the moment, instances for WMMA not defined
    std::tuple<     Row, Row, Row, BF16, I8, BF16>,
    std::tuple<     Row, Col, Row, BF16, I8, BF16>,
#endif

#if (defined(CK_USE_XDL) && (defined(__gfx9__) || defined(__gfx12__))) || (defined(CK_USE_WMMA) && defined(__gfx12__))
    std::tuple<     Row, Row, Row, F16, F8, F16>,
    std::tuple<     Row, Col, Row, F16, F8, F16>,
#endif

    std::tuple<     Row, Row, Row, F16, F16, F16>,
    std::tuple<     Row, Col, Row, F16, F16, F16>,


    std::tuple<     Row, Row, Row, BF16, BF16, BF16>,
    std::tuple<     Row, Col, Row, BF16, BF16, BF16>,

    std::tuple<Row, Row, Row, F16, I8, F16>,
    std::tuple<Row, Col, Row, F16, I8, F16>
    >;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemm, KernelTypes);

#include "test_grouped_gemm_fixed_nk_cases.inc"
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
