// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include <tuple>
#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"

#include "gtest/gtest.h"
#include "test_grouped_gemm_util.hpp"

ck::index_t param_mask     = 0xffffff;
ck::index_t instance_index = -1;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F8   = ck::f8_t;
using I8   = int8_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
class TestGroupedGemm : public ck::test::TestGroupedGemm<Tuple>
{
    public:
    void SetUp() override
    {
        ck::test::TestGroupedGemm<Tuple>::SetUp();

#if defined(CK_USE_WMMA)
        // The old XDL tests didn't fail if instances were not supported, so we want to keep that
        // behaviour. When compiling WMMA instances and WMMA is supported, then we'll fail if a
        // specific case is not supported
        this->fail_if_no_supported_instances_ =
            ck::is_gfx11_supported() || ck::is_gfx12_supported();
#endif
    }
};

// clang-format off
using KernelTypes = ::testing::Types<

#if defined(CK_USE_WMMA)
    // WWMA only. No reason to not have it for XDL, but the instance was not defined and it was not in the original test.
    ck::Tuple<     Col, Col, Row, BF16, BF16, BF16>,
#endif
    
#if defined(CK_USE_XDL) && !defined(CK_USE_WMMA)
    // XDL only at the moment, instances for WMMA not defined
    // (And XDL instances don't run on gfx11/12, so we conditionally keep them out)
    ck::Tuple<     Row, Row, Row, BF16, I8, BF16>,
    ck::Tuple<     Row, Col, Row, BF16, I8, BF16>,
#endif

#if CK_USE_OCP_FP8 || CK_USE_FNUZ_FP8 || defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || defined(CK_USE_WMMA_FP8)
    // FP8 instances. Unfortunately CK_ENABLE_FP8 is always defined when not explicitly disabled, even if FP8 is 
    // not supported for any included architecture.
    ck::Tuple<     Row, Row, Row, F8, F16, F16>,
    ck::Tuple<     Row, Row, Row, F16, F8, F16>,
#endif

    ck::Tuple<     Row, Row, Row, F16, F16, F16>,
    ck::Tuple<     Row, Col, Row, F16, F16, F16>,
    ck::Tuple<     Col, Row, Row, F16, F16, F16>,
    ck::Tuple<     Col, Col, Row, F16, F16, F16>,

    ck::Tuple<     Row, Row, Row, BF16, BF16, BF16>,
    ck::Tuple<     Row, Col, Row, BF16, BF16, BF16>,
    ck::Tuple<     Col, Row, Row, BF16, BF16, BF16>
    >;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemm, KernelTypes);

#include "test_grouped_gemm_ut_cases.inc"
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
