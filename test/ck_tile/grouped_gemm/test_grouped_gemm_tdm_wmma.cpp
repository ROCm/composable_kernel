// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Grouped GEMM with the gfx1250 TDM block pipelines (CompTDMV1 / CompTDMV2) and TdmEpilogue.
// Non-persistent kernel only, k_batch == 1 only.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util.hpp"

using F8    = ck_tile::fp8_t;
using F16   = ck_tile::half_t;
using F32   = float;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using False = ck_tile::bool_constant<false>;
using TDMV1 = GroupedGemmPipelineTypeConstant<GroupedGemmPipelineType::CompTDMV1>;
using TDMV2 = GroupedGemmPipelineTypeConstant<GroupedGemmPipelineType::CompTDMV2>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, Persistent, Pipeline
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,      False,    TDMV1>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,      False,    TDMV1>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,      False,    TDMV1>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,      False,    TDMV1>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,      False,    TDMV2>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,      False,    TDMV2>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,      False,    TDMV2>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,      False,    TDMV2>,
    std::tuple<    Row,     Row,     Row,        F8,        F8,         F32,       F16,      False,    TDMV1>,
    std::tuple<    Col,     Row,     Row,        F8,        F8,         F32,       F16,      False,    TDMV1>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileGroupedGemmTdmWmma : public TestCkTileGroupedGemm<Tuple>
{
    using Base  = TestCkTileGroupedGemm<Tuple>;
    using Param = typename Base::ActiveKernelParam;

#if defined(CK_USE_GFX1250)
    // Same filter the universal TDM GEMM tests apply through check_data_type: the dtype and
    // warp tile combination must have a gfx1250 WMMA instruction.
    static_assert(ck_tile::has_wmma_traits_v<ck_tile::gfx125_t,
                                             typename Base::ADataType,
                                             typename Base::BDataType,
                                             typename Base::AccDataType,
                                             Param::M_Warp_Tile,
                                             Param::N_Warp_Tile,
                                             Param::K_Warp_Tile>,
                  "Unsupported gfx1250 WMMA dtype / warp tile combination");
#endif

    public:
    // No asicRevision == 0 skip, unlike test_gemm_pipeline_util.hpp: that skip exists because
    // the universal TDM tests use cluster launch (multicast), which revision 0 lacks. The
    // grouped TDM kernel static_asserts that cluster launch is off, so the reason does not
    // apply. Add a skip here only if a revision 0 run actually fails.
    // Device run: skip when no gfx125 device is present (host-only cases do not call this).
    void RunOnDevice(const std::vector<int>& Ms,
                     const std::vector<int>& Ns,
                     const std::vector<int>& Ks,
                     const int group_count)
    {
        if(!ck_tile::is_gfx125_supported())
        {
            GTEST_SKIP() << "TDM grouped GEMM requires a gfx125 device.";
        }
        std::vector<int> stride_As(group_count, 0);
        std::vector<int> stride_Bs(group_count, 0);
        std::vector<int> stride_Cs(group_count, 0);
        this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, /*kbatch=*/1, group_count);
    }
};

TYPED_TEST_SUITE(TestCkTileGroupedGemmTdmWmma, KernelTypes);

#define TEST_CKTILE_GGEMM_TDM_SUITE_NAME TestCkTileGroupedGemmTdmWmma

#include "test_grouped_gemm_tdm_ut_cases.inc"

#undef TEST_CKTILE_GGEMM_TDM_SUITE_NAME
