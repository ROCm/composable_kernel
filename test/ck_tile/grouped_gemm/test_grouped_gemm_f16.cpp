// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_util.hpp"

namespace {

using AddressTestShape       = ck_tile::TileGemmShape<ck_tile::sequence<256, 128, 64>,
                                                      ck_tile::sequence<2, 2, 1>,
                                                      ck_tile::sequence<32, 32, 16>>;
using AddressTestPartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<AddressTestShape, 8, 4>;

template <typename ALayout, typename BLayout, bool Persistent = false>
using AddressTestTraits = ck_tile::TileGemmUniversalTraits<false,
                                                           false,
                                                           false,
                                                           false,
                                                           ALayout,
                                                           BLayout,
                                                           ck_tile::tensor_layout::gemm::RowMajor,
                                                           false,
                                                           false,
                                                           Persistent>;

template <typename ALayout, typename BLayout, bool Persistent = false>
using AddressTestProblem =
    ck_tile::UniversalGemmPipelineProblem<ck_tile::half_t,
                                          ck_tile::half_t,
                                          float,
                                          AddressTestShape,
                                          AddressTestTraits<ALayout, BLayout, Persistent>,
                                          ck_tile::GemmPipelineScheduler::Intrawave>;

template <typename ALayout, typename BLayout, bool Persistent = false>
using AddressTestPipeline =
    ck_tile::GemmPipelineAgBgCrCompV3<AddressTestProblem<ALayout, BLayout, Persistent>>;

template <typename ALayout, typename BLayout>
using AddressTestEpilogue = ck_tile::CShuffleEpilogue<
    ck_tile::CShuffleEpilogueProblem<ck_tile::half_t,
                                     ck_tile::half_t,
                                     ck_tile::tuple<>,
                                     float,
                                     ck_tile::half_t,
                                     ck_tile::tuple<>,
                                     ck_tile::tensor_layout::gemm::RowMajor,
                                     ck_tile::element_wise::PassThrough,
                                     256,
                                     128,
                                     2,
                                     2,
                                     32,
                                     32,
                                     16,
                                     false>>;

template <typename ALayout, typename BLayout, bool Persistent = false>
using AddressTestKernel =
    ck_tile::GroupedGemmKernel<AddressTestPartitioner,
                               AddressTestPipeline<ALayout, BLayout, Persistent>,
                               AddressTestEpilogue<ALayout, BLayout>>;

auto MakeAddressTestArg(const ck_tile::index_t m,
                        const ck_tile::index_t n,
                        const ck_tile::index_t k,
                        const ck_tile::index_t stride_a,
                        const ck_tile::index_t stride_b,
                        const ck_tile::index_t stride_c)
{
    return ck_tile::GemmTransKernelArg<>{ck_tile::UniversalGemmKernelArgs<1, 1, 0>{
        {nullptr}, {nullptr}, {}, nullptr, m, n, k, {stride_a}, {stride_b}, {}, stride_c, 1}};
}

TEST(TestCkTileGroupedGemmAddressability, RowMajorMViewIsRebased)
{
    using Kernel = AddressTestKernel<ck_tile::tensor_layout::gemm::RowMajor,
                                     ck_tile::tensor_layout::gemm::ColumnMajor>;
    auto below   = MakeAddressTestArg(524287, 4096, 64, 64, 64, 4096);
    auto exact   = MakeAddressTestArg(524288, 4096, 64, 64, 64, 4096);
    auto above   = MakeAddressTestArg(524289, 4096, 64, 64, 64, 4096);
    EXPECT_TRUE(Kernel::IsGroupedGemmAddressable(below.group_karg));
    EXPECT_TRUE(Kernel::IsGroupedGemmAddressable(exact.group_karg));
    EXPECT_TRUE(Kernel::IsGroupedGemmAddressable(above.group_karg));

    const std::vector<ck_tile::GemmTransKernelArg<>> accepted{std::move(exact)};
    EXPECT_TRUE(Kernel::IsSupportedArgument(accepted));

    using PersistentKernel = AddressTestKernel<ck_tile::tensor_layout::gemm::RowMajor,
                                               ck_tile::tensor_layout::gemm::ColumnMajor,
                                               true>;
    EXPECT_TRUE(PersistentKernel::IsGroupedGemmAddressable(accepted[0].group_karg));
    EXPECT_TRUE(PersistentKernel::IsSupportedArgument(accepted));
}

TEST(TestCkTileGroupedGemmAddressability, UnsafeFullViewsAreRejected)
{
    using UnsafeA = AddressTestKernel<ck_tile::tensor_layout::gemm::ColumnMajor,
                                      ck_tile::tensor_layout::gemm::ColumnMajor>;
    auto unsafe_a = MakeAddressTestArg(524288, 128, 4096, 524288, 4096, 128);
    EXPECT_FALSE(UnsafeA::IsGroupedGemmAddressable(unsafe_a.group_karg));
    const std::vector<ck_tile::GemmTransKernelArg<>> rejected{std::move(unsafe_a)};
    EXPECT_FALSE(UnsafeA::IsSupportedArgument(rejected));

    using UnsafeB = AddressTestKernel<ck_tile::tensor_layout::gemm::RowMajor,
                                      ck_tile::tensor_layout::gemm::RowMajor>;
    auto unsafe_b = MakeAddressTestArg(128, 4096, 524288, 524288, 4096, 4096);
    EXPECT_FALSE(UnsafeB::IsGroupedGemmAddressable(unsafe_b.group_karg));

    auto safe_boundary = MakeAddressTestArg(128, 4096, 524287, 524287, 4096, 4096);
    EXPECT_TRUE(UnsafeB::IsGroupedGemmAddressable(safe_boundary.group_karg));
}

} // namespace

using F8    = ck_tile::fp8_t;
using F16   = ck_tile::half_t;
using F32   = float;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using True  = ck_tile::bool_constant<true>;
using False = ck_tile::bool_constant<false>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, Persistent
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       True>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,      False>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,       True>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,      False>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,       True>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,      False>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,       True>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,      False>
    >;
// clang-format on

template <typename Tuple>
class TestCkTileGroupedGemmF16 : public TestCkTileGroupedGemm<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGroupedGemmF16, KernelTypes);

#define TEST_CKTILE_GGEMM_SUITE_NAME TestCkTileGroupedGemmF16

#include "test_grouped_gemm_ut_cases.inc"
