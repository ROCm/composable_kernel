// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"

namespace {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

struct MockBlockGemmShape
{
    [[maybe_unused]] static constexpr ck_tile::index_t kclusterM = 1;
    [[maybe_unused]] static constexpr ck_tile::index_t kclusterN = 1;
    [[maybe_unused]] static constexpr ck_tile::index_t kclusterK = 1;
    static constexpr bool PermuteA                               = false;
    static constexpr bool PermuteB                               = false;

    struct WarpTile
    {
        template <typename Index>
        static constexpr ck_tile::index_t at(Index)
        {
            return 16;
        }
    };
};

struct MockTilePartitioner
{
    using BlockGemmShape = MockBlockGemmShape;

    static constexpr ck_tile::index_t MPerBlock = 64;
    static constexpr ck_tile::index_t NPerBlock = 64;
    static constexpr ck_tile::index_t KPerBlock = 64;
};

struct MockGemmPipeline
{
    // UniversalGemmKernel also consumes these scalar aliases when it forms
    // tensor descriptors, even though this test only reaches validation.
    using ADataType      = float;
    using BDataType      = float;
    using ALayout        = Row;
    using BLayout        = Col;
    using AsDataType     = ck_tile::tuple<float>;
    using BsDataType     = ck_tile::tuple<float>;
    using AsLayout       = ck_tile::tuple<Row>;
    using BsLayout       = ck_tile::tuple<Col>;
    using CLayout        = Row;
    using AElementWise   = ck_tile::element_wise::PassThrough;
    using BElementWise   = ck_tile::element_wise::PassThrough;
    using BlockGemmShape = MockBlockGemmShape;

    [[maybe_unused]] static constexpr ck_tile::index_t BlockSize = 64;
    static constexpr bool kPadM                                  = false;
    static constexpr bool kPadN                                  = false;
    static constexpr bool kPadK                                  = false;
    static constexpr bool Preshuffle                             = false;

    template <bool>
    static constexpr ck_tile::index_t GetVectorSizeA()
    {
        return 1;
    }

    template <bool>
    static constexpr ck_tile::index_t GetVectorSizeB()
    {
        return 1;
    }
};

struct MockEpiloguePipeline
{
    using DsDataType = ck_tile::tuple<float>;
    using DsLayout   = ck_tile::tuple<Col>;
    using ODataType  = float;

    static constexpr ck_tile::index_t GetVectorSizeC() { return 1; }

    template <typename Index>
    static constexpr ck_tile::index_t GetVectorSizeD(Index)
    {
        return 1;
    }
};

using Kernel =
    ck_tile::UniversalGemmKernel<MockTilePartitioner, MockGemmPipeline, MockEpiloguePipeline>;

TEST(UniversalGemmArgumentValidation, ReportsMismatchedDTensorLayout)
{
    Kernel::KernelArgs args{};
    args.M       = 64;
    args.N       = 64;
    args.K       = 64;
    args.k_batch = 1;

    ck_tile::UpdateEnvVar(CK_TILE_ENV(CK_TILE_LOGGING), true);
    testing::internal::CaptureStderr();
    const bool supported         = Kernel::IsSupportedArgument(args);
    const std::string diagnostic = testing::internal::GetCapturedStderr();
    ck_tile::EnvUnset(CK_TILE_ENV(CK_TILE_LOGGING));

    EXPECT_FALSE(supported);
    EXPECT_NE(diagnostic.find("D tensor layout must match the C layout: D0"), std::string::npos)
        << diagnostic;
    EXPECT_NE(diagnostic.find("D0 is ColumnMajor, but C is RowMajor"), std::string::npos)
        << diagnostic;
}

} // namespace
