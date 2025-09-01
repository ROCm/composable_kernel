#include <gtest/gtest.h>

#include <ck_tile/builder/conv_builder.hpp>

namespace {

namespace ckb = ck_tile::builder;

struct FwdConvSignature
{
    static constexpr int SPATIAL_DIM = 2;
    static constexpr auto DIRECTION  = ckb::ConvDirection::Forward;
    static constexpr auto LAYOUT     = ckb::GroupConvLayout::NHWGC_GKYXC_NHWGK;
    static constexpr auto DATA_TYPE  = ckb::DataType::FP16;
};

TEST(ConvBuilderTest, TestSignature)
{
    static_assert(ckb::ConvSignature<FwdConvSignature>);
    SUCCEED();
}

struct FwdConvAlgorithm
{
    // TODO: Add algorithm info.
};

TEST(ConvBuilderTest, TestAlgorithm)
{
    static_assert(ckb::ConvAlgorithm<FwdConvAlgorithm>);
    SUCCEED();
}

static constexpr char API_VERSION[] = "0.1.0";
using FwdConvBuilder = ckb::ConvBuilder<FwdConvSignature, FwdConvAlgorithm, API_VERSION>;

TEST(ConvBuilderTest, TestInstance)
{
    EXPECT_EQ(
        FwdConvBuilder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}

struct ConvFwdXdlBf16CompInstances2xAlgorithm0
{
    static constexpr ckb::ThreadBlock THREAD_BLOCK{
        .block_size = 256,
        .sub_matrix = {.m = 256, .n = 256, .k = 32},
    };
};

TEST(ConvBuilderTest, TestInstance0)
{
    using Builder =
        ckb::ConvBuilder<FwdConvSignature, ConvFwdXdlBf16CompInstances2xAlgorithm0, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}

} // namespace
