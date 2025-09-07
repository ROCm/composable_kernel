#include <gtest/gtest.h>

#include <ck_tile/builder/conv_builder.hpp>

namespace {

namespace ckb = ck_tile::builder;

struct FwdConvSignature
{
    static constexpr int spatial_dim = 2;
    static constexpr auto direction  = ckb::ConvDirection::Forward;
    static constexpr auto layout     = ckb::GroupConvLayout::NHWGC_GKYXC_NHWGK;
    static constexpr auto data_type  = ckb::DataType::FP16;
};
static_assert(ckb::ConvSignature<FwdConvSignature>);

struct DefaultFwdConvAlgorithm
{
};
static_assert(ckb::ConvAlgorithmDescriptor<DefaultFwdConvAlgorithm>);

constexpr char API_VERSION[] = "0.1.0";
static_assert(ckb::SupportedVersion<API_VERSION>);

TEST(ConvBuilderTest, TestDefaultInstance)
{
    static constexpr const FwdConvSignature SIGNATURE;
    static constexpr const DefaultFwdConvAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}

} // namespace
