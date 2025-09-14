#include <gtest/gtest.h>

#include <ck_tile/builder/conv_builder.hpp>

namespace {

namespace ckb = ck_tile::builder;
using P       = ckb::BlockGemmPipelineVersion;

// Defines the signature of the convolution operation to be tested.
// This includes dimensionality, direction, data layout, and data type.
struct ConvSignature
{
    int spatial_dim              = 3;
    ckb::ConvDirection direction = ckb::ConvDirection::Forward;
    ckb::GroupConvLayout layout  = ckb::GroupConvLayout::NDHWGC_GKZYXC_NDHWGK;
    ckb::DataType data_type      = ckb::DataType::BF16;
};
static_assert(ckb::ConvSignatureDescriptor<ConvSignature>);

struct DefaultAlgorithm
{
};
static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);

constexpr char API_VERSION[] = "0.1.0";
static_assert(ckb::SupportedVersion<API_VERSION>);

TEST(ConvBuilderGrpFwd3d, TestDefaultInstance)
{
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}

} // namespace
