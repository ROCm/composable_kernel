#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ck_tile/builder/conv_builder.hpp>
#include "testing_utils.hpp"

namespace {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::test;
using P       = ckb::BlockGemmPipelineVersion;

// Defines the signature of the convolution operation to be tested.
// This includes dimensionality, direction, data layout, and data type.
struct ConvSignature
{
    int spatial_dim              = 2;
    ckb::ConvDirection direction = ckb::ConvDirection::FORWARD;
    ckb::GroupConvLayout layout  = ckb::GroupConvLayout::CHANNELS_LAST;
    ckb::DataType data_type      = ckb::DataType::FP16;
};
static_assert(ckb::ConvSignatureDescriptor<ConvSignature>);

struct DefaultAlgorithm
{
};
static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);

TEST(ConvBuilderTest, TestDefaultInstance)
{
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
    EXPECT_THAT(
        Builder::Instance::TypeString(),
        ckt::StringEqWithDiff("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, "
                              "Default, 32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: "
                              "Intrawave, BlkGemmPipelineVersion: v4>"));
}

} // namespace
