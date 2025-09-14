#include <gtest/gtest.h>

#include <ck_tile/builder/conv_builder.hpp>

namespace {

namespace ckb = ck_tile::builder;
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

TEST(ConvBuilderGrpFwd2d, TestDefaultInstance)
{
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}

TEST(ConvBuilderGrpFwd2d, TestDefaultFP32Instance)
{
    static constexpr const ConvSignature SIGNATURE{.data_type = ckb::DataType::FP32};
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
    EXPECT_NE(Builder::Instance::TypeString(), "");
    // It's difficult to check the types direction on the kernel, so we instead
    // check that the builder has the correct data type aliases for FP32.
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::ADataType, float>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::BDataType, float>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::CShuffleDataType, float>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::AccDataType, float>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::EDataType, float>));
}

TEST(ConvBuilderGrpFwd2d, TestDefaultFP16Instance)
{
    static constexpr const ConvSignature SIGNATURE{.data_type = ckb::DataType::FP16};
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;

    // Check that the builder has the correct data type aliases for FP16.
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::ADataType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::BDataType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::CShuffleDataType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::AccDataType, float>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::EDataType, ck::half_t>));
}

TEST(ConvBuilderGrpFwd2d, TestDefaultBF16Instance)
{
    static constexpr const ConvSignature SIGNATURE{.data_type = ckb::DataType::BF16};
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;

    // Check that the builder has the correct data type aliases for BF16.
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::ADataType, ck::bhalf_t>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::BDataType, ck::bhalf_t>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::CShuffleDataType, ck::bhalf_t>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::AccDataType, float>));
    EXPECT_TRUE((std::is_same_v<typename Builder::Factory::Types::EDataType, ck::bhalf_t>));
}

} // namespace
