#include <gtest/gtest.h>

#include <ck_tile/builder/conv_builder.hpp>

namespace {

namespace ckb = ck_tile::builder;

// Example of kernel description for Forward Conv with default settings
struct GroupedConvFwdXdlImplicitGemm : public GroupedConvBaseXdlV1
{
    static constexpr ConvolutionDirection ConvolutionDirection_ = ConvolutionDirection::Forward;
    static constexpr ElementwiseOperation ElementwiseOperation_ = ElementwiseOperation::Bias;
};

// Example of kernel description for Backward Weight Conv with default settings and Split K Two
// Stage
struct GroupedConvBwdWeightXdlImplicitGemmTwoStage : public GroupedConvBaseXdlV1
{
    [[maybe_unused]] static constexpr ConvolutionDirection ConvolutionDirection_ =
        ConvolutionDirection::BackwardWeight;
    [[maybe_unused]] static constexpr SplitKSupport SplitKSupport_ =
        SplitKSupport::SupportedTwoStage;
};

struct Implementation16x16 : ImplementationDefaultV1
{
    static constexpr ck::index_t BlockSize_                   = 64;
    static constexpr auto TileSizes_                          = std::make_tuple(16, 16, 32);
    static constexpr ck::index_t K1_                          = 8;
    static constexpr MFMAInstructionSize MFMAInstructionSize_ = MFMAInstructionSize::M16N16;
    static constexpr auto XdlPerWave_                         = std::make_tuple(16, 16);
    static constexpr auto GlobalTransferVectorSize_           = std::make_tuple(1, 1, 1);
    static constexpr auto LDSStoreVectorSize_                 = std::make_tuple(4, 4);
};

struct ProblemBF16NHWGC : public BF16ProblemBaseV1, public NHWGCProblemBaseV1
{
};

TEST(ConvBuilderTest, TestBuilderV0_0_0)
{
    ConvolutionBuilder<GroupedConvFwdXdlImplicitGemm, ProblemBF16NHWGC, Implementation16x16>
        builder_fwd;

    EXPECT_EQ(builder_fwd.GetInstanceName(),
              "GroupedConvFwdMultipleABD_Xdl_CShuffle<64, 16, 16, 32, Default, 8, 16x16, 16, 16, "
              "1, 4, 1, 4, 1, Intrawave, v1, 1>");
    // It would be nice if this worked, but it fails.
    // [[maybe_unused]] auto instance = builder_fwd.GetInstance();
}

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
    //
};

TEST(ConvBuilderTest, TestAlgorithm)
{
    static_assert(ckb::ConvAlgorithm<FwdConvAlgorithm>);
    SUCCEED();
}

static constexpr char API_VERSION[] = "0.1.0";
using FwdConvBuilder = ckb::ConvBuilder<FwdConvSignature, FwdConvAlgorithm, API_VERSION>;

TEST(ConvBuilderTest, TestKernel)
{
    EXPECT_EQ(
        FwdConvBuilder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}
} // namespace
