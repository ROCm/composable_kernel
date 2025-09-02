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
static_assert(ckb::ConvSignature<FwdConvSignature>);

struct FwdConvAlgorithm
{
};
static_assert(ckb::ConvAlgorithm<FwdConvAlgorithm>);

static constexpr char API_VERSION[] = "0.1.0";

TEST(ConvBuilderTest, TestDefaultInstance)
{
    using Builder = ckb::ConvBuilder<FwdConvSignature, FwdConvAlgorithm, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}

[[maybe_unused]] static constexpr ckb::ThreadBlock THREAD_BLOCK_256_256_256_32{
    .block_size = 256,
    .sub_matrix = {.m = 256, .n = 256, .k = 32},
};

struct ConvFwdXdlBf16CompInstances2xAlgorithm0
{
    static constexpr ckb::ThreadBlock THREAD_BLOCK{
        .block_size = 256,
        .sub_matrix = {.m = 256, .n = 128, .k = 64},
    };
    static constexpr ckb::ConvTuningParams TUNING_PARAMS{
        .ak1            = 16,
        .bk1            = 16,
        .m_xdl_per_wave = 2,
        .n_xdl_per_wave = 2,
    };
};

TEST(ConvBuilderTest, TestConvFwdXdlBf16CompInstances2xInstance0)
{
    using Builder =
        ckb::ConvBuilder<FwdConvSignature, ConvFwdXdlBf16CompInstances2xAlgorithm0, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}

struct ConvFwdXdlBf16CompAlgorithm0
{
    static constexpr ckb::ThreadBlock THREAD_BLOCK{
        .block_size = 256,
        .sub_matrix = {.m = 256, .n = 256, .k = 32},
    };
    static constexpr ckb::ConvTuningParams TUNING_PARAMS{
        .ak1            = 8,
        .bk1            = 8,
        .m_xdl_per_wave = 4,
        .n_xdl_per_wave = 4,
    };
};

TEST(ConvBuilderTest, GroupedConvFwdXdlBf16CompInstance0)
{
    using Builder = ckb::ConvBuilder<FwdConvSignature, ConvFwdXdlBf16CompAlgorithm0, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
};

} // namespace
