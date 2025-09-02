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

struct DefaultFwdConvAlgorithm
{
};
static_assert(ckb::ConvAlgorithm<DefaultFwdConvAlgorithm>);

static constexpr char API_VERSION[] = "0.1.0";

TEST(ConvBuilderTest, TestDefaultInstance)
{
    static constexpr DefaultFwdConvAlgorithm algorithm;
    using Builder = ckb::ConvBuilder<FwdConvSignature, algorithm, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
}

struct FwdConvAlgorithm
{
    ckb::ThreadBlock thread_block;
    ckb::ConvTuningParams tuning_params;
    struct BlockTransfer
    {
        ckb::BlockATransferLengthsInfo thread_cluster_lengths_a;
        ckb::BlockBTransferLengthsInfo thread_cluster_lengths_b;
        ckb::BlockCTransferLengthsInfo thread_cluster_lengths_c;
    } block_transfer;
};
static_assert(ckb::ConvAlgorithm<FwdConvAlgorithm>);
static_assert(ckb::HasThreadBlockInfo<FwdConvAlgorithm>);
static_assert(ckb::HasConvTuningInfo<FwdConvAlgorithm>);
static_assert(ckb::HasABlockTransferInfo<FwdConvAlgorithm>);
static_assert(ckb::HasBBlockTransferInfo<FwdConvAlgorithm>);
static_assert(ckb::HasCBlockTransferInfo<FwdConvAlgorithm>);

TEST(ConvBuilderTest, TestConvFwdXdlBf16CompInstances2xInstance0)
{
    static constexpr FwdConvAlgorithm algorithm{
        .thread_block{.block_size = 256, .sub_matrix = {.m = 256, .n = 128, .k = 64}},
        .tuning_params{.ak1 = 16, .bk1 = 16, .m_xdl_per_wave = 2, .n_xdl_per_wave = 2},
        .block_transfer{
            .thread_cluster_lengths_a = {.k0 = 4, .m = 64, .k1 = 1},
            .thread_cluster_lengths_b = {.k0 = 4, .n = 64, .k1 = 1},
            .thread_cluster_lengths_c =
                {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
        }};
    using Builder = ckb::ConvBuilder<FwdConvSignature, algorithm, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 64, Default, 32, 32, 2, 2, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
    EXPECT_EQ(Builder::factory::TUNING.ak1, 16);
    EXPECT_EQ(Builder::factory::TUNING.bk1, 16);
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[0], 4);
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[1], 64);
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[2], 1);
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[0], 4);
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[1], 64);
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[2], 1);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[0], 1);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[1], 32);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[2], 1);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[3], 8);
}
TEST(ConvBuilderTest, GroupedConvFwdXdlBf16CompInstance0)
{
    static constexpr FwdConvAlgorithm algorithm{
        .thread_block{.block_size = 256, .sub_matrix = {.m = 256, .n = 256, .k = 32}},
        .tuning_params{.ak1 = 8, .bk1 = 8, .m_xdl_per_wave = 4, .n_xdl_per_wave = 4},
        .block_transfer{
            .thread_cluster_lengths_a = {.k0 = 4, .m = 64, .k1 = 1},
            .thread_cluster_lengths_b = {.k0 = 4, .n = 64, .k1 = 1},
            .thread_cluster_lengths_c =
                {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
        }};
    using Builder = ckb::ConvBuilder<FwdConvSignature, algorithm, API_VERSION>;
    EXPECT_EQ(
        Builder::Instance::TypeString(),
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 32, Default, 32, 32, 4, 4, "
        "8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>");
    EXPECT_EQ(Builder::factory::TUNING.ak1, 8);
    EXPECT_EQ(Builder::factory::TUNING.bk1, 8);
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[0], 4);
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[1], 64);
    EXPECT_EQ(Builder::factory::A_BLOCK_TRANSFER.thread_cluster_lengths[2], 1);
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[0], 4);
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[1], 64);
    EXPECT_EQ(Builder::factory::B_BLOCK_TRANSFER.thread_cluster_lengths[2], 1);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[0], 1);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[1], 32);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[2], 1);
    EXPECT_EQ(Builder::factory::C_BLOCK_TRANSFER.thread_cluster_lengths[3], 8);
};

} // namespace
