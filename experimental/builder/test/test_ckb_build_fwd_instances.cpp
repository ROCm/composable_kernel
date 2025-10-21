#include <gtest/gtest.h>

#include "impl/conv_algorithm_types.hpp"
#include "impl/conv_signature_types.hpp"
#include "ck_tile/builder/conv_builder.hpp"

class FwdConvBuilderTest : public ::testing::Test
{
};

template <ck_tile::builder::test::ConvSignature FwdConvSignature,
          ck_tile::builder::test::ThreadBlock FwdThreadBlock,
          ck_tile::builder::ConvFwdSpecialization FwdConvSpecialization>
constexpr void run_test()
{
    using namespace ck_tile::builder;
    using namespace ck_tile::builder::test;

    constexpr ConvTuningParams FwdTuningParams{.ak1            = 8,
                                               .bk1            = 8,
                                               .m_per_xdl      = 32,
                                               .n_per_xdl      = 32,
                                               .m_xdl_per_wave = 4,
                                               .n_xdl_per_wave = 4};

    constexpr InputOutputBlockTransfer FwdBlockTransfer{
        .thread_cluster_dims_a         = {.k0 = 4, .m_n = 64, .k1 = 1},
        .thread_cluster_dims_b         = {.k0 = 4, .m_n = 64, .k1 = 1},
        .thread_cluster_dims_c         = {.m_block        = 1,
                                          .m_wave_per_xdl = 32,
                                          .n_block        = 1,
                                          .n_wave_per_xdl = 8},
        .vector_transfer_a             = {.src_vector_dim            = 2,
                                          .src_scalar_per_vector     = 2,
                                          .dest_scalar_per_vector_k1 = 8,
                                          .add_extra                 = false},
        .vector_transfer_b             = {.src_vector_dim            = 2,
                                          .src_scalar_per_vector     = 8,
                                          .dest_scalar_per_vector_k1 = 8,
                                          .add_extra                 = false},
        .vector_transfer_c             = {.m_xdl_per_wave_per_shuffle = 1,
                                          .n_xdl_per_wave_per_shuffle = 1,
                                          .scalar_per_vector          = 8},
        .thread_cluster_access_order_a = {1, 0, 2},
        .thread_cluster_access_order_b = {1, 0, 2},
        .src_access_order_a            = {1, 0, 2},
        .src_access_order_b            = {1, 0, 2}};

    constexpr ConvAlgorithm FwdConvAlgorithm{.thread_block       = FwdThreadBlock,
                                             .tuning_params      = FwdTuningParams,
                                             .block_transfer     = FwdBlockTransfer,
                                             .pipeline_version   = BlockGemmPipelineVersion::V4,
                                             .fwd_specialization = FwdConvSpecialization};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    auto instance = typename Builder::Instance{};

    const auto kernel_string = instance.GetTypeString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;
    EXPECT_GT(kernel_string.size(), 0);

    EXPECT_TRUE(kernel_string.starts_with("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"));

    const auto invoker_ptr = instance.MakeInvokerPointer();
    EXPECT_NE(invoker_ptr, nullptr);
}

TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_BF16_ChannelsLast)
{
    using namespace ck_tile::builder;
    using namespace ck_tile::builder::test;

    constexpr ConvSignature FwdConvSignature{.spatial_dim = 2,
                                             .direction   = ConvDirection::FORWARD,
                                             .layout      = GroupConvLayout::CHANNELS_LAST,
                                             .data_type   = DataType::BF16};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 256, .n = 256, .k = 32}};

    run_test<FwdConvSignature, FwdThreadBlock, ConvFwdSpecialization::DEFAULT>();
}

TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_FP32_ChannelsFirst)
{
    using namespace ck_tile::builder;
    using namespace ck_tile::builder::test;

    constexpr ConvSignature FwdConvSignature{.spatial_dim = 3,
                                             .direction   = ConvDirection::FORWARD,
                                             .layout      = GroupConvLayout::CHANNELS_FIRST,
                                             .data_type   = DataType::FP32};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 128, .n = 128, .k = 32}};

    run_test<FwdConvSignature, FwdThreadBlock, ConvFwdSpecialization::FILTER_1X1_PAD0>();
}
