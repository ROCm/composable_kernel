#include <gtest/gtest.h>

#include "impl/conv_algorithm_types.hpp"
#include "impl/conv_signature_types.hpp"
#include "ck_tile/builder/conv_builder.hpp"

class FwdConvBuilderTest : public ::testing::Test
{
};

using namespace ck_tile::builder;
using namespace test;

template <auto FwdConvSignature,
          ThreadBlock FwdThreadBlock,
          BlockGemmPipelineVersion FwdPipelineVersion,
          ConvFwdSpecialization FwdConvSpecialization>
constexpr void run_test()
{
    constexpr GridwiseGemm FwdGemmParams{.ak1            = 8,
                                         .bk1            = 8,
                                         .m_per_xdl      = 32,
                                         .n_per_xdl      = 32,
                                         .m_xdl_per_wave = 4,
                                         .n_xdl_per_wave = 4};

    constexpr BlockTransferABC FwdBlockTransfer{
        .block_transfer_a         = {.k0 = 4, .m_n = 64, .k1 = 1},
        .block_transfer_b         = {.k0 = 4, .m_n = 64, .k1 = 1},
        .thread_cluster_dims_c    = {.m_block        = 1,
                                     .m_wave_per_xdl = 32,
                                     .n_block        = 1,
                                     .n_wave_per_xdl = 8},
        .lds_padding_a            = {.src_vector_dim            = 2,
                                     .src_scalar_per_vector     = 2,
                                     .dest_scalar_per_vector_k1 = 8,
                                     .add_extra                 = false},
        .lds_padding_b            = {.src_vector_dim            = 2,
                                     .src_scalar_per_vector     = 8,
                                     .dest_scalar_per_vector_k1 = 8,
                                     .add_extra                 = false},
        .epilogue_c               = {.m_xdl_per_wave_per_shuffle = 1,
                                     .n_xdl_per_wave_per_shuffle = 1,
                                     .scalar_per_vector          = 8},
        .block_transfer_access_order_a = {1, 0, 2},
        .block_transfer_access_order_b = {1, 0, 2},
        .src_access_order_a            = {1, 0, 2},
        .src_access_order_b            = {1, 0, 2}};

    constexpr ConvAlgorithm FwdConvAlgorithm{.thread_block       = FwdThreadBlock,
                                             .gridwise_gemm      = FwdGemmParams,
                                             .block_transfer     = FwdBlockTransfer,
                                             .pipeline_version   = FwdPipelineVersion,
                                             .fwd_specialization = FwdConvSpecialization};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    auto instance = typename Builder::Instance{};

    const auto kernel_string = instance.GetTypeString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;
    EXPECT_GT(kernel_string.size(), 0);

    EXPECT_TRUE(kernel_string.starts_with("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"));

    // Verify pipeline version is correct
    if(FwdPipelineVersion == BlockGemmPipelineVersion::V1)
        EXPECT_TRUE(kernel_string.find("BlkGemmPipelineVersion: v1") != std::string::npos);
    else if(FwdPipelineVersion == BlockGemmPipelineVersion::V3)
        EXPECT_TRUE(kernel_string.find("BlkGemmPipelineVersion: v3") != std::string::npos);
    else if(FwdPipelineVersion == BlockGemmPipelineVersion::V4)
        EXPECT_TRUE(kernel_string.find("BlkGemmPipelineVersion: v4") != std::string::npos);
    else if(FwdPipelineVersion == BlockGemmPipelineVersion::V5)
        EXPECT_TRUE(kernel_string.find("BlkGemmPipelineVersion: v5") != std::string::npos);

    // Verify specialization is correct
    if(FwdConvSpecialization == ConvFwdSpecialization::DEFAULT)
        EXPECT_TRUE(kernel_string.find("Default") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_1X1_PAD0)
        EXPECT_TRUE(kernel_string.find("Filter1x1Pad0") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0)
        EXPECT_TRUE(kernel_string.find("Filter1x1Stride1Pad0") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_3x3)
        EXPECT_TRUE(kernel_string.find("Filter3x3") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::ODD_C)
        EXPECT_TRUE(kernel_string.find("OddC") != std::string::npos);

    const auto invoker_ptr = instance.MakeInvokerPointer();
    EXPECT_NE(invoker_ptr, nullptr);
}

//==============================================================================
// 2D Forward Convolution Tests
//==============================================================================

// Test 1: 2D BF16 NHWGC (channels-last) with Pipeline V1 and DEFAULT
TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_BF16_ChannelsLast)
{
    constexpr ConvSignature<GroupConvLayout2D> FwdConvSignature{
        .spatial_dim = 2,
        .direction   = ConvDirection::FORWARD,
        .layout      = GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
        .data_type   = DataType::BF16};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 256, .n = 256, .k = 32}};

    run_test<FwdConvSignature,
             FwdThreadBlock,
             BlockGemmPipelineVersion::V1,
             ConvFwdSpecialization::DEFAULT>();
}

// Test 2: 2D FP16 GNHWC (group-first, channels-last) with Pipeline V3 and FILTER_1X1_PAD0
TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_FP16_GNHWC)
{
    constexpr ConvSignature<GroupConvLayout2D> FwdConvSignature{
        .spatial_dim = 2,
        .direction   = ConvDirection::FORWARD,
        .layout      = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
        .data_type   = DataType::FP16};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 256, .n = 256, .k = 32}};

    run_test<FwdConvSignature,
             FwdThreadBlock,
             BlockGemmPipelineVersion::V3,
             ConvFwdSpecialization::FILTER_1X1_PAD0>();
}

// Test 3: 2D FP32 NGCHW_GKCYX (channels-first, different weight layout) with Pipeline V4 and
// FILTER_1X1_STRIDE1_PAD0
TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_FP32_NGCHW_GKCYX)
{
    constexpr ConvSignature<GroupConvLayout2D> FwdConvSignature{
        .spatial_dim = 2,
        .direction   = ConvDirection::FORWARD,
        .layout      = GroupConvLayout2D::NGCHW_GKCYX_NGKHW,
        .data_type   = DataType::FP32};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 128, .n = 128, .k = 32}};

    run_test<FwdConvSignature,
             FwdThreadBlock,
             BlockGemmPipelineVersion::V4,
             ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0>();
}

// Test 4: 2D BF16 NHWGC (channels-last) with Pipeline V5 and FILTER_3x3
TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_BF16_NHWGC_Filter3x3)
{
    constexpr ConvSignature<GroupConvLayout2D> FwdConvSignature{
        .spatial_dim = 2,
        .direction   = ConvDirection::FORWARD,
        .layout      = GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
        .data_type   = DataType::BF16};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 256, .n = 256, .k = 32}};

    run_test<FwdConvSignature,
             FwdThreadBlock,
             BlockGemmPipelineVersion::V5,
             ConvFwdSpecialization::FILTER_3x3>();
}

//==============================================================================
// 3D Forward Convolution Tests
//==============================================================================

// Test 5: 3D FP32 NGCDHW (channels-first) with Pipeline V1 and FILTER_1X1_PAD0
TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_FP32_ChannelsFirst)
{
    constexpr ConvSignature<GroupConvLayout3D> FwdConvSignature{
        .spatial_dim = 3,
        .direction   = ConvDirection::FORWARD,
        .layout      = GroupConvLayout3D::NGCDHW_GKCZYX_NGKDHW,
        .data_type   = DataType::FP32};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 128, .n = 128, .k = 32}};

    run_test<FwdConvSignature,
             FwdThreadBlock,
             BlockGemmPipelineVersion::V1,
             ConvFwdSpecialization::FILTER_1X1_PAD0>();
}

// Test 6: 3D BF16 GNDHWC (group-first, channels-last) with Pipeline V3 and DEFAULT
TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_BF16_GNDHWC)
{
    constexpr ConvSignature<GroupConvLayout3D> FwdConvSignature{
        .spatial_dim = 3,
        .direction   = ConvDirection::FORWARD,
        .layout      = GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK,
        .data_type   = DataType::BF16};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 256, .n = 256, .k = 32}};

    run_test<FwdConvSignature,
             FwdThreadBlock,
             BlockGemmPipelineVersion::V3,
             ConvFwdSpecialization::DEFAULT>();
}

// Test 7: 3D FP16 NDHWGC (channels-last) with Pipeline V4 and FILTER_1X1_PAD0
TEST_F(FwdConvBuilderTest,
       Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_FP16_NDHWGC_ChannelsLast)
{
    constexpr ConvSignature<GroupConvLayout3D> FwdConvSignature{
        .spatial_dim = 3,
        .direction   = ConvDirection::FORWARD,
        .layout      = GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK,
        .data_type   = DataType::FP16};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 128, .n = 128, .k = 32}};

    run_test<FwdConvSignature,
             FwdThreadBlock,
             BlockGemmPipelineVersion::V4,
             ConvFwdSpecialization::FILTER_1X1_PAD0>();
}
