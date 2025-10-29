// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <ck_tile/builder/reflect/conv_traits.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>

namespace {

using ::testing::ElementsAre;

// Test fixture for ConvTraits tests
class ConvTraitsTest : public ::testing::Test
{
};

// Test ConvTraits with DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
TEST_F(ConvTraitsTest, ConvFwdTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default,                                               // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::Default, // GemmSpec
            256,                                                       // BlockSize
            128,                                                       // MPerBlock
            128,                                                       // NPerBlock
            16,                                                        // KPerBlock
            8,                                                         // AK1
            8,                                                         // BK1
            32,                                                        // MPerXDL
            32,                                                        // NPerXDL
            4,                                                         // MXdlPerWave
            4,                                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_AK1
            1,                      // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_BK1
            1,                      // BBlockLdsExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,               // CDEBlockTransferScalarPerVector_NPerBlock
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t,                                // BComputeDataType
            false>;                                    // DirectLoad

    // Use ConvTraits to extract compile-time information
    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    // Verify signature information
    EXPECT_EQ(Traits::spatial_dim, 2);
    EXPECT_EQ(Traits::direction, ck_tile::builder::ConvDirection::FORWARD);
    EXPECT_EQ(Traits::layout, ck_tile::builder::GroupConvLayout2D::GNHWC_GKYXC_GNHWK);
    EXPECT_EQ(Traits::data_type, ck_tile::builder::DataType::FP16);

    // Verify specializations
    EXPECT_EQ(Traits::gemm_specialization,
              ck::tensor_operation::device::GemmSpecialization::Default);
    EXPECT_EQ(Traits::conv_specialization,
              ck::tensor_operation::device::ConvolutionForwardSpecialization::Default);

    // Verify algorithm information
    EXPECT_EQ(Traits::thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(Traits::tile_dims.m, 128);
    EXPECT_EQ(Traits::tile_dims.n, 128);
    EXPECT_EQ(Traits::tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(Traits::a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(Traits::a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(Traits::a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(Traits::a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(Traits::a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(Traits::a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(Traits::a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(Traits::a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(Traits::a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(Traits::a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(Traits::a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(Traits::b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(Traits::b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(Traits::b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(Traits::b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(Traits::b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(Traits::b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(Traits::b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(Traits::b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(Traits::b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(Traits::b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(Traits::b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(Traits::warp_gemm.gemm_m, 32);
    EXPECT_EQ(Traits::warp_gemm.gemm_n, 32);
    EXPECT_EQ(Traits::warp_gemm.num_m_gemms, 4);
    EXPECT_EQ(Traits::warp_gemm.num_n_gemms, 4);

    // Verify output tile transfer info
    EXPECT_EQ(Traits::c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(Traits::c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(Traits::c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(Traits::c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(Traits::pipeline_scheduler, ck::BlockGemmPipelineScheduler::Intrawave);
    EXPECT_EQ(Traits::pipeline_version, ck::BlockGemmPipelineVersion::v1);
}

} // anonymous namespace
