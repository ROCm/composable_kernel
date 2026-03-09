// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <concepts>

#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_to_conv_traits.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp>

#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_wmma_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp>

namespace {

using ck_tile::builder::ConvDirection;
using ck_tile::builder::DataType;
using ck_tile::builder::ElementwiseOperation;
using ck_tile::builder::PipelineScheduler;
using ck_tile::builder::PipelineVersion;
using ck_tile::builder::TensorLayout;
using ::testing::ElementsAre;

// Test fixture for ConvTraits tests
class ConvTraitsTest : public ::testing::Test
{
};

// Test ConvTraits with DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle
TEST_F(ConvTraitsTest, ConvBwdDataMultipleDCshuffleWmmaTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWK,           // OutLayout
            ck::tensor_layout::convolution::GKYXC,           // WeiLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWC,           // InLayout
            ck::half_t,                                      // OutDataType
            ck::half_t,                                      // WeiDataType
            ck::half_t,                                      // OutDataType
            float,                                           // AccDataType
            ck::Tuple<>,                                     // DsDataType
            float,                                           // OutComputeType
            ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
            ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                Default,            // ConvBackwardDataSpecialization
            256,                    // BlockSize
            128,                    // MPerBlock
            128,                    // NPerBlock
            16,                     // K0PerBlock
            8,                      // K1
            32,                     // MPerWMMA
            32,                     // NPerWMMA
            4,                      // MRepeat
            4,                      // NRepeat
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_K1
            1,                      // ABlockLdsAddExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_K1
            1,                      // BBlockLdsAddExtraN
            1,                      // CShuffleMRepeatPerWavePerShuffle
            1,                      // CShuffleNRepeatPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
            8,               // CDEBlockTransferScalarPerVector_NPerBlock_
            2,               // NumGemmKPrefetchStage
            ck::LoopScheduler::Default, // BlkGemmPipeSched
            ck::PipelineVersion::v1>;   // PipelineVerison

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_DATA);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP32);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);
    EXPECT_EQ(traits.num_gemm_k_prefetch_stage, 2);

    // Verify pipeline configuration
}

// Test ConvTraits with DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle_V3
TEST_F(ConvTraitsTest, ConvBwdDataMultipleDCshuffleWmmaV3TraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWK,           // OutLayout
            ck::tensor_layout::convolution::GKYXC,           // WeiLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWC,           // InLayout
            ck::half_t,                                      // OutDataType
            ck::half_t,                                      // WeiDataType
            ck::half_t,                                      // OutDataType
            float,                                           // AccDataType
            ck::Tuple<>,                                     // DsDataType
            float,                                           // OutComputeType
            ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
            ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                Default,            // ConvBackwardDataSpecialization
            false,                  // DoPadGemmM
            false,                  // DoPadGemmN
            256,                    // BlockSize
            128,                    // MPerBlock
            128,                    // NPerBlock
            16,                     // K0PerBlock
            8,                      // AK1
            8,                      // BK1
            32,                     // MPerWMMA
            32,                     // NPerWMMA
            4,                      // MRepeat
            4,                      // NRepeat
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_K1
            1,                      // ABlockLdsAddExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_K1
            1,                      // BBlockLdsAddExtraN
            1,                      // CShuffleMRepeatPerWavePerShuffle
            1,                      // CShuffleNRepeatPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
            ck::Sequence<8, 8, 8>,                     // CDEBlockTransferScalarPerVector_NPerBlock_
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t,                                // BComputeDataType
            1,                                         // MaxTransposeTransferSrcScalarPerVector
            1>;                                        // MaxTransposeTransferDstScalarPerVector

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_DATA);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP32);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);
    EXPECT_FALSE(traits.do_pad_gemm_n.value());
    EXPECT_FALSE(traits.do_pad_gemm_m.value());

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    EXPECT_EQ(traits.max_transpose_transfer_src_scalar_per_vector, 1);
    EXPECT_EQ(traits.max_transpose_transfer_dst_scalar_per_vector, 1);

    // Verify pipeline configuration
}

// Test ConvTraits with DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle
TEST_F(ConvTraitsTest, ConvBwdDataMultipleDCshuffleXDLTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWK,           // OutLayout
            ck::tensor_layout::convolution::GKYXC,           // WeiLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWC,           // InLayout
            ck::half_t,                                      // OutDataType
            ck::half_t,                                      // WeiDataType
            ck::half_t,                                      // OutDataType
            float,                                           // AccDataType
            ck::Tuple<>,                                     // DsDataType
            float,                                           // OutComputeType
            ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
            ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                Default,            // ConvBackwardDataSpecialization
            false,                  // DoPadGemmM
            false,                  // DoPadGemmN
            1,                      // num_gemm_k_prefetch_stage
            256,                    // BlockSize
            128,                    // MPerBlock
            128,                    // NPerBlock
            16,                     // K0PerBlock
            8,                      // AK1
            8,                      // BK1
            32,                     // MPerXDL
            32,                     // NPerXDL
            4,                      // MXdlPerWave
            4,                      // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_K1
            1,                      // ABlockLdsAddExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_K1
            1,                      // BBlockLdsAddExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
            8,               // CDEBlockTransferScalarPerVector_NPerBlock_
            ck::LoopScheduler::Default, // BlkGemmPipeSched
            ck::half_t,                 // AComputeDataType
            ck::half_t,                 // BComputeDataType
            1,                          // MaxTransposeTransferSrcScalarPerVector
            1>;                         // MaxTransposeTransferDstScalarPerVector

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_DATA);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP32);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(traits.num_gemm_k_prefetch_stage, 1);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);
    EXPECT_FALSE(traits.do_pad_gemm_n.value());
    EXPECT_FALSE(traits.do_pad_gemm_m.value());

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    EXPECT_EQ(traits.max_transpose_transfer_src_scalar_per_vector, 1);
    EXPECT_EQ(traits.max_transpose_transfer_dst_scalar_per_vector, 1);
}

// Test ConvTraits with DeviceGroupedConvBwdWeight_Wmma_CShuffle
TEST_F(ConvTraitsTest, ConvBwdWeightCshuffleWmmaTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffle<
        3,                                               // NDimSpatial
        ck::tensor_layout::convolution::GNDHWC,          // InLayout
        ck::tensor_layout::convolution::GKZYXC,          // WeiLayout
        ck::tensor_layout::convolution::GNDHWK,          // OutLayout
        ck::half_t,                                      // InDataType
        ck::half_t,                                      // WeiDataType
        ck::half_t,                                      // OutDataType
        float,                                           // AccDataType
        ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
            Default,            // ConvBackwardWeightSpecialization
        256,                    // BlockSize
        128,                    // MPerBlock
        128,                    // NPerBlock
        16,                     // K0PerBlock
        8,                      // K1
        32,                     // MPerWmma
        32,                     // NPerWmma
        4,                      // MRepeat
        4,                      // NRepeat
        ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
        ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
        ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,                      // ABlockTransferSrcVectorDim
        8,                      // ABlockTransferSrcScalarPerVector
        8,                      // ABlockTransferDstScalarPerVector_K1
        1,                      // ABlockLdsAddExtraM
        ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
        ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
        ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
        2,                      // BBlockTransferSrcVectorDim
        8,                      // BBlockTransferSrcScalarPerVector
        8,                      // BBlockTransferDstScalarPerVector_K1
        1,                      // BBlockLdsAddExtraN
        1,                      // CShuffleMXdlPerWavePerShuffle
        1,                      // CShuffleNXdlPerWavePerShuffle
        ck::Sequence<1,
                     32,
                     1,
                     8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
        8,               // CDEBlockTransferScalarPerVector_NPerBlock_
        1,               // NummGemmKPrefetchStage
        ck::LoopScheduler::Default, // BlkGemmPipeSched
        ck::PipelineVersion::v1,    // BlkGemmPipelineVer
        false>;                     // BComputeDataType

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 3);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_WEIGHT);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNDHWC, TensorLayout::GKZYXC, TensorLayout::GNDHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
}

// Test ConvTraits with DeviceGroupedConvBwdWeight_Xdl_CShuffle_V3
TEST_F(ConvTraitsTest, ConvBwdWeightCshuffleWmmaV3TraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<
        2,                                               // NDimSpatial
        ck::tensor_layout::convolution::GNHWC,           // InLayout
        ck::tensor_layout::convolution::GKYXC,           // WeiLayout
        ck::tensor_layout::convolution::GNHWK,           // OutLayout
        ck::half_t,                                      // InDataType
        ck::half_t,                                      // WeiDataType
        ck::half_t,                                      // OutDataType
        float,                                           // AccDataType
        ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
            Default,            // ConvBackwardWeightSpecialization
        256,                    // BlockSize
        128,                    // MPerBlock
        128,                    // NPerBlock
        16,                     // K0PerBlock
        8,                      // K1
        32,                     // MPerWmma
        32,                     // NPerWmma
        4,                      // MRepeat
        4,                      // NRepeat
        ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
        ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
        ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,                      // ABlockTransferSrcVectorDim
        8,                      // ABlockTransferSrcScalarPerVector
        8,                      // ABlockTransferDstScalarPerVector_K1
        1,                      // ABlockLdsAddExtraM
        ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
        ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
        ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
        2,                      // BBlockTransferSrcVectorDim
        8,                      // BBlockTransferSrcScalarPerVector
        8,                      // BBlockTransferDstScalarPerVector_K1
        1,                      // BBlockLdsAddExtraN
        1,                      // CShuffleMXdlPerWavePerShuffle
        1,                      // CShuffleNXdlPerWavePerShuffle
        ck::Sequence<1,
                     32,
                     1,
                     8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
        8,               // CDEBlockTransferScalarPerVector_NPerBlock_
        ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
        ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
        ck::half_t,                                // AComputeDataType
        ck::half_t,                                // BComputeDataType
        1,                                         // MaxTransposeTransferSrcScalarPerVector
        1>;                                        // MaxTransposeTransferDstScalarPerVector>

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_WEIGHT);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    EXPECT_EQ(traits.max_transpose_transfer_src_scalar_per_vector, 1);
    EXPECT_EQ(traits.max_transpose_transfer_dst_scalar_per_vector, 1);

    // Verify pipeline configuration
}

// Test ConvTraits with DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffle_V3
TEST_F(ConvTraitsTest, ConvBwdWeightMultipleDCshuffleWmmaV3TraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // InLayout
            ck::tensor_layout::convolution::GKYXC,           // WeiLayout
            ck::tensor_layout::convolution::GNHWK,           // OutLayout
            ck::Tuple<>,                                     // DsLayout
            ck::half_t,                                      // InDataType
            ck::half_t,                                      // WeiDataType
            ck::half_t,                                      // OutDataType
            float,                                           // AccDataType
            ck::Tuple<>,                                     // DsDataType
            ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
            ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
                Default,            // ConvBackwardWeightSpecialization
            256,                    // BlockSize
            128,                    // MPerBlock
            128,                    // NPerBlock
            16,                     // K0PerBlock
            8,                      // K1
            32,                     // MPerWmma
            32,                     // NPerWmma
            4,                      // MRepeat
            4,                      // NRepeat
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_K1
            1,                      // ABlockLdsAddExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_K1
            1,                      // BBlockLdsAddExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
            8,               // CDEBlockTransferScalarPerVector_NPerBlock_
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t>;                               // BComputeDataType

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_WEIGHT);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
}

// Test ConvTraits with DeviceGroupedConvBwdWeight_two_stage_Xdl_CShuffleV3
TEST_F(ConvTraitsTest, ConvBwdWeightTwoStageWmmaCshuffleTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // InLayout
            ck::tensor_layout::convolution::GKYXC,           // WeiLayout
            ck::tensor_layout::convolution::GNHWK,           // OutLayout
            ck::half_t,                                      // InDataType
            ck::half_t,                                      // WeiDataType
            ck::half_t,                                      // OutDataType
            float,                                           // AccDataType
            ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
            ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
                Default,            // ConvBackwardWeightSpecialization
            256,                    // BlockSize
            128,                    // MPerBlock
            128,                    // NPerBlock
            16,                     // K0PerBlock
            8,                      // AK1
            32,                     // MPerWMMA
            32,                     // NPerXDL
            4,                      // MRepeat
            4,                      // NRepeat
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_K1
            1,                      // ABlockLdsAddExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_K1
            1,                      // BBlockLdsAddExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
            8,               // CDEBlockTransferScalarPerVector_NPerBlock_
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            4,                                         // NumGroupsToMerge
            ck::half_t,                                // AComputeDataType
            ck::half_t,                                // BComputeDataType
            1,                                         // MaxTransposeTransferSrcScalarPerVector
            1>;                                        // MaxTransposeTransferDstScalarPerVector>

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_WEIGHT);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);

    EXPECT_EQ(traits.max_transpose_transfer_src_scalar_per_vector, 1);
    EXPECT_EQ(traits.max_transpose_transfer_dst_scalar_per_vector, 1);
}

// Test ConvTraits with DeviceGroupedConvBwdWeight_two_stage_Xdl_CShuffleV3
TEST_F(ConvTraitsTest, ConvBwdWeightTwoStageXdlCshuffleTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // InLayout
            ck::tensor_layout::convolution::GKYXC,           // WeiLayout
            ck::tensor_layout::convolution::GNHWK,           // OutLayout
            ck::half_t,                                      // InDataType
            ck::half_t,                                      // WeiDataType
            ck::half_t,                                      // OutDataType
            float,                                           // AccDataType
            ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
            ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
                Default,            // ConvBackwardWeightSpecialization
            256,                    // BlockSize
            128,                    // MPerBlock
            128,                    // NPerBlock
            16,                     // K0PerBlock
            8,                      // K1
            32,                     // MPerXDL
            32,                     // NPerXDL
            4,                      // MXdlPerWave
            4,                      // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_K1
            1,                      // ABlockLdsAddExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_K1
            1,                      // BBlockLdsAddExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
            8,               // CDEBlockTransferScalarPerVector_NPerBlock_
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            4,                                         // NumGroupsToMerge
            ck::half_t,                                // AComputeDataType
            ck::half_t,                                // BComputeDataType
            1,                                         // MaxTransposeTransferSrcScalarPerVector
            1>;                                        // MaxTransposeTransferDstScalarPerVector>

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_WEIGHT);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);

    EXPECT_EQ(traits.max_transpose_transfer_src_scalar_per_vector, 1);
    EXPECT_EQ(traits.max_transpose_transfer_dst_scalar_per_vector, 1);
}

// Test ConvTraits with DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle
TEST_F(ConvTraitsTest, ConvBwdWeightMultipleDCshuffleXDLTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // InLayout
            ck::tensor_layout::convolution::GKYXC,           // WeiLayout
            ck::tensor_layout::convolution::GNHWK,           // OutLayout
            ck::Tuple<>,                                     // DsLayout
            ck::half_t,                                      // InDataType
            ck::half_t,                                      // WeiDataType
            ck::half_t,                                      // OutDataType
            float,                                           // AccDataType
            ck::Tuple<>,                                     // DsDataType
            ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
            ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
                Default,            // ConvBackwardWeightSpecialization
            256,                    // BlockSize
            128,                    // MPerBlock
            128,                    // NPerBlock
            16,                     // K0PerBlock
            8,                      // K1
            32,                     // MPerXDL
            32,                     // NPerXDL
            4,                      // MXdlPerWave
            4,                      // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_K1
            1,                      // ABlockLdsAddExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_K1
            1,                      // BBlockLdsAddExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
            8,               // CDEBlockTransferScalarPerVector_NPerBlock_
            ck::half_t,      // AComputeDataType
            ck::half_t>;     // BComputeDataType

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_WEIGHT);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
}

// Test ConvTraits with DeviceGroupedConvBwdWeight_Xdl_CShuffleV3
TEST_F(ConvTraitsTest, ConvBwdWeightXdlCshuffleV3TraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffleV3<
        2,                                               // NDimSpatial
        ck::tensor_layout::convolution::GNHWC,           // InLayout
        ck::tensor_layout::convolution::GKYXC,           // WeiLayout
        ck::tensor_layout::convolution::GNHWK,           // OutLayout
        ck::half_t,                                      // InDataType
        ck::half_t,                                      // WeiDataType
        ck::half_t,                                      // OutDataType
        float,                                           // AccDataType
        ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
            Default,            // ConvBackwardWeightSpecialization
        256,                    // BlockSize
        128,                    // MPerBlock
        128,                    // NPerBlock
        16,                     // K0PerBlock
        8,                      // K1
        32,                     // MPerXDL
        32,                     // NPerXDL
        4,                      // MXdlPerWave
        4,                      // NXdlPerWave
        ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
        ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
        ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,                      // ABlockTransferSrcVectorDim
        8,                      // ABlockTransferSrcScalarPerVector
        8,                      // ABlockTransferDstScalarPerVector_K1
        1,                      // ABlockLdsAddExtraM
        ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
        ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
        ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
        2,                      // BBlockTransferSrcVectorDim
        8,                      // BBlockTransferSrcScalarPerVector
        8,                      // BBlockTransferDstScalarPerVector_K1
        1,                      // BBlockLdsAddExtraN
        1,                      // CShuffleMXdlPerWavePerShuffle
        1,                      // CShuffleNXdlPerWavePerShuffle
        ck::Sequence<1,
                     32,
                     1,
                     8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
        8,               // CDEBlockTransferScalarPerVector_NPerBlock_
        ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
        ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
        ck::half_t,                                // AComputeDataType
        ck::half_t,                                // BComputeDataType
        false,                                     // DirectLoad
        1>;                                        // NumGroupsToMerge

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_WEIGHT);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);
}

// Test ConvTraits with DeviceGroupedConvBwdWeight_Xdl_CShuffle
TEST_F(ConvTraitsTest, ConvBwdWeightXdlCshuffleTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffle<
        2,                                               // NDimSpatial
        ck::tensor_layout::convolution::GNHWC,           // InLayout
        ck::tensor_layout::convolution::GKYXC,           // WeiLayout
        ck::tensor_layout::convolution::GNHWK,           // OutLayout
        ck::half_t,                                      // InDataType
        ck::half_t,                                      // WeiDataType
        ck::half_t,                                      // OutDataType
        float,                                           // AccDataType
        ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
            Default,            // ConvBackwardWeightSpecialization
        256,                    // BlockSize
        128,                    // MPerBlock
        128,                    // NPerBlock
        16,                     // K0PerBlock
        8,                      // K1
        32,                     // MPerXDL
        32,                     // NPerXDL
        4,                      // MXdlPerWave
        4,                      // NXdlPerWave
        ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
        ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder_
        ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,                      // ABlockTransferSrcVectorDim
        8,                      // ABlockTransferSrcScalarPerVector
        8,                      // ABlockTransferDstScalarPerVector_K1
        1,                      // ABlockLdsAddExtraM
        ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
        ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder_
        ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder_
        2,                      // BBlockTransferSrcVectorDim
        8,                      // BBlockTransferSrcScalarPerVector
        8,                      // BBlockTransferDstScalarPerVector_K1
        1,                      // BBlockLdsAddExtraN
        1,                      // CShuffleMXdlPerWavePerShuffle
        1,                      // CShuffleNXdlPerWavePerShuffle
        ck::Sequence<1,
                     32,
                     1,
                     8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_
        8,               // CDEBlockTransferScalarPerVector_NPerBlock_
        ck::half_t,      // AComputeDataType
        ck::half_t,      // BComputeDataType
        1,               // MaxTransposeTransferSrcScalarPerVector
        1>;              // MaxTransposeTransferDstScalarPerVector>

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::BACKWARD_WEIGHT);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);

    EXPECT_EQ(traits.max_transpose_transfer_src_scalar_per_vector, 1);
    EXPECT_EQ(traits.max_transpose_transfer_dst_scalar_per_vector, 1);
}

// test conv traits device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp
TEST_F(ConvTraitsTest, ConvFwdTraitsMultipleDCshuffleWmmaExtraction)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<
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
            1,                                                         // NummGemmKPrefetchStage
            256,                                                       // BlockSize
            128,                                                       // MPerBlock
            128,                                                       // NPerBlock
            16,                                                        // KPerBlock
            8,                                                         // K1
            32,                                                        // MPerWmma
            32,                                                        // NPerWmma
            4,                                                         // MRepeat
            4,                                                         // NRepeat
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
            1,                      // CShuffleMRepeatPerShuffle
            1,                      // CShuffleNRepeatPerShuffle
            ck::Sequence<
                1,
                32,
                1,
                8>, // CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,      // CDEShuffleBlockTransferScalarPerVector_NPerBlock
            ck::LoopScheduler::Default, // BlkGemmPipeSched
            ck::PipelineVersion::v1>;   // BlkGemmPipelineVer

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::FORWARD);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.gemm_padding, ck_tile::builder::GemmPadding::DEFAULT);
    EXPECT_EQ(traits.conv_specialization, ck_tile::builder::ConvSpecialization::DEFAULT);
    EXPECT_EQ(traits.num_gemm_k_prefetch_stage, 1);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(traits.pipeline_scheduler,
              ck_tile::reflect::conv::convert_pipeline_scheduler<ck::LoopScheduler::Default>());
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);
}

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
            false,                                     // DirectLoad
            1>;                                        // NumGroupsToMerge

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::FORWARD);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.gemm_padding, ck_tile::builder::GemmPadding::DEFAULT);
    EXPECT_EQ(traits.conv_specialization, ck_tile::builder::ConvSpecialization::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);

    // Verify A tile transfer info
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.a_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.a_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.a_tile_transfer.transfer_params.lds_padding);

    // Verify B tile transfer info
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k0, 2);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.m_or_n, 128);
    EXPECT_EQ(traits.b_tile_transfer.tile_dimensions.k1, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.k1, 8);
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_dims, ElementsAre(4, 64, 1));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.thread_cluster_order, ElementsAre(1, 0, 2));
    EXPECT_THAT(traits.b_tile_transfer.transfer_params.src_access_order, ElementsAre(1, 0, 2));
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_vector_dim, 2);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.src_scalar_per_vector, 8);
    EXPECT_EQ(traits.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1, 8);
    EXPECT_TRUE(traits.b_tile_transfer.transfer_params.lds_padding);

    // Verify warp GEMM params
    EXPECT_EQ(traits.warp_gemm.gemm_m, 32);
    EXPECT_EQ(traits.warp_gemm.gemm_n, 32);
    EXPECT_EQ(traits.warp_gemm.m_iter, 4);
    EXPECT_EQ(traits.warp_gemm.n_iter, 4);

    // Verify output tile transfer info
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.m_gemms_per_shuffle, 1);
    EXPECT_EQ(traits.c_tile_transfer.shuffle_params.n_gemms_per_shuffle, 1);
    EXPECT_THAT(traits.c_tile_transfer.thread_cluster_dims, ElementsAre(1, 32, 1, 8));
    EXPECT_EQ(traits.c_tile_transfer.scalar_per_vector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::INTRAWAVE);
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);
}

// Test ConvTraits with DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
TEST_F(ConvTraitsTest, ConvFwdBaseTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
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
            1,                                                         // NumGemmKPrefetchStage
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
            ck::half_t,      // AComputeDataType
            ck::half_t,      // BComputeDataType
            ck::LoopScheduler::Default, // LoopSched
            1>;                         // NumGroupsToMerge

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::FORWARD);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.gemm_padding, ck_tile::builder::GemmPadding::DEFAULT);
    EXPECT_EQ(traits.conv_specialization, ck_tile::builder::ConvSpecialization::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);
}
// Test ConvTraits with DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor
TEST_F(ConvTraitsTest, ConvFwdLargeTensorTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
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
            1,                                                         // NumGemmKPrefetchStage
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
                         8>,             // CDEBlockTransferClusterLengths
            8,                           // CDEBlockTransferScalarPerVector_NPerBlock
            ck::half_t,                  // AComputeDataType
            ck::half_t,                  // BComputeDataType
            ck::LoopScheduler::Default>; // LoopSched

    // Use ConvTraitsTmpl to extract compile-time information
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();

    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, 2);
    EXPECT_EQ(traits.direction, ConvDirection::FORWARD);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.input_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.weight_element_op, ElementwiseOperation::PASS_THROUGH);
    EXPECT_EQ(traits.output_element_op, ElementwiseOperation::PASS_THROUGH);

    // Verify specializations
    EXPECT_EQ(traits.gemm_padding, ck_tile::builder::GemmPadding::DEFAULT);
    EXPECT_EQ(traits.conv_specialization, ck_tile::builder::ConvSpecialization::DEFAULT);

    // Verify algorithm information
    EXPECT_EQ(traits.thread_block_size, 256);

    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, 128);
    EXPECT_EQ(traits.tile_dims.n, 128);
    EXPECT_EQ(traits.tile_dims.k, 16);
}
} // anonymous namespace
