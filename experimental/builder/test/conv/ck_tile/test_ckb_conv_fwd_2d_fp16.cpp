// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_tile_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances, Create_ConvAlgorithm_Tile_GroupedConvolutionForwardKernel_2D_FP16_NHWGC)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim = 2,
                                             .direction   = ConvDirection::FORWARD,
                                             .layout      = GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
                                             .data_type   = DataType::FP16,
                                             .elementwise_operation =
                                                 ElementwiseOperation::PASS_THROUGH};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_tile_thread_block(FwdTileThreadBlock_64x64x64)
            .with_tile_block_gemm(TileBlockGemmDesc_16x16_v1_intrawave)
            .with_tile_transfer(FwdTileTransfer_4x4x4);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "256, 256, 256, 32",
                       "Filter1x1Pad0",
                       "BlkGemmPipelineScheduler: Intrawave",
                       "BlkGemmPipelineVersion: v3"});
}

} // namespace
