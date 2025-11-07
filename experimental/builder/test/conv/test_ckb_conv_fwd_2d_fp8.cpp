// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 2D FP8 NHWGC (channels-last) with Pipeline V1 and DEFAULT
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_Instance_2D_FP8_ChannelsLast)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim = 2,
                                             .direction   = ConvDirection::FORWARD,
                                             .layout      = GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
                                             .data_type   = DataType::FP8,
                                             .elementwise_operation =
                                                 ElementwiseOperation::PASS_THROUGH};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
            .with_thread_block(FwdThreadBlock_256_256x128x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x2_per_wave)
            .with_block_transfer(FwdBlockTransfer_4x64x1_fp8)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, 1, PipelineScheduler::DEFAULT);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>(
        {"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle", "256, 256, 128, 32", "Default"});
}

} // namespace
