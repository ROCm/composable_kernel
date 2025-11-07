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
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
        .data_type             = DataType::FP8,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle FwdConvAlgorithm{
        .thread_block        = FwdThreadBlock_256_256x128x32,
        .gridwise_gemm       = FwdGemmParams_Xdl_4x2_per_wave,
        .block_transfer      = FwdBlockTransfer_4x64x1_fp8,
        .fwd_specialization  = ConvFwdSpecialization::DEFAULT,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .num_gemm_k_prefetch_stages =1,
        .num_groups_to_merge = 1,
        .loop_scheduler      = PipelineScheduler::DEFAULT
    };

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle",
                       "256, 256, 128, 32",
                       "Default"});
}

} // namespace
