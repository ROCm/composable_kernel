// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 1D FP16 (channels-last) with DEFAULT specialization
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_Instance_1D_FP16_ChannelsFirst_scale)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim = 1,
                                             .direction   = ConvDirection::FORWARD,
                                             .layout      = GroupConvLayout1D::NWGC_GKXC_NWGK,
                                             .data_type   = DataType::FP16,
                                             .elementwise_operation =
                                                 ElementwiseOperation::PASS_THROUGH};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
            .with_thread_block(FwdThreadBlock_64_64x32x32)
            .with_gemm_config(FwdGemmParams_Xdl_2x1_per_wave)
            .with_transfer(FwdTransfer_4x16x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, 2, PipelineScheduler::DEFAULT);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>(
        {"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle", "64, 64, 32, 32", "Default"});
}

} // namespace
