// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Instance_2D_FP16_GNHWC)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim = 2,
                                             .direction   = ConvDirection::FORWARD,
                                             .layout      = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
                                             .data_type   = DataType::FP16,
                                             .elementwise_operation =
                                                 ElementwiseOperation::PASS_THROUGH};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor{
            .base_algorithm = ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
                                  .with_thread_block(FwdThreadBlock_256_256x128x32)
                                  .with_gemm_config(FwdGemmParams_Xdl_2x1_per_wave)
                                  .with_block_transfer(FwdBlockTransfer_4x16x1)
                                  .with_specializations(ConvFwdSpecialization::DEFAULT,
                                                        GemmSpecialization::MNKPadding)
                                  .with_prefetch_config(1, 1, PipelineScheduler::DEFAULT)};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor",
                       "256, 256, 128, 32",
                       "Default"});
}

TEST(
    FwdConvInstances,
    Create_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Instance_2D_FP16_GNHWC_Filter1x1Pad0)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim = 2,
                                             .direction   = ConvDirection::FORWARD,
                                             .layout      = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
                                             .data_type   = DataType::FP16,
                                             .elementwise_operation =
                                                 ElementwiseOperation::PASS_THROUGH};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor{
            .base_algorithm = ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
                                  .with_thread_block(FwdThreadBlock_128_128x128x32)
                                  .with_gemm_config(FwdGemmParams_Xdl_2x1_per_wave)
                                  .with_block_transfer(FwdBlockTransfer_4x16x1)
                                  .with_specializations(ConvFwdSpecialization::FILTER_1X1_PAD0,
                                                        GemmSpecialization::MNKPadding)
                                  .with_prefetch_config(1, 1, PipelineScheduler::DEFAULT)};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor",
                       "128, 128, 128, 32",
                       "Filter1x1Pad0"});
}

} // namespace
