// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Instance_2D_FP16_GNHWC)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor FwdConvAlgorithm{
        .base_algorithm = {
            .thread_block               = FwdThreadBlock_256_256x128x32,
            .gridwise_gemm              = FwdGemmParams_Xdl_2x1_per_wave,
            .block_transfer             = FwdBlockTransfer_4x16x1,
            .fwd_specialization         = ConvFwdSpecialization::DEFAULT,
            .gemm_specialization        = GemmSpecialization::MNKPadding,
            .num_gemm_k_prefetch_stages = 1,
            .num_groups_to_merge        = 1,
            .loop_scheduler             = PipelineScheduler::DEFAULT}};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor",
                       "256, 256, 128, 32",
                       "Default"});
}

TEST(
    FwdConvInstances,
    Create_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Instance_2D_FP16_GNHWC_Filter1x1Pad0)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor FwdConvAlgorithm{
        .base_algorithm = {
            .thread_block               = FwdThreadBlock_128_128x128x32,
            .gridwise_gemm              = FwdGemmParams_Xdl_2x1_per_wave,
            .block_transfer             = FwdBlockTransfer_4x16x1,
            .fwd_specialization         = ConvFwdSpecialization::FILTER_1X1_PAD0,
            .gemm_specialization        = GemmSpecialization::MNKPadding,
            .num_gemm_k_prefetch_stages = 1,
            .num_groups_to_merge        = 1,
            .loop_scheduler             = PipelineScheduler::DEFAULT}};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor",
                       "128, 128, 128, 32",
                       "Filter1x1Pad0"});
}

} // namespace
