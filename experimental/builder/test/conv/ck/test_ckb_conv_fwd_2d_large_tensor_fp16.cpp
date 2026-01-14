// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Instance_2D_FP16_GNHWC)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;

    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 2,
                                             .direction              = FORWARD,
                                             .data_type              = FP16,
                                             .accumulation_data_type = FP32,
                                             .input  = {.config = {.layout = GNHWC}},
                                             .weight = {.config = {.layout = GKYXC}},
                                             .output = {.config = {.layout = GNHWK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor{}
            .with_thread_block(ThreadBlock_256_256x128x32)
            .with_gemm_config(FwdGemmParams_Xdl_2x1_per_wave)
            .with_transfer(Transfer_4x16x1)
            .with_fwd_specializations(ConvSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, PipelineScheduler::DEFAULT)
            .with_num_conv_groups_to_merge(1);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor",
                       expected_transfer_parameters,
                       "Default",
                       "GNHWC,GKYXC,EmptyTuple,GNHWK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding"});
}

TEST(
    FwdConvInstances,
    Create_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Instance_2D_FP16_GNHWC_Filter1x1Pad0)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;

    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 2,
                                             .direction              = FORWARD,
                                             .data_type              = FP16,
                                             .accumulation_data_type = FP32,
                                             .input  = {.config = {.layout = GNHWC}},
                                             .weight = {.config = {.layout = GKYXC}},
                                             .output = {.config = {.layout = GNHWK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor{}
            .with_thread_block(ThreadBlock_128_128x128x32)
            .with_gemm_config(FwdGemmParams_Xdl_2x1_per_wave)
            .with_transfer(Transfer_4x16x1)
            .with_fwd_specializations(ConvSpecialization::FILTER_1X1_PAD0,
                                      GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, PipelineScheduler::DEFAULT)
            .with_num_conv_groups_to_merge(1);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor",
                       expected_transfer_parameters,
                       "Filter1x1Pad0",
                       "GNHWC,GKYXC,EmptyTuple,GNHWK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding"});
}

} // namespace
