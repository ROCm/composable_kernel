// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances, Create_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK_Instance_2D_FP16_GNHWC)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 2,
                                             .direction              = ConvDirection::FORWARD,
                                             .data_type              = DataType::FP16,
                                             .accumulation_data_type = DataType::FP32,
                                             .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                             .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                             .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK{}
            .with_thread_block(FwdThreadBlock_256_128x128x16)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_dl_thread_config(DlThreadConfig_16x2x4x4x1)
            .with_dl_thread_cluster(DlThreadCluster_8x2)
            .with_dl_transfer(DlFwdTransfer);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK",
                       "256,128,128,16",
                       "Default",
                       "MNKPadding",
                       "GNHWC,GKYXC,EmptyTuple,GNHWK",
                       "PassThrough,PassThrough,PassThrough"});
}

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK_Instance_2D_FP16_FILTER_1X1_PAD0)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 2,
                                             .direction              = ConvDirection::FORWARD,
                                             .data_type              = DataType::FP16,
                                             .accumulation_data_type = DataType::FP32,
                                             .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                             .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                             .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK{}
            .with_thread_block(FwdThreadBlock_256_128x128x16)
            .with_specializations(ConvFwdSpecialization::FILTER_1X1_PAD0,
                                  GemmSpecialization::MNKPadding)
            .with_dl_thread_config(DlThreadConfig_16x2x4x4x1)
            .with_dl_thread_cluster(DlThreadCluster_8x2)
            .with_dl_transfer(DlFwdTransfer);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK",
                       "256,128,128,16",
                       "Filter1x1Pad0",
                       "MNKPadding",
                       "GNHWC,GKYXC,EmptyTuple,GNHWK",
                       "PassThrough,PassThrough,PassThrough"});
}

} // namespace
