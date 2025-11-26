// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// TEST(FwdConvInstances,
//      Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_BF16_ChannelsLast)
// {
//     constexpr ConvSignature FwdConvSignature{.spatial_dim = 2,
//                                              .direction   = ConvDirection::FORWARD,
//                                              .layout      = GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
//                                              .data_type   = DataType::BF16,
//                                              .elementwise_operation =
//                                                  ElementwiseOperation::PASS_THROUGH};

//     constexpr auto FwdConvAlgorithm =
//         ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
//             .with_thread_block(FwdThreadBlock_64_64x32x32)
//             .with_gemm_config(FwdGemmParams_Xdl_2x2_per_wave)
//             .with_transfer(FwdTransfer_4x16x1)
//             .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
//             .with_prefetch_config(1, 1, PipelineScheduler::DEFAULT)
//             .with_elementwise_ops({ElementwiseOperation::PASS_THROUGH, ElementwiseOperation::PASS_THROUGH, 
//                                    ElementwiseOperation::SCALEADD_SCALEADD_RELU});

//     using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
//     run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
//                        "256, 256, 256, 32",
//                        "Default",
//                        "BlkGemmPipelineScheduler: Intrawave",
//                        "BlkGemmPipelineVersion: v1"});
// }

} // namespace
