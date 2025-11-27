// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder;
using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_Instance_2D_BF16_scale_add_relu)
{
    constexpr auto G_K = BiasLayout::G_K_strided;
    constexpr auto NHWGK = ConvOutputLayout2D::NHWGK; 

    constexpr auto FwdConvLayout = ConvLayout
            {
                .input_layout  = ConvInputLayout2D::NHWGC,
                .weight_layout = ConvWeightLayout2D::GKYXC,
                .output_layout = ConvOutputLayout2D::NHWGK
            }
        .with_bias_layout<NHWGK, G_K>();

    constexpr ConvSignature FwdConvSignature{.spatial_dim = 2,
                                             .direction   = ConvDirection::FORWARD,
                                             .layout      = FwdConvLayout,
                                             .data_type   = DataType::BF16,
                                             .elementwise_operation =
                                                { .output_op = ElementwiseOperation::SCALEADD_SCALEADD_RELU}
                                            };

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
            .with_thread_block(FwdThreadBlock_64_64x32x32)
            .with_gemm_config(FwdGemmParams_Xdl_2x2_per_wave)
            .with_transfer(FwdTransfer_4x16x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, 1, PipelineScheduler::DEFAULT);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle",
                       "NHWGC,GKYXC,Tuple(NHWGK,G_K),NHWGK", // Check layouts
                       "PassThrough,PassThrough,ScaleAddScaleAddRelu", // Check elementwise ops
                       "64,64,32,32",
                       "MNKPadding",
                       "Default"});
}

} // namespace
