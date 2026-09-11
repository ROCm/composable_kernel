// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/conv/bwd_weight.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "utils/ckb_conv_tile_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "testing_utils.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;
namespace ckf = ck_tile::builder::factory;

using enum ck_tile::builder::TensorLayout;
using ck_tile::test::MatchesReference;
using ck_tile::test::SuccessfulRun;

constexpr auto SIGNATURE = cku::ConvSignature{.spatial_dim = 2,
                                              .direction   = ckb::ConvDirection::BACKWARD_WEIGHT,
                                              .data_type   = ckb::DataType::FP16,
                                              .accumulation_data_type = ckb::DataType::FP32,
                                              .input  = {.config = {.layout = NHWGC}},
                                              .weight = {.config = {.layout = GKYXC}},
                                              .output = {.config = {.layout = NHWGK}}};

constexpr auto ALGORITHM =
    cku::ConvAlgorithm_Tile_GroupedConvolutionKernel{}
        .with_tile_specializations(ckb::TileConvSpecialization::DEFAULT)
        .with_tile_thread_block(cku::TileThreadBlock_64x64x64)
        .with_tile_block_gemm(cku::TileBlockGemmDesc_16x16_v3_intrawave)
        .with_tile_transfer(cku::TileTransfer_4x4x4)
        .with_tile_optimizations(ckt::TileOptimizations{.num_groups_to_merge = 1,
                                                        .split_image         = false,
                                                        .explicit_gemm       = false,
                                                        .two_stage           = false});

constexpr auto TWO_STAGE_ALGORITHM =
    cku::ConvAlgorithm_Tile_GroupedConvolutionKernel{}
        .with_tile_specializations(ckb::TileConvSpecialization::DEFAULT)
        .with_tile_thread_block(cku::TileThreadBlock_64x64x64)
        .with_tile_block_gemm(cku::TileBlockGemmDesc_16x16_v3_intrawave)
        .with_tile_transfer(cku::TileTransfer_4x4x4)
        .with_tile_optimizations(ckt::TileOptimizations{.num_groups_to_merge = 1,
                                                        .split_image         = false,
                                                        .explicit_gemm       = false,
                                                        .two_stage           = true});

constexpr ckt::Args<SIGNATURE> Args = {
    .lengths =
        {
            .batch_size      = 2,
            .groups          = 4,
            .input_channels  = 32,
            .output_channels = 48,
            .image           = {.width = 32, .height = 56},
            .filter          = {.width = 3, .height = 3},
        },
    .filter_strides     = {.width = 1, .height = 1},
    .filter_dilation    = {.width = 1, .height = 1},
    .input_left_pad     = {.width = 0, .height = 0},
    .input_right_pad    = {.width = 0, .height = 0},
    .a_elementwise_op   = {},
    .b_elementwise_op   = {},
    .cde_elementwise_op = {},
};

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

using TwoStageBuilder       = ckb::ConvBuilder<SIGNATURE, TWO_STAGE_ALGORITHM>;
using TwoStageInstance      = TwoStageBuilder::Instance;
using ElementwiseOpBuilder  = ckf::ElementwiseOpTileFactory<SIGNATURE, TWO_STAGE_ALGORITHM>;
using ElementwiseOpInstance = ElementwiseOpBuilder::Instance;

using Reference = ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;

TEST(BwdWeight_2D_FP16_NHWGC, Create)
{
    cku::run_ck_tile_test<Builder>({
        "grouped_convolution_backward_weight",
        "fp16",
        "NHWGC_GKYXC_NHWGK",
        "64x64x64",
        "2x2",
        "16x16x16",
        //    "4x4x4", // TODO: Enable this check
        "Default",
        "Intrawave",
        "CShuffleEpilogue",
        "pipeline_AgBgCrCompV3",
        "DoubleSmemBuffer_0",
        "NumWaveGroups_1",
        "MergedGroups_1",
        "SplitImage_0",
        "ExplicitGemm_0",
    });
}

TEST(ElementWiseOp, CreateBwdWeightTwoStageElementwiseOp)
{
    cku::run_ck_tile_test<ElementwiseOpBuilder>({"elementwise_kernel",
                                                 "4096_256_4_4_64_4_256",
                                                 "UnaryConvert",
                                                 "kPad_1",
                                                 "ElementWiseDefaultPolicy"});
}

TEST(BwdWeight_2D_FP16_NHWGC, Execution)
{
    auto inputs    = ckt::alloc_inputs(Args);
    auto outputs   = ckt::alloc_outputs(Args);
    auto reference = ckt::alloc_outputs(Args);

    ckt::init_inputs(Args, inputs.get());

    auto conv = Instance{};
    EXPECT_THAT(ckt::run(conv, Args, inputs.get(), outputs.get()), SuccessfulRun());

    auto ref_conv = Reference{};
    EXPECT_THAT(ckt::run(ref_conv, Args, inputs.get(), reference.get()), SuccessfulRun());

    EXPECT_THAT(outputs.get(), MatchesReference(Args, reference.get()));
}

TEST(BwdWeight_TwoStage_2D_FP16_NHWGC, Execution)
{
    auto inputs    = ckt::alloc_inputs(Args);
    auto outputs   = ckt::alloc_outputs(Args);
    auto reference = ckt::alloc_outputs(Args);

    ckt::init_inputs(Args, inputs.get());

    auto conv           = TwoStageInstance{};
    auto elementwise_op = ElementwiseOpInstance{};

    EXPECT_THAT(ckt::run(conv, elementwise_op, Args, inputs.get(), outputs.get()), SuccessfulRun());

    auto ref_conv = Reference{};
    EXPECT_THAT(ckt::run(ref_conv, Args, inputs.get(), reference.get()), SuccessfulRun());

    EXPECT_THAT(outputs.get(), MatchesReference(Args, reference.get()));
}
