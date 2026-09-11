// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_tile_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "testing_utils.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

using ck_tile::test::MatchesReference;
using ck_tile::test::SuccessfulRun;

constexpr auto SIGNATURE =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto ALGORITHM =
    cku::ConvAlgorithm_Tile_GroupedConvolutionKernel{}
        .with_tile_specializations(ckb::TileConvSpecialization::DEFAULT)
        .with_tile_thread_block(cku::FwdTileThreadBlock_64x64x64)
        .with_tile_block_gemm(cku::TileBlockGemmDesc_16x16_v3_intrawave)
        .with_tile_transfer(cku::FwdTileTransfer_4x4x4)
        .with_tile_optimizations(ckt::TileOptimizations{.num_groups_to_merge = 1,
                                                        .split_image         = false,
                                                        .explicit_gemm       = false,
                                                        .two_stage           = false});

using Builder   = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance  = Builder::Instance;
using Reference = ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;

TEST(Fwd2DFp16_CShufV3_NHWGC, EndToEnd)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        GTEST_SKIP() << "unsupported architecture";
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = 16,
                .groups          = 1,
                .input_channels  = 32,
                .output_channels = 48,
                .image =
                    {
                        .width  = 56,
                        .height = 64,
                    },
                .filter =
                    {
                        .width  = 3,
                        .height = 5,
                    },
            },
        .filter_strides     = {.width = 1, .height = 1},
        .filter_dilation    = {.width = 1, .height = 1},
        .input_left_pad     = {.width = 0, .height = 0},
        .input_right_pad    = {.width = 0, .height = 0},
        .a_elementwise_op   = {},
        .b_elementwise_op   = {},
        .cde_elementwise_op = {},
    };

    auto inputs    = alloc_inputs(args);
    auto outputs   = alloc_outputs(args);
    auto reference = alloc_outputs(args);
    ckt::init_inputs(args, inputs.get());

    auto conv = Instance{};
    EXPECT_THAT(ckt::run(conv, args, inputs.get(), outputs.get()), SuccessfulRun());

    auto ref_conv = Reference{};
    EXPECT_THAT(ckt::run(ref_conv, args, inputs.get(), reference.get()), SuccessfulRun());

    EXPECT_THAT(outputs.get(), MatchesReference(args, reference.get()));
}
