// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/conv/bwd_weight.hpp"
#include "ck_tile/builder/testing/conv/bwd_weight_ck.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"
#include "testing_utils.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

using enum ck_tile::builder::TensorLayout;
using ck_tile::test::MatchesReference;
using ck_tile::test::SuccessfulRun;

constexpr auto SIGNATURE = ckt::ConvSignature{.spatial_dim = 1,
                                              .direction   = ckb::ConvDirection::BACKWARD_WEIGHT,
                                              .data_type   = ckb::DataType::BF16,
                                              .accumulation_data_type = ckb::DataType::FP32,
                                              .input  = {.config = {.layout = GNWC}},
                                              .weight = {.config = {.layout = GKXC}},
                                              .output = {.config = {.layout = GNWK}}};

constexpr auto ALGORITHM =
    cku::ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle_V3{}
        .with_thread_block(cku::ThreadBlock_64_32x32x32)
        .with_gemm_config(cku::BwdGemmParams_Xdl_1x1_per_wave)
        .with_transfer(cku::BwdTransfer_4x8x1_4x16x1_v3)
        .with_bwd_specialization(ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0)
        .with_block_gemm(cku::BlockGemmDesc_v2_intrawave)
        .with_num_conv_groups_to_merge(1);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

using Reference = ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;

TEST(BwdWeight_1DBf16_CShuffle_V3, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    cku::run_test<Builder>({"DeviceGroupedConvBwdWeight_Xdl_CShuffleV3",
                            expected_transfer_parameters,
                            "Filter1x1Stride1Pad0",
                            "GNWC,GKXC,GNWK",
                            "PassThrough,PassThrough,PassThrough",
                            "Intrawave",
                            "v2"});
}

TEST(BwdWeight_1DBf16_CShuffle_V3, Execution)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        // Note: XDL kernel
        GTEST_SKIP() << "unsupported architecture";
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = 16,
                .groups          = 1,
                .input_channels  = 32,
                .output_channels = 48,
                .image           = {.width = 64},
                .filter          = {.width = 1},
            },
        .filter_strides     = {.width = 1},
        .filter_dilation    = {.width = 1},
        .input_left_pad     = {.width = 0},
        .input_right_pad    = {.width = 0},
        .a_elementwise_op   = {},
        .b_elementwise_op   = {},
        .cde_elementwise_op = {},
    };

    auto inputs    = ckt::alloc_inputs(args);
    auto outputs   = ckt::alloc_outputs(args);
    auto reference = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    auto conv = Instance{};
    EXPECT_THAT(ckt::run(conv, args, inputs.get(), outputs.get()), SuccessfulRun());

    auto ref_conv = Reference{};
    EXPECT_THAT(ckt::run(ref_conv, args, inputs.get(), reference.get()), SuccessfulRun());

    EXPECT_THAT(outputs.get(), MatchesReference(args, reference.get()));
}
