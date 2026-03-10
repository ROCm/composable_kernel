// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/fwd_ck.hpp"
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
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNHWK}}};

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3{}
                               .with_thread_block(cku::ThreadBlock_64_64x64x32)
                               .with_gemm_config(cku::GemmParamsABK1_Wmma_16x16_4x2_per_wave)
                               .with_transfer(cku::Transfer_4x16x1)
                               .with_fwd_specializations(ckb::ConvSpecialization::DEFAULT,
                                                         ckb::GemmSpecialization::MNKPadding)
                               .with_block_gemm(cku::BlockGemmDesc_v1_intrawave)
                               .with_num_conv_groups_to_merge(1);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

using Reference = ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;

TEST(Fwd2DFp16_WmmaV3_GNHWC, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    cku::run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3",
                            expected_transfer_parameters,
                            "Default",
                            "Intrawave",
                            "v1",
                            "GNHWC,GKYXC,EmptyTuple,GNHWK",
                            "PassThrough,PassThrough,PassThrough",
                            "MNKPadding"});
}

TEST(Fwd2DFp16_WmmaV3_GNHWC, Execution)
{
    if(!ck_tile::get_device_name().starts_with("gfx11") &&
       !ck_tile::get_device_name().starts_with("gfx12"))
    {
        // Note: WMMA kernel requires gfx11 or gfx12
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
