// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"
#include "ck_tile/builder/testing/conv_fwd_ck.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "testing_utils.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

using ck_tile::test::MatchesReference;

constexpr auto SIGNATURE =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNHWK}}};

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
                               .with_thread_block(cku::FwdThreadBlock_256_256x256x32)
                               .with_gemm_config(cku::FwdGemmParams_Xdl_4x4_per_wave)
                               .with_transfer(cku::FwdTransfer_4x64x1)
                               .with_specializations(ckb::ConvFwdSpecialization::DEFAULT,
                                                     ckb::GemmSpecialization::MNKPadding)
                               .with_block_gemm(cku::BlockGemmDesc_v3_intrawave);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

TEST(Fwd2DFp16_CShufV3_GNHWC, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    cku::run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                            expected_transfer_parameters,
                            "Default",
                            "Intrawave",
                            "v3",
                            "GNHWC,GKYXC,EmptyTuple,GNHWK",
                            "PassThrough,PassThrough,PassThrough",
                            "MNKPadding"});
}

TEST(Fwd2DFp16_CShufV3_GNHWC, EndToEnd)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        GTEST_SKIP() << "unsupported architecture";
    }

    constexpr auto ref_alg = ckt::ConvAlgorithm_Reference{};
    using RefBuilder       = ckb::ConvBuilder<SIGNATURE, ref_alg>;
    using RefInstance      = RefBuilder::Instance;

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

    auto inputs            = ckt::alloc_inputs(args);
    auto outputs           = ckt::alloc_outputs(args);
    auto reference_outputs = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    auto conv = Instance{};
    ckt::run(conv, args, inputs.get(), outputs.get());

    // Run GPU reference (Builder REFERENCE algorithm) and compare to optimized kernel output.
    // Note: The builder test harness does not yet provide a `ckt::run()` overload for reference
    // instances, so we call the reference convenience `Run()` API directly.
    {
        using DataT = typename ckb::factory::internal::DataTypeToCK<SIGNATURE.data_type>::type;

        const int G = static_cast<int>(args.lengths.groups);
        const int N = static_cast<int>(args.lengths.batch_size);
        const int C = static_cast<int>(args.lengths.input_channels);
        const int K = static_cast<int>(args.lengths.output_channels);

        // CK builder uses spatial ordering {H, W} for 2D.
        std::vector<ck_tile::long_index_t> input_spatial{
            static_cast<ck_tile::long_index_t>(args.lengths.image.height),
            static_cast<ck_tile::long_index_t>(args.lengths.image.width),
        };
        std::vector<ck_tile::long_index_t> filter_spatial{
            static_cast<ck_tile::long_index_t>(args.lengths.filter.height),
            static_cast<ck_tile::long_index_t>(args.lengths.filter.width),
        };

        const auto out_desc    = args.make_output_descriptor();
        const auto out_lengths = out_desc.get_lengths();
        std::vector<ck_tile::long_index_t> output_spatial{
            static_cast<ck_tile::long_index_t>(out_lengths[3]),
            static_cast<ck_tile::long_index_t>(out_lengths[4]),
        };

        std::vector<ck_tile::long_index_t> strides{
            static_cast<ck_tile::long_index_t>(args.filter_strides.height),
            static_cast<ck_tile::long_index_t>(args.filter_strides.width),
        };
        std::vector<ck_tile::long_index_t> dilations{
            static_cast<ck_tile::long_index_t>(args.filter_dilation.height),
            static_cast<ck_tile::long_index_t>(args.filter_dilation.width),
        };
        std::vector<ck_tile::long_index_t> left_pads{
            static_cast<ck_tile::long_index_t>(args.input_left_pad.height),
            static_cast<ck_tile::long_index_t>(args.input_left_pad.width),
        };

        auto ref = RefInstance{};
        ref.Run(reinterpret_cast<const DataT*>(inputs.get().input),
                reinterpret_cast<const DataT*>(inputs.get().weight),
                reinterpret_cast<DataT*>(reference_outputs.get().output),
                G,
                N,
                K,
                C,
                input_spatial,
                filter_spatial,
                output_spatial,
                strides,
                dilations,
                left_pads);
    }

    auto reference = reference_outputs.get();

    EXPECT_THAT(outputs.get(), MatchesReference(args, reference));
}
