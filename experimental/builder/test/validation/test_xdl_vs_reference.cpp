// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/testing/conv_fwd_ck.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "impl/conv_algorithm_types.hpp"
#include "utils/ckb_conv_test_configs.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

namespace {

constexpr auto SIGNATURE_FWD_2D_FP16 =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNHWK}}};

constexpr auto SIGNATURE_FWD_2D_BF16 =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNHWK}}};

constexpr auto XDL_ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
                                   .with_thread_block(cku::FwdThreadBlock_256_128x128x32)
                                   .with_gemm_config(cku::FwdGemmParams_Xdl_2x2_per_wave)
                                   .with_transfer(cku::FwdTransfer_4x64x1)
                                   .with_specializations(ckb::ConvFwdSpecialization::DEFAULT,
                                                         ckb::GemmSpecialization::MNKPadding)
                                   .with_block_gemm(cku::BlockGemmDesc_v3_intrawave);

constexpr auto REFERENCE_ALGORITHM = ckt::ConvAlgorithm_Reference{};

template <auto SIGNATURE, typename RefConv>
void run_reference_fwd(RefConv& ref_conv,
                       const ckt::Args<SIGNATURE>& args,
                       ckt::Inputs<SIGNATURE> inputs,
                       ckt::Outputs<SIGNATURE> outputs)
{
    static_assert(ckb::ValidConvSignature<SIGNATURE>);
    static_assert(ckb::ConvDirectionIsForward<SIGNATURE>);

    constexpr int spatial_dim = SIGNATURE.spatial_dim;

    using DataT = typename ckb::factory::internal::DataTypeToCK<SIGNATURE.data_type>::type;

    const auto to_vec = [](const auto& extent) {
        if constexpr(spatial_dim == 1)
        {
            return std::vector<ck_tile::long_index_t>{
                static_cast<ck_tile::long_index_t>(extent.width),
            };
        }
        else if constexpr(spatial_dim == 2)
        {
            return std::vector<ck_tile::long_index_t>{
                static_cast<ck_tile::long_index_t>(extent.height),
                static_cast<ck_tile::long_index_t>(extent.width),
            };
        }
        else
        {
            return std::vector<ck_tile::long_index_t>{
                static_cast<ck_tile::long_index_t>(extent.depth),
                static_cast<ck_tile::long_index_t>(extent.height),
                static_cast<ck_tile::long_index_t>(extent.width),
            };
        }
    };

    const int G = static_cast<int>(args.lengths.groups);
    const int N = static_cast<int>(args.lengths.batch_size);
    const int C = static_cast<int>(args.lengths.input_channels);
    const int K = static_cast<int>(args.lengths.output_channels);

    const auto input_spatial  = to_vec(args.lengths.image);
    const auto filter_spatial = to_vec(args.lengths.filter);

    const auto out_desc    = args.make_output_descriptor();
    const auto out_lengths = out_desc.get_lengths();

    std::vector<ck_tile::long_index_t> output_spatial;
    output_spatial.reserve(static_cast<size_t>(spatial_dim));
    for(int i = 0; i < spatial_dim; ++i)
    {
        output_spatial.push_back(static_cast<ck_tile::long_index_t>(out_lengths[3 + i]));
    }

    const auto strides   = to_vec(args.filter_strides);
    const auto dilations = to_vec(args.filter_dilation);
    const auto left_pads = to_vec(args.input_left_pad);

    ref_conv.Run(reinterpret_cast<const DataT*>(inputs.input),
                 reinterpret_cast<const DataT*>(inputs.weight),
                 reinterpret_cast<DataT*>(outputs.output),
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

template <auto SIGNATURE>
void expect_close_fwd(const ckt::Args<SIGNATURE>& args,
                      ckt::Outputs<SIGNATURE> actual,
                      ckt::Outputs<SIGNATURE> expected,
                      double rtol,
                      double atol)
{
    ckt::ValidationReport report;
    report.check(
        "output", args.make_output_descriptor(), actual.output, expected.output, rtol, atol);

    const auto errors = report.get_errors();
    EXPECT_TRUE(errors.empty());
}

template <auto SIGNATURE>
ckt::Args<SIGNATURE> make_small_fwd_args()
{
    return {
        .lengths =
            {
                .batch_size      = 2,
                .groups          = 1,
                .input_channels  = 32,
                .output_channels = 32,
                .image =
                    {
                        .width  = 7,
                        .height = 7,
                    },
                .filter =
                    {
                        .width  = 3,
                        .height = 3,
                    },
            },
        .filter_strides     = {.width = 1, .height = 1},
        .filter_dilation    = {.width = 1, .height = 1},
        .input_left_pad     = {.width = 1, .height = 1},
        .input_right_pad    = {.width = 1, .height = 1},
        .a_elementwise_op   = {},
        .b_elementwise_op   = {},
        .cde_elementwise_op = {},
    };
}

} // namespace

TEST(XdlVsReference, Forward2D_XDL_FP16_Small)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        GTEST_SKIP() << "unsupported architecture";
    }

    const auto args = make_small_fwd_args<SIGNATURE_FWD_2D_FP16>();

    auto inputs      = ckt::alloc_inputs(args);
    auto outputs     = ckt::alloc_outputs(args);
    auto ref_outputs = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    using XdlKernel = ckb::ConvBuilder<SIGNATURE_FWD_2D_FP16, XDL_ALGORITHM>::Instance;
    using RefKernel = ckb::ConvBuilder<SIGNATURE_FWD_2D_FP16, REFERENCE_ALGORITHM>::Instance;

    auto xdl = XdlKernel{};
    ckt::run(xdl, args, inputs.get(), outputs.get());

    auto ref = RefKernel{};
    run_reference_fwd<SIGNATURE_FWD_2D_FP16>(ref, args, inputs.get(), ref_outputs.get());

    expect_close_fwd<SIGNATURE_FWD_2D_FP16>(args, outputs.get(), ref_outputs.get(), 1e-3, 1e-3);
}

TEST(XdlVsReference, Forward2D_XDL_BF16_Small)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        GTEST_SKIP() << "unsupported architecture";
    }

    const auto args = make_small_fwd_args<SIGNATURE_FWD_2D_BF16>();

    auto inputs      = ckt::alloc_inputs(args);
    auto outputs     = ckt::alloc_outputs(args);
    auto ref_outputs = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    using XdlKernel = ckb::ConvBuilder<SIGNATURE_FWD_2D_BF16, XDL_ALGORITHM>::Instance;
    using RefKernel = ckb::ConvBuilder<SIGNATURE_FWD_2D_BF16, REFERENCE_ALGORITHM>::Instance;

    auto xdl = XdlKernel{};
    ckt::run(xdl, args, inputs.get(), outputs.get());

    auto ref = RefKernel{};
    run_reference_fwd<SIGNATURE_FWD_2D_BF16>(ref, args, inputs.get(), ref_outputs.get());

    expect_close_fwd<SIGNATURE_FWD_2D_BF16>(args, outputs.get(), ref_outputs.get(), 1e-2, 1e-2);
}
