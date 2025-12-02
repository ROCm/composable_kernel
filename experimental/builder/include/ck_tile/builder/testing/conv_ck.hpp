// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <span>
#include <cstddef>

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/testing/conv_args.hpp"

/// This file contains the implementation details for invoking/testing
/// grouped convolution operations in old CK. The main item is the
/// ckt::run function, which is the main implementation used to invoke
/// CK grouped forward convolution kernels.

namespace ck_tile::builder::test {

/// This concept is used to tell whether a convolution implementation is likely to
/// be an "old CK" implementation - that is, whether we should invoke it as an old
/// CK kernel. This is mainly used with ckt::run() to differentiate the implementation
/// that should be called.
template <auto SIGNATURE, typename Conv>
concept IsCkConvInstance =
    // TODO: This should be implemented by converting the signature into the
    // type parameters for DeviceGroupedConvFwdMultipleABD. For now, just leave
    // it empty. Improve when needed, you get the point. Also we should probably
    // move this to the ck conv factory helper.
    true;

template <auto SIGNATURE, typename Conv>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE> &&
             IsCkConvInstance<SIGNATURE, Conv>
void run(Conv& conv,
         const ConvArgs<SIGNATURE>& args,
         const ConvInputs<SIGNATURE>& inputs,
         const ConvOutputs<SIGNATURE>& outputs)
{
    constexpr auto spatial_dim = SIGNATURE.spatial_dim;

    const auto copy = [](const auto& src, auto& dst) {
        std::copy(src.begin(), src.end(), dst.begin());
    };

    const auto to_ck_lengths = [&](const auto& src) {
        std::array<ck::index_t, spatial_dim + 3> result;
        copy(src, result);
        return result;
    };

    const auto to_ck_extent = [&](const auto& extent) {
        std::array<ck::index_t, spatial_dim> result;
        copy(extent, result);
        return result;
    };

    const auto param = args.to_ck_conv_param();

    const auto input_desc  = args.make_input_descriptor();
    const auto weight_desc = args.make_weight_descriptor();
    const auto output_desc = args.make_output_descriptor();

    auto ck_args = conv.MakeArgument(inputs.input,
                                     inputs.weight,
                                     {},
                                     outputs.output,
                                     to_ck_lengths(input_desc.get_lengths()),
                                     to_ck_lengths(input_desc.get_strides()),
                                     to_ck_lengths(weight_desc.get_lengths()),
                                     to_ck_lengths(weight_desc.get_strides()),
                                     {},
                                     {},
                                     to_ck_lengths(output_desc.get_lengths()),
                                     to_ck_lengths(output_desc.get_strides()),
                                     to_ck_extent(param.conv_filter_strides_),
                                     to_ck_extent(param.conv_filter_dilations_),
                                     to_ck_extent(param.input_left_pads_),
                                     to_ck_extent(param.input_right_pads_),
                                     args.a_elementwise_op,
                                     args.b_elementwise_op,
                                     args.cde_elementwise_op);

    if(!conv.IsSupportedArgument(ck_args))
    {
        throw std::runtime_error("invalid argument");
    }

    conv.MakeInvoker().Run(ck_args, {});
}

} // namespace ck_tile::builder::test
