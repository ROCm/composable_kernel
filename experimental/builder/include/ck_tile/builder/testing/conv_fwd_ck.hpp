// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <span>
#include <cstddef>

#include "ck_tile/builder/testing/conv_fwd.hpp"

/// This file contains the implementation details for invoking/testing
/// grouped convolution operations in old CK. The main item is the
/// `run()` function, which is the main implementation used to invoke
/// CK grouped forward convolution kernels.

namespace ck_tile::builder::test {

/// @brief Concept for checking whether a convolution is invoked like old CK.
///
/// This concept is used to tell whether a convolution implementation is
/// likely to be an "old CK" implementation - that is, whether we should
/// invoke it as an old CK kernel. This is mainly used with `run()` to
/// differentiate which implementation that should be invoked.
///
/// - SIGNATURE is the operation signature.
/// - Conv is a convolution instance created by the CK Builder API.
template <auto SIGNATURE, typename Conv>
concept IsCkConvInstance =
    // TODO: This should be implemented by converting the signature into the
    // type parameters for DeviceGroupedConvFwdMultipleABD. For now, just leave
    // it empty. Improve when needed, you get the point. Also we should probably
    // move this to the ck conv factory helper.
    true;

/// @brief `run()` specialization for forward convolution and old CK.
///
/// @tparam SIGNATURE Forward convolution signature.
/// @throws std::runtime_error if the arguments werent actually valid for the
/// operation. This should be caught and reported by the testing framework.
///
/// @see run()
template <auto SIGNATURE, typename Conv>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE> &&
             IsCkConvInstance<SIGNATURE, Conv>
void run(Conv& conv,
         const Args<SIGNATURE>& args,
         const Inputs<SIGNATURE>& inputs,
         const Outputs<SIGNATURE>& outputs)
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
