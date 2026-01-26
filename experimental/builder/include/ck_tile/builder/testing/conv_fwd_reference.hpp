// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/conv_fwd.hpp"
#include <stdexcept>
#include <vector>

/// This file contains the implementation details for invoking/testing
/// grouped convolution operations using the reference implementation.
/// The main item is the `run()` function, which is the primary way to
/// invoke the reference execution mechanism.
/// The implementation of this file mostly looks like `conv_fwd_ck.hpp`,
/// but its made specific to the reference implementation, which is
/// invoked in a slightly different way.

namespace ck_tile::builder::test {

/// @brief Concept for checking whether this is the reference convolution
/// implementation.
///
/// This concept is used to tell whether a convolution implementation is
/// likely to be the reference implementation - that is, whether we should
/// invoke it like the reference kernel. This is mainly used with `run()` to
/// differentiate which implementation that should be invoked.
///
/// - SIGNATURE is the operation signature.
/// - Conv is a convolution instance created by the CK Builder API.
template <typename Conv, auto SIGNATURE>
concept RefConvInstance = requires(Conv& conv,
                                   const void* input,
                                   const void* weight,
                                   void* output,
                                   ck::utils::conv::ConvParam param) {
    { conv.Run(input, weight, output, param) };
};

/// @brief `run()` specialization for forward convolution and the reference
/// implementation.
///
/// @tparam SIGNATURE Forward convolution signature.
/// @throws std::runtime_error if the arguments weren't actually valid for the
/// operation. This should be caught and reported by the testing framework.
///
/// @return std::tuple<bool, float> - whether the problem is supported and
///         kernel execution time (0.0f for reference).
/// @see run()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> &&
             // TODO: Maybe we can unify this implementation for bwd/weight too?
             // for now, just concern outselves with reference and see when the
             // rest of the bwd/weight plumbing is there.
             ConvDirectionIsForward<SIGNATURE>
std::tuple<bool, float> run(RefConvInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs)
{
    // We don't want to compute the output dims manually, just get
    // them via the existing infrastructure
    const auto param = args.to_ck_conv_param();

    // TODO: The reference convolution is currently missing a few features.
    // Just throw for now, but regard these as TODO items that should be resolved
    // eventually.

    if(!args.make_input_descriptor().is_packed())
    {
        std::cout << "TODO: Support non-packed input tensor in reference conv" << std::endl;
        return std::make_tuple(false, 0.0f);
    }
    if(!args.make_weight_descriptor().is_packed())
    {
        std::cout << "TODO: Support non-packed weight tensor in reference conv" << std::endl;
        return std::make_tuple(false, 0.0f);
    }
    if(!args.make_output_descriptor().is_packed())
    {
        std::cout << "TODO: Support non-packed output tensor in reference conv" << std::endl;
        return std::make_tuple(false, 0.0f);
    }

    conv.Run(inputs.input, inputs.weight, outputs.output, param);
    return std::make_tuple(true, 0.0f);
}

} // namespace ck_tile::builder::test
