// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/tensor_initialization.hpp"
#include "ck_tile/builder/testing/testing_reflect.hpp"
#include "ck_tile/builder/testing/conv/args.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/error.hpp"

/// This file deals with the backward weight-specific details of running grouped
/// convolution backwards weight operations. It mainly defines the data
/// structures (`Input` and `Output`), initialization, and validation. Note
/// that for this operation specifically, many of the operations are
/// implemented automatically via testing_reflect.hpp.

namespace ck_tile::builder::test {

/// @brief `Inputs` specialization for backwards weight convolution.
///
/// @tparam SIGNATURE Backwards weight convolution signature.
///
/// @see Inputs
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsBackwardWeight<SIGNATURE>
struct Inputs<SIGNATURE>
{
    void* input;
    void* output;

    // See testing_reflect.hpp
    static void reflect(const Args<SIGNATURE>& args, const auto& inspect)
    {
        inspect("input", args.make_input_descriptor(), &Inputs<SIGNATURE>::input);
        inspect("output", args.make_output_descriptor(), &Inputs<SIGNATURE>::output);
    }
};

/// @brief `Outputs` specialization for backwards weight convolution.
///
/// @tparam SIGNATURE Backwards weight convolution signature.
///
/// @see Outputs
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsBackwardWeight<SIGNATURE>
struct Outputs<SIGNATURE>
{
    void* weight;

    // See testing_reflect.hpp
    static void reflect(const Args<SIGNATURE>& args, const auto& inspect)
    {
        inspect("weight", args.make_weight_descriptor(), &Outputs<SIGNATURE>::weight);
    }
};

/// @brief `init_inputs()` specialization for backwards convolution.
///
/// @tparam SIGNATURE Backwards weight convolution signature.
///
/// @see init_inputs()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsBackwardWeight<SIGNATURE>
void init_inputs(const Args<SIGNATURE>& args, Inputs<SIGNATURE> inputs)
{
    init_tensor_buffer_uniform_fp(inputs.input, args.make_input_descriptor(), -2.0f, 2.0f);
    init_tensor_buffer_uniform_fp(inputs.output, args.make_output_descriptor(), -2.0f, 2.0f);
}

} // namespace ck_tile::builder::test
