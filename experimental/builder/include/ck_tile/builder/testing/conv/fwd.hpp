// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/tensor_initialization.hpp"
#include "ck_tile/builder/testing/testing_reflect.hpp"
#include "ck_tile/builder/testing/conv/args.hpp"

/// This file deals with the forward-specific details of running grouped
/// convolution forward operations. It mainly defines the data structures
/// (`Input` and `Output`), initialization, and validation. Note that
/// for this operation specifically, many of the operations are implemented
/// automatically via testing_reflect.hpp.

namespace ck_tile::builder::test {

/// @brief `Inputs` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see Inputs
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
struct Inputs<SIGNATURE>
{
    void* input;
    void* weight;

    // See testing_reflect.hpp
    static void reflect(const Args<SIGNATURE>& args, const auto& inspect)
    {
        inspect("input", args.make_input_descriptor(), &Inputs<SIGNATURE>::input);
        inspect("weight", args.make_weight_descriptor(), &Inputs<SIGNATURE>::weight);
    }
};

/// @brief `Outputs` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see Outputs
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
struct Outputs<SIGNATURE>
{
    void* output;

    // See testing_reflect.hpp
    static void reflect(const Args<SIGNATURE>& args, const auto& inspect)
    {
        inspect("output", args.make_output_descriptor(), &Outputs<SIGNATURE>::output);
    }
};

/// @brief `init_inputs()` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see init_inputs()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
void init_inputs(const Args<SIGNATURE>& args, Inputs<SIGNATURE> inputs)
{
    init_tensor_buffer_uniform_fp(inputs.input, args.make_input_descriptor(), -2.0f, 2.0f);
    init_tensor_buffer_uniform_fp(inputs.weight, args.make_weight_descriptor(), -2.0f, 2.0f);
}

} // namespace ck_tile::builder::test
