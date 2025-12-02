// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

/// This file defines some common utilities that can be used and
/// structs that can be overridden by more specialized testing
/// utilities.

/// This file defines common types used for the general testing convention
/// of CK using CK Builder. A number of types which are parameterized over
/// a signature are defined here: Each of these should be specialized for
/// the particular types of signatures that we want to test, restricted using
/// concepts where applicable. Each of these structures has a short
/// description explaining its use and how it should be defined. Also see
/// the high-level testing description in `ck_tile/builder/testing/README.md`.
///
/// This file also defines a few common utilities which are used throughout
/// the CK Builder testing library.

namespace ck_tile::builder::test {

template <auto SIGNATURE>
struct Args;

template <auto SIGNATURE>
struct Inputs;

template <auto SIGNATURE>
struct Outputs;

template <auto SIGNATURE>
struct UniqueInputs;

template <auto SIGNATURE>
struct UniqueOutputs;

template <auto SIGNATURE>
UniqueInputs<SIGNATURE> alloc_inputs(const Args<SIGNATURE>&);

template <auto SIGNATURE>
UniqueInputs<SIGNATURE> alloc_outputs(const Args<SIGNATURE>&);

template <auto SIGNATURE, typename Conv>
void run(Conv& conv,
         const Args<SIGNATURE>& args,
         const Inputs<SIGNATURE>& inputs,
         const Outputs<SIGNATURE>& outputs);

/// This structure describes a 1-, 2-, or 3-D extent. Its used to
/// communicate 1-, 2- or 3-D sizes and strides of tensors.
template <int SPATIAL_DIM>
struct Extent;

template <>
struct Extent<1>
{
    size_t width = 1;
};

template <>
struct Extent<2>
{
    size_t width  = 1;
    size_t height = 1;
};

template <>
struct Extent<3>
{
    size_t width  = 1;
    size_t height = 1;
    size_t depth  = 1;
};

using Extent1D = Extent<1>;
using Extent2D = Extent<2>;
using Extent3D = Extent<3>;

} // namespace ck_tile::builder::test
