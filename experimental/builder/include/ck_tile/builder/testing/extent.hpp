// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::builder::test {

/// This structure describes a 1-, 2-, or 3-D extent. Its used to
/// communicate 1-, 2- or 3-D sizes and strides of tensors.
/// Depending on the dimension, the structure will have the `width`,
/// `height`, and `depth` fields available.
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

} // namespace ck_tile::builder::test
