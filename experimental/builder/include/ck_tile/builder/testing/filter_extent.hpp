// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::builder::test {

/// This structure describes a 1-, 2-, or 3-D extent for convolution
/// filters. Its used to communicate 1-, 2- or 3-D sizes and strides
/// of tensors, specifically for convolution filters. Depending on the
/// dimension, the structure will have the `width`, `height`, and
/// `depth` fields available.
template <int SPATIAL_DIM>
struct FilterExtent;

template <>
struct FilterExtent<1>
{
    size_t width = 1;
};

template <>
struct FilterExtent<2>
{
    size_t width  = 1;
    size_t height = 1;
};

template <>
struct FilterExtent<3>
{
    size_t width  = 1;
    size_t height = 1;
    size_t depth  = 1;
};

template <int SPATIAL_DIM>
inline FilterExtent<SPATIAL_DIM> filter_extent_from_vector(const std::vector<std::size_t>& vec);

template <>
inline FilterExtent<1> filter_extent_from_vector<1>(const std::vector<std::size_t>& vec)
{
    return FilterExtent<1>{.width = vec[0]};
}

template <>
inline FilterExtent<2> filter_extent_from_vector<2>(const std::vector<std::size_t>& vec)
{
    return FilterExtent<2>{.width = vec[1], .height = vec[0]};
}

template <>
inline FilterExtent<3> filter_extent_from_vector<3>(const std::vector<std::size_t>& vec)
{
    return FilterExtent<3>{.width = vec[2], .height = vec[1], .depth = vec[0]};
}

} // namespace ck_tile::builder::test
