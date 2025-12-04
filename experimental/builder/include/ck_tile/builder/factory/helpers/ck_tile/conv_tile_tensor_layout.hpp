// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory::internal {

// Type mappings from the builder FwdGroupConvLayout enum classes to the CK Tile tensor data types.
template <auto LayoutValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM> && ValidConvLayoutForSpatialDim<LayoutValue, SPATIAL_DIM>)
struct TileConvTensorLayouts
{
    // This will trigger if a specialization for the given layout is not found.
    // We should always catch this in an earlier validation check.
    using Layout = decltype(LayoutValue);
    static_assert(sizeof(Layout) == 0,
                  "Internal error. Unsupported layout for convolution factory.");
};

// 1D Forward Convolution Layout Specializations
template <>
struct TileConvTensorLayouts<GroupConvLayout1D::NWGC_GKXC_NWGK, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck_tile::tensor_layout::convolution::NWGC;
    using BLayout  = ck_tile::tensor_layout::convolution::GKXC;
    using DsLayout = ck_tile::tuple<>;
    using ELayout  = ck_tile::tensor_layout::convolution::NWGK;
};

template <>
struct TileConvTensorLayouts<GroupConvLayout1D::GNWC_GKXC_GNWK, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck_tile::tensor_layout::convolution::GNWC;
    using BLayout  = ck_tile::tensor_layout::convolution::GKXC;
    using DsLayout = ck_tile::tuple<>;
    using ELayout  = ck_tile::tensor_layout::convolution::GNWK;
};

template <>
struct TileConvTensorLayouts<GroupConvLayout2D::NHWGC_GKYXC_NHWGK, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck_tile::tensor_layout::convolution::NHWGC;
    using BLayout  = ck_tile::tensor_layout::convolution::GKYXC;
    using DsLayout = ck_tile::tuple<>;
    using ELayout  = ck_tile::tensor_layout::convolution::NHWGK;
};

template <>
struct TileConvTensorLayouts<GroupConvLayout2D::GNHWC_GKYXC_GNHWK, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck_tile::tensor_layout::convolution::GNHWC;
    using BLayout  = ck_tile::tensor_layout::convolution::GKYXC;
    using DsLayout = ck_tile::tuple<>;
    using ELayout  = ck_tile::tensor_layout::convolution::GNHWK;
};

template <>
struct TileConvTensorLayouts<GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck_tile::tensor_layout::convolution::NDHWGC;
    using BLayout  = ck_tile::tensor_layout::convolution::GKZYXC;
    using DsLayout = ck_tile::tuple<>;
    using ELayout  = ck_tile::tensor_layout::convolution::NDHWGK;
};

template <>
struct TileConvTensorLayouts<GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck_tile::tensor_layout::convolution::GNDHWC;
    using BLayout  = ck_tile::tensor_layout::convolution::GKZYXC;
    using DsLayout = ck_tile::tuple<>;
    using ELayout  = ck_tile::tensor_layout::convolution::GNDHWK;
};

template <GroupConvLayout Layout, size_t SPATIAL_DIM, ConvDirection DIR>
consteval auto GetTileTensorLayout()
{

    if constexpr(SPATIAL_DIM == 1)
    {
        return internal::TileConvTensorLayouts<Layout._1d, 1, DIR>{};
    }
    else if constexpr(SPATIAL_DIM == 2)
    {
        return internal::TileConvTensorLayouts<Layout._2d, 2, DIR>{};
    }
    else if constexpr(SPATIAL_DIM == 3)
    {
        return internal::TileConvTensorLayouts<Layout._3d, 3, DIR>{};
    }
    else
    {
        static_assert(false, "Unsupported spatial dimension for convolution layout.");
    }
}

} // namespace ck_tile::builder::factory::internal
