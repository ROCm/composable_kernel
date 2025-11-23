// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/tuple.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory::internal {

// Type mappings from the builder FwdGroupConvLayout enum classes to the CK tensor data types.
template <auto LayoutValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM> && ValidConvLayoutForSpatialDim<LayoutValue, SPATIAL_DIM>)
struct ConvTensorLayouts
{
    // This will trigger if a specialization for the given layout is not found.
    // We should always catch this in an earlier validation check.
    using Layout = decltype(LayoutValue);
    static_assert(sizeof(Layout) == 0,
                  "Internal error. Unsupported layout for convolution factory.");
};

// 1D Forward Convolution Layout Specializations
template <>
struct ConvTensorLayouts<GroupConvLayout1D::NWGC_GKXC_NWGK, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NWGC;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NWGK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout1D::NGCW_GKXC_NGKW, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCW;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout1D::GNWC_GKXC_GNWK, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNWC;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::GNWK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout1D::NGCW_GKCX_NGKW, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCW;
    using BLayout  = ck::tensor_layout::convolution::GKCX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout2D::NGCHW_GKYXC_NGKHW, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCHW;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout2D::NHWGC_GKYXC_NHWGK, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NHWGK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout2D::GNHWC_GKYXC_GNHWK, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNHWC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::GNHWK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout2D::NGCHW_GKCYX_NGKHW, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCHW;
    using BLayout  = ck::tensor_layout::convolution::GKCYX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout3D::NGCDHW_GKCZYX_NGKDHW, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCDHW;
    using BLayout  = ck::tensor_layout::convolution::GKCZYX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKDHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NDHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKZYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NDHWGK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNDHWC;
    using BLayout  = ck::tensor_layout::convolution::GKZYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::GNDHWK;
};

template <GroupConvLayout Layout, size_t SPATIAL_DIM, ConvDirection DIR>
consteval auto GetTensorLayout()
{

    if constexpr(SPATIAL_DIM == 1)
    {
        return internal::ConvTensorLayouts<Layout._1d, 1, DIR>{};
    }
    else if constexpr(SPATIAL_DIM == 2)
    {
        return internal::ConvTensorLayouts<Layout._2d, 2, DIR>{};
    }
    else if constexpr(SPATIAL_DIM == 3)
    {
        return internal::ConvTensorLayouts<Layout._3d, 3, DIR>{};
    }
    else
    {
        static_assert(false, "Unsupported spatial dimension for convolution layout.");
    }
}

} // namespace ck_tile::builder::factory::internal
