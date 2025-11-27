// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// #include <concepts>
// #include <type_traits>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory_internal
{

struct EmptyBiasLayout 
{
    using DsLayout = ck::Tuple<>;
    using DsDataTypes = ck::Tuple<>;
};

// Type mappings from the builder ConvBiasLayout enum classes to the CK tensor data types.
template <auto BiasLayoutValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM>)
struct ConvBiasTensorLayouts
{
    // This will trigger if a specialization for the given layout is not found.
    // We should always catch this in an earlier validation check.
    using BiasLayoutType = decltype(BiasLayoutValue);
    static_assert(sizeof(BiasLayoutType) == 0,
                  "Internal error. Unsupported layout for convolution factory.");
};

constexpr std::array<ConvBiasLayout, 2> NHWGK_G_K_STRIDED_LAYOUT = {
    ConvBiasLayout{ConvOutputLayout2D::NHWGK}, 
    ConvBiasLayout{BiasLayout::G_K_strided}
};

template<>
struct ConvBiasTensorLayouts<NHWGK_G_K_STRIDED_LAYOUT, 2, ConvDirection::FORWARD>
{
    using DsLayout = ck::Tuple<ck::tensor_layout::convolution::NHWGK, ck::tensor_layout::convolution::G_K>;
    using DsDataTypes = ck::Tuple<ck::bhalf_t, ck::bhalf_t>;
};

template <auto Layout, size_t SPATIAL_DIM, ConvDirection DIR>
requires (HasBiasLayout<decltype(Layout)>)
consteval auto GetBiasTensorLayout()
{
    return factory_internal::ConvBiasTensorLayouts<Layout.bias_layout, SPATIAL_DIM, DIR>{};
}

template <auto Layout, size_t SPATIAL_DIM, ConvDirection DIR>
requires (!HasBiasLayout<decltype(Layout)>)
consteval auto GetBiasTensorLayout()
{
    return EmptyBiasLayout{};
}

// Type mappings from the builder ConvLayout enum classes to the CK tensor data types.
template <auto InputLayoutValue, auto WeightLayoutValue, auto OutputLayoutValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM> 
        && ValidConvInputLayoutForSpatialDim<InputLayoutValue, SPATIAL_DIM>
        && ValidConvWeightLayoutForSpatialDim<WeightLayoutValue, SPATIAL_DIM>
        && ValidConvOutputLayoutForSpatialDim<OutputLayoutValue, SPATIAL_DIM>)
struct ConvTensorLayouts
{
    // This will trigger if a specialization for the given layout is not found.
    // We should always catch this in an earlier validation check.
    using InputLayout = decltype(InputLayoutValue);
    using WeightLayout = decltype(WeightLayoutValue);
    using OutputLayout = decltype(OutputLayoutValue);
    static_assert(sizeof(InputLayout) == 0 && sizeof(WeightLayout) == 0 && sizeof(OutputLayout) == 0,
                  "Internal error. Unsupported layout for convolution factory.");
};

// 1D Forward Convolution Layout Specializations
template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout1D::NWGC}, 
                        ConvWeightLayout{ConvWeightLayout1D::GKXC}, 
                        ConvOutputLayout{ConvOutputLayout1D::NWGK}, 
                        1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NWGC;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using ELayout  = ck::tensor_layout::convolution::NWGK;
};

template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout1D::NGCW}, 
                        ConvWeightLayout{ConvWeightLayout1D::GKXC}, 
                        ConvOutputLayout{ConvOutputLayout1D::NGKW}, 
                        1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCW;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using ELayout  = ck::tensor_layout::convolution::NGKW;
};

template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout1D::GNWC}, 
                        ConvWeightLayout{ConvWeightLayout1D::GKXC}, 
                        ConvOutputLayout{ConvOutputLayout1D::GNWK}, 
                        1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNWC;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using ELayout  = ck::tensor_layout::convolution::GNWK;
};

template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout1D::NGCW}, 
                        ConvWeightLayout{ConvWeightLayout1D::GKCX}, 
                        ConvOutputLayout{ConvOutputLayout1D::NGKW}, 
                        1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCW;
    using BLayout  = ck::tensor_layout::convolution::GKCX;
    using ELayout  = ck::tensor_layout::convolution::NGKW;
};

// 2D Forward Convolution Layout Specializations
template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout2D::NGCHW}, 
                        ConvWeightLayout{ConvWeightLayout2D::GKYXC}, 
                        ConvOutputLayout{ConvOutputLayout2D::NGKHW}, 
                        2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCHW;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout2D::NHWGC}, 
                        ConvWeightLayout{ConvWeightLayout2D::GKYXC}, 
                        ConvOutputLayout{ConvOutputLayout2D::NHWGK}, 
                        2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using ELayout  = ck::tensor_layout::convolution::NHWGK;
};

template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout2D::GNHWC}, 
                        ConvWeightLayout{ConvWeightLayout2D::GKYXC}, 
                        ConvOutputLayout{ConvOutputLayout2D::GNHWK}, 
                        2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNHWC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using ELayout  = ck::tensor_layout::convolution::GNHWK;
};

template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout2D::NGCHW}, 
                        ConvWeightLayout{ConvWeightLayout2D::GKCYX}, 
                        ConvOutputLayout{ConvOutputLayout2D::NGKHW}, 
                        2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCHW;
    using BLayout  = ck::tensor_layout::convolution::GKCYX;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

// 3D Forward Convolution Layout Specializations
template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout3D::NGCDHW}, 
                        ConvWeightLayout{ConvWeightLayout3D::GKCZYX}, 
                        ConvOutputLayout{ConvOutputLayout3D::NGKDHW}, 
                        3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCDHW;
    using BLayout  = ck::tensor_layout::convolution::GKCZYX;
    using ELayout  = ck::tensor_layout::convolution::NGKDHW;
};

template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout3D::NDHWGC}, 
                        ConvWeightLayout{ConvWeightLayout3D::GKZYXC}, 
                        ConvOutputLayout{ConvOutputLayout3D::NDHWGK}, 
                        3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NDHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKZYXC;
    using ELayout  = ck::tensor_layout::convolution::NDHWGK;
};

template <>
struct ConvTensorLayouts<ConvInputLayout{ConvInputLayout3D::GNDHWC}, 
                        ConvWeightLayout{ConvWeightLayout3D::GKZYXC}, 
                        ConvOutputLayout{ConvOutputLayout3D::GNDHWK}, 
                        3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNDHWC;
    using BLayout  = ck::tensor_layout::convolution::GKZYXC;
    using ELayout  = ck::tensor_layout::convolution::GNDHWK;
};

template <ConvInputLayout INPUT_LAYOUT, 
          ConvWeightLayout WEIGHT_LAYOUT, 
          ConvOutputLayout OUTPUT_LAYOUT, 
          size_t SPATIAL_DIM, 
          ConvDirection DIR>
consteval auto GetTensorLayoutInternal()
{
    return factory_internal::ConvTensorLayouts<INPUT_LAYOUT, WEIGHT_LAYOUT, OUTPUT_LAYOUT, SPATIAL_DIM, DIR>{};
}

template <auto Layout, size_t SPATIAL_DIM, ConvDirection DIR>
consteval auto GetTensorLayout()
{
    constexpr auto INPUT_LAYOUT = Layout.input_layout;
    constexpr auto WEIGHT_LAYOUT = Layout.weight_layout;  
    constexpr auto OUTPUT_LAYOUT = Layout.output_layout;

    return GetTensorLayoutInternal<INPUT_LAYOUT, WEIGHT_LAYOUT, OUTPUT_LAYOUT, SPATIAL_DIM, DIR>();
}

}
