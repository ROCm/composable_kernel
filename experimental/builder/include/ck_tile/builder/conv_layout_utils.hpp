// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory_internal
{

struct EmptyBiasLayout 
{
    using DsLayout = ck::Tuple<>;
    using DsDataTypes = ck::Tuple<>;
};

template <auto Layout>
consteval bool IsGenericBiasLayoutActive() {
    return requires { typename std::integral_constant<BiasLayout, Layout._bias_layout>; };
}

template <auto Layout, size_t SPATIAL_DIM>
consteval auto GetBiasLayoutValue()
{
    if constexpr (IsGenericBiasLayoutActive<Layout>())
    {
        constexpr auto val = Layout._bias_layout;
        if constexpr (val == BiasLayout::G_K_strided)
            return ck::tensor_layout::convolution::G_K{};
        else if constexpr (val == BiasLayout::GC)
            return ck::tensor_layout::convolution::GC{};
        else if constexpr (val == BiasLayout::G_C_strided)
            return ck::tensor_layout::convolution::G_C{};
        else
            static_assert(false, "Unsupported generic bias layout");
    }
    else
    {
        constexpr auto out_layout = Layout._conv_output_layout;
        
        if constexpr (SPATIAL_DIM == 1)
        {
            constexpr auto val = out_layout._1d;
            if constexpr (val == ConvOutputLayout1D::NWGK) return ck::tensor_layout::convolution::NWGK{};
            else if constexpr (val == ConvOutputLayout1D::NGKW) return ck::tensor_layout::convolution::NGKW{};
            else if constexpr (val == ConvOutputLayout1D::GNWK) return ck::tensor_layout::convolution::GNWK{};
        }
        else if constexpr (SPATIAL_DIM == 2)
        {
            constexpr auto val = out_layout._2d;
            if constexpr (val == ConvOutputLayout2D::NHWGK) return ck::tensor_layout::convolution::NHWGK{};
            else if constexpr (val == ConvOutputLayout2D::GNHWK) return ck::tensor_layout::convolution::GNHWK{};
            else if constexpr (val == ConvOutputLayout2D::NGKHW) return ck::tensor_layout::convolution::NGKHW{};
        }
        else if constexpr (SPATIAL_DIM == 3)
        {
            constexpr auto val = out_layout._3d;
            if constexpr (val == ConvOutputLayout3D::NDHWGK) return ck::tensor_layout::convolution::NDHWGK{};
            else if constexpr (val == ConvOutputLayout3D::GNDHWK) return ck::tensor_layout::convolution::GNDHWK{};
            else if constexpr (val == ConvOutputLayout3D::NGKDHW) return ck::tensor_layout::convolution::NGKDHW{};
        }
    }
}

template <auto BiasLayoutsArray, size_t SPATIAL_DIM, size_t... Indices>
consteval auto GetBiasLayoutTuple(std::index_sequence<Indices...>)
{
    return ck::Tuple<decltype(GetBiasLayoutValue<BiasLayoutsArray[Indices], SPATIAL_DIM>())...>{};
}

// TODO: Remove hardcoding of bhalf_t
template <size_t N, size_t... Is>
consteval auto GetBiasTypesTuple(std::index_sequence<Is...>)
{
    return ck::Tuple<decltype((void(Is), ck::bhalf_t{}))...>{}; 
}

template <auto BiasLayoutValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM>)
struct ConvBiasTensorLayouts
{
    static constexpr auto Size = BiasLayoutValue.size();
    
    using DsLayout = decltype(GetBiasLayoutTuple<BiasLayoutValue, SPATIAL_DIM>(std::make_index_sequence<Size>{}));
    using DsDataTypes = decltype(GetBiasTypesTuple<Size>(std::make_index_sequence<Size>{}));
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

template <auto Layout, size_t SPATIAL_DIM>
consteval auto GetCKInputLayout()
{
    if constexpr(SPATIAL_DIM == 1)
    {
        constexpr auto val = Layout._1d;
        if constexpr(val == ConvInputLayout1D::NWGC)
            return ck::tensor_layout::convolution::NWGC{};
        else if constexpr(val == ConvInputLayout1D::NGCW)
            return ck::tensor_layout::convolution::NGCW{};
        else if constexpr(val == ConvInputLayout1D::GNWC)
            return ck::tensor_layout::convolution::GNWC{};
    }
    else if constexpr(SPATIAL_DIM == 2)
    {
        constexpr auto val = Layout._2d;
        if constexpr(val == ConvInputLayout2D::NGCHW)
            return ck::tensor_layout::convolution::NGCHW{};
        else if constexpr(val == ConvInputLayout2D::NHWGC)
            return ck::tensor_layout::convolution::NHWGC{};
        else if constexpr(val == ConvInputLayout2D::GNHWC)
            return ck::tensor_layout::convolution::GNHWC{};
    }
    else if constexpr(SPATIAL_DIM == 3)
    {
        constexpr auto val = Layout._3d;
        if constexpr(val == ConvInputLayout3D::NGCDHW)
            return ck::tensor_layout::convolution::NGCDHW{};
        else if constexpr(val == ConvInputLayout3D::NDHWGC)
            return ck::tensor_layout::convolution::NDHWGC{};
        else if constexpr(val == ConvInputLayout3D::GNDHWC)
            return ck::tensor_layout::convolution::GNDHWC{};
    }
}

template <auto Layout, size_t SPATIAL_DIM>
consteval auto GetCKWeightLayout()
{
    if constexpr(SPATIAL_DIM == 1)
    {
        constexpr auto val = Layout._1d;
        if constexpr(val == ConvWeightLayout1D::GKXC)
            return ck::tensor_layout::convolution::GKXC{};
        else if constexpr(val == ConvWeightLayout1D::GKCX)
            return ck::tensor_layout::convolution::GKCX{};
    }
    else if constexpr(SPATIAL_DIM == 2)
    {
        constexpr auto val = Layout._2d;
        if constexpr(val == ConvWeightLayout2D::GKYXC)
            return ck::tensor_layout::convolution::GKYXC{};
        else if constexpr(val == ConvWeightLayout2D::GKCYX)
            return ck::tensor_layout::convolution::GKCYX{};
    }
    else if constexpr(SPATIAL_DIM == 3)
    {
        constexpr auto val = Layout._3d;
        if constexpr(val == ConvWeightLayout3D::GKCZYX)
            return ck::tensor_layout::convolution::GKCZYX{};
        else if constexpr(val == ConvWeightLayout3D::GKZYXC)
            return ck::tensor_layout::convolution::GKZYXC{};
    }
}

template <auto Layout, size_t SPATIAL_DIM>
consteval auto GetCKOutputLayout()
{
    if constexpr(SPATIAL_DIM == 1)
    {
        constexpr auto val = Layout._1d;
        if constexpr(val == ConvOutputLayout1D::NWGK)
            return ck::tensor_layout::convolution::NWGK{};
        else if constexpr(val == ConvOutputLayout1D::NGKW)
            return ck::tensor_layout::convolution::NGKW{};
        else if constexpr(val == ConvOutputLayout1D::GNWK)
            return ck::tensor_layout::convolution::GNWK{};
    }
    else if constexpr(SPATIAL_DIM == 2)
    {
        constexpr auto val = Layout._2d;
        if constexpr(val == ConvOutputLayout2D::NGKHW)
            return ck::tensor_layout::convolution::NGKHW{};
        else if constexpr(val == ConvOutputLayout2D::NHWGK)
            return ck::tensor_layout::convolution::NHWGK{};
        else if constexpr(val == ConvOutputLayout2D::GNHWK)
            return ck::tensor_layout::convolution::GNHWK{};
    }
    else if constexpr(SPATIAL_DIM == 3)
    {
        constexpr auto val = Layout._3d;
        if constexpr(val == ConvOutputLayout3D::NGKDHW)
            return ck::tensor_layout::convolution::NGKDHW{};
        else if constexpr(val == ConvOutputLayout3D::NDHWGK)
            return ck::tensor_layout::convolution::NDHWGK{};
        else if constexpr(val == ConvOutputLayout3D::GNDHWK)
            return ck::tensor_layout::convolution::GNDHWK{};
    }
}

template <auto InputLayoutValue, auto WeightLayoutValue, auto OutputLayoutValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM> 
        && ValidConvInputLayoutForSpatialDim<InputLayoutValue, SPATIAL_DIM>
        && ValidConvWeightLayoutForSpatialDim<WeightLayoutValue, SPATIAL_DIM>
        && ValidConvOutputLayoutForSpatialDim<OutputLayoutValue, SPATIAL_DIM>)
struct ConvTensorLayouts
{
    static_assert(DIR == ConvDirection::FORWARD, "Only Forward convolution is supported.");
    using ALayout = decltype(GetCKInputLayout<InputLayoutValue, SPATIAL_DIM>());
    using BLayout = decltype(GetCKWeightLayout<WeightLayoutValue, SPATIAL_DIM>());
    using ELayout = decltype(GetCKOutputLayout<OutputLayoutValue, SPATIAL_DIM>());
};

template <auto Layout, size_t SPATIAL_DIM, ConvDirection DIR>
consteval auto GetTensorLayout()
{
    constexpr auto INPUT_LAYOUT = Layout.input_layout;
    constexpr auto WEIGHT_LAYOUT = Layout.weight_layout;  
    constexpr auto OUTPUT_LAYOUT = Layout.output_layout;

    return factory_internal::ConvTensorLayouts<INPUT_LAYOUT, WEIGHT_LAYOUT, OUTPUT_LAYOUT, SPATIAL_DIM, DIR>{};
}

}
