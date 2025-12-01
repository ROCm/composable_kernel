// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory_internal {

struct EmptyAuxiliaryTensorLayout
{
    using DsLayout = ck::Tuple<>;
};

template <auto Layout>
consteval bool IsGenericBiasLayoutActive()
{
    return requires {
        typename std::integral_constant<BiasLayout, Layout._aux_tensor_layout._bias_layout>;
    };
}

template <auto Config, size_t SPATIAL_DIM>
consteval auto GetAuxiliaryTensorLayoutValue()
{
    constexpr auto Layout = Config.layout;
    if constexpr(IsGenericBiasLayoutActive<Layout>())
    {
        constexpr auto val = Layout._aux_tensor_layout._bias_layout;
        if constexpr(val == BiasLayout::G_K_strided)
            return ck::tensor_layout::convolution::G_K{};
        else if constexpr(val == BiasLayout::GC)
            return ck::tensor_layout::convolution::GC{};
        else if constexpr(val == BiasLayout::G_C_strided)
            return ck::tensor_layout::convolution::G_C{};
        else
            static_assert(false, "Unsupported generic bias layout");
    }
    else
    {
        constexpr auto out_layout = Layout._output_layout;

        if constexpr(SPATIAL_DIM == 1)
        {
            constexpr auto val = out_layout._1d;
            if constexpr(val == ConvOutputLayout1D::NWGK)
                return ck::tensor_layout::convolution::NWGK{};
            else if constexpr(val == ConvOutputLayout1D::NGKW)
                return ck::tensor_layout::convolution::NGKW{};
            else if constexpr(val == ConvOutputLayout1D::GNWK)
                return ck::tensor_layout::convolution::GNWK{};
        }
        else if constexpr(SPATIAL_DIM == 2)
        {
            constexpr auto val = out_layout._2d;
            if constexpr(val == ConvOutputLayout2D::NHWGK)
                return ck::tensor_layout::convolution::NHWGK{};
            else if constexpr(val == ConvOutputLayout2D::GNHWK)
                return ck::tensor_layout::convolution::GNHWK{};
            else if constexpr(val == ConvOutputLayout2D::NGKHW)
                return ck::tensor_layout::convolution::NGKHW{};
        }
        else if constexpr(SPATIAL_DIM == 3)
        {
            constexpr auto val = out_layout._3d;
            if constexpr(val == ConvOutputLayout3D::NDHWGK)
                return ck::tensor_layout::convolution::NDHWGK{};
            else if constexpr(val == ConvOutputLayout3D::GNDHWK)
                return ck::tensor_layout::convolution::GNDHWK{};
            else if constexpr(val == ConvOutputLayout3D::NGKDHW)
                return ck::tensor_layout::convolution::NGKDHW{};
        }
    }
}

template <auto AuxiliaryTensorConfigsArray, size_t SPATIAL_DIM, size_t... Indices>
consteval auto GetAuxiliaryTensorLayoutTuple(std::index_sequence<Indices...>)
{
    // TODO: Use std::tuple instead of ck::Tuple
    return ck::Tuple<decltype(GetAuxiliaryTensorLayoutValue<AuxiliaryTensorConfigsArray[Indices],
                                                            SPATIAL_DIM>())...>{};
}

template <auto AuxiliaryTensorConfigsValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM>)
struct AuxiliaryTensorLayouts
{
    static constexpr auto Size = AuxiliaryTensorConfigsValue.size();
    using DsLayout =
        decltype(GetAuxiliaryTensorLayoutTuple<AuxiliaryTensorConfigsValue, SPATIAL_DIM>(
            std::make_index_sequence<Size>{}));
};

// TODO: Currently only the ouput tensor can have auxiliary tensors (e.g., bias).
template <auto Signature, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(HasElementwiseOpWithAuxiliaryOperands<decltype(Signature.output)>)
consteval auto GetAuxiliaryTensorLayouts()
{
    return AuxiliaryTensorLayouts<Signature.output.operation.auxiliary_operand_configs,
                                  SPATIAL_DIM,
                                  DIR>{};
}

template <auto Signature, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(!HasElementwiseOpWithAuxiliaryOperands<decltype(Signature.output)>)
consteval auto GetAuxiliaryTensorLayouts()
{
    return EmptyAuxiliaryTensorLayout{};
}

template <auto Layout, size_t SPATIAL_DIM>
consteval auto GetInputLayout()
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
consteval auto GetWeightLayout()
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
consteval auto GetOutputLayout()
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

template <auto InputLayoutValue,
          auto WeightLayoutValue,
          auto OutputLayoutValue,
          size_t SPATIAL_DIM,
          ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM> &&
             ValidConvInputLayoutForSpatialDim<InputLayoutValue, SPATIAL_DIM> &&
             ValidConvWeightLayoutForSpatialDim<WeightLayoutValue, SPATIAL_DIM> &&
             ValidConvOutputLayoutForSpatialDim<OutputLayoutValue, SPATIAL_DIM>)
struct ConvTensorLayouts
{
    static_assert(DIR == ConvDirection::FORWARD, "Only Forward convolution is supported.");
    using ALayout = decltype(GetInputLayout<InputLayoutValue, SPATIAL_DIM>());
    using BLayout = decltype(GetWeightLayout<WeightLayoutValue, SPATIAL_DIM>());
    using ELayout = decltype(GetOutputLayout<OutputLayoutValue, SPATIAL_DIM>());
};

template <auto Signature, size_t SPATIAL_DIM, ConvDirection DIR>
consteval auto GetTensorLayout()
{
    constexpr auto INPUT_LAYOUT  = Signature.input.config.layout._input_layout;
    constexpr auto WEIGHT_LAYOUT = Signature.weight.config.layout._weight_layout;
    constexpr auto OUTPUT_LAYOUT = Signature.output.config.layout._output_layout;

    return factory_internal::
        ConvTensorLayouts<INPUT_LAYOUT, WEIGHT_LAYOUT, OUTPUT_LAYOUT, SPATIAL_DIM, DIR>{};
}

} // namespace ck_tile::builder::factory_internal
