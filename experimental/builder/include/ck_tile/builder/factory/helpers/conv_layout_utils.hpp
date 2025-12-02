// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory_internal {

template <auto Layout>
struct LayoutToCK
{
    static_assert(sizeof(UnsupportedEnumValue<Layout>) == 0,
                  "Unsupported layout conversion to CK.");
};

// BiasLayout
template <> struct LayoutToCK<BiasLayout::G_K_strided> { using type = ck::tensor_layout::convolution::G_K; };
template <> struct LayoutToCK<BiasLayout::GC>          { using type = ck::tensor_layout::convolution::GC; };
template <> struct LayoutToCK<BiasLayout::G_C_strided> { using type = ck::tensor_layout::convolution::G_C; };

// Input 1D
template <> struct LayoutToCK<ConvInputLayout1D::NWGC> { using type = ck::tensor_layout::convolution::NWGC; };
template <> struct LayoutToCK<ConvInputLayout1D::NGCW> { using type = ck::tensor_layout::convolution::NGCW; };
template <> struct LayoutToCK<ConvInputLayout1D::GNWC> { using type = ck::tensor_layout::convolution::GNWC; };

// Input 2D
template <> struct LayoutToCK<ConvInputLayout2D::NGCHW> { using type = ck::tensor_layout::convolution::NGCHW; };
template <> struct LayoutToCK<ConvInputLayout2D::NHWGC> { using type = ck::tensor_layout::convolution::NHWGC; };
template <> struct LayoutToCK<ConvInputLayout2D::GNHWC> { using type = ck::tensor_layout::convolution::GNHWC; };

// Input 3D
template <> struct LayoutToCK<ConvInputLayout3D::NGCDHW> { using type = ck::tensor_layout::convolution::NGCDHW; };
template <> struct LayoutToCK<ConvInputLayout3D::NDHWGC> { using type = ck::tensor_layout::convolution::NDHWGC; };
template <> struct LayoutToCK<ConvInputLayout3D::GNDHWC> { using type = ck::tensor_layout::convolution::GNDHWC; };

// Weight 1D
template <> struct LayoutToCK<ConvWeightLayout1D::GKXC> { using type = ck::tensor_layout::convolution::GKXC; };
template <> struct LayoutToCK<ConvWeightLayout1D::GKCX> { using type = ck::tensor_layout::convolution::GKCX; };

// Weight 2D
template <> struct LayoutToCK<ConvWeightLayout2D::GKYXC> { using type = ck::tensor_layout::convolution::GKYXC; };
template <> struct LayoutToCK<ConvWeightLayout2D::GKCYX> { using type = ck::tensor_layout::convolution::GKCYX; };

// Weight 3D
template <> struct LayoutToCK<ConvWeightLayout3D::GKCZYX> { using type = ck::tensor_layout::convolution::GKCZYX; };
template <> struct LayoutToCK<ConvWeightLayout3D::GKZYXC> { using type = ck::tensor_layout::convolution::GKZYXC; };

// Output 1D
template <> struct LayoutToCK<ConvOutputLayout1D::NWGK> { using type = ck::tensor_layout::convolution::NWGK; };
template <> struct LayoutToCK<ConvOutputLayout1D::NGKW> { using type = ck::tensor_layout::convolution::NGKW; };
template <> struct LayoutToCK<ConvOutputLayout1D::GNWK> { using type = ck::tensor_layout::convolution::GNWK; };

// Output 2D
template <> struct LayoutToCK<ConvOutputLayout2D::NGKHW> { using type = ck::tensor_layout::convolution::NGKHW; };
template <> struct LayoutToCK<ConvOutputLayout2D::NHWGK> { using type = ck::tensor_layout::convolution::NHWGK; };
template <> struct LayoutToCK<ConvOutputLayout2D::GNHWK> { using type = ck::tensor_layout::convolution::GNHWK; };

// Output 3D
template <> struct LayoutToCK<ConvOutputLayout3D::NGKDHW> { using type = ck::tensor_layout::convolution::NGKDHW; };
template <> struct LayoutToCK<ConvOutputLayout3D::NDHWGK> { using type = ck::tensor_layout::convolution::NDHWGK; };
template <> struct LayoutToCK<ConvOutputLayout3D::GNDHWK> { using type = ck::tensor_layout::convolution::GNDHWK; };

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
        return typename LayoutToCK<Layout._aux_tensor_layout._bias_layout>::type{};
    }
    else
    {
        constexpr auto out_layout = Layout._output_layout;
        if constexpr(SPATIAL_DIM == 1)
            return typename LayoutToCK<out_layout._1d>::type{};
        else if constexpr(SPATIAL_DIM == 2)
            return typename LayoutToCK<out_layout._2d>::type{};
        else if constexpr(SPATIAL_DIM == 3)
            return typename LayoutToCK<out_layout._3d>::type{};
    }
}

template <auto AuxiliaryTensorConfigsArray, size_t SPATIAL_DIM, size_t... Indices>
consteval auto GetAuxiliaryTensorLayoutTuple(std::index_sequence<Indices...>)
{
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
        return typename LayoutToCK<Layout._1d>::type{};
    else if constexpr(SPATIAL_DIM == 2)
        return typename LayoutToCK<Layout._2d>::type{};
    else if constexpr(SPATIAL_DIM == 3)
        return typename LayoutToCK<Layout._3d>::type{};
}

template <auto Layout, size_t SPATIAL_DIM>
consteval auto GetWeightLayout()
{
    if constexpr(SPATIAL_DIM == 1)
        return typename LayoutToCK<Layout._1d>::type{};
    else if constexpr(SPATIAL_DIM == 2)
        return typename LayoutToCK<Layout._2d>::type{};
    else if constexpr(SPATIAL_DIM == 3)
        return typename LayoutToCK<Layout._3d>::type{};
}

template <auto Layout, size_t SPATIAL_DIM>
consteval auto GetOutputLayout()
{
    if constexpr(SPATIAL_DIM == 1)
        return typename LayoutToCK<Layout._1d>::type{};
    else if constexpr(SPATIAL_DIM == 2)
        return typename LayoutToCK<Layout._2d>::type{};
    else if constexpr(SPATIAL_DIM == 3)
        return typename LayoutToCK<Layout._3d>::type{};
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
