// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <variant>
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::test {

using namespace ck_tile::builder;

template <ConvInputBiasLayout... InputBiasLayouts>
struct ConvInputBiasLayouts
{
    std::array<ConvInputBiasLayout, sizeof...(InputBiasLayouts)> input_bias_layout{InputBiasLayouts...};

    template<InputBiasLayout Layout>
    constexpr auto with_layout() const
    {
        return ConvInputBiasLayouts<InputBiasLayouts..., ConvInputBiasLayout{Layout}>{};
    }
    
    template<ConvInputLayout2D Layout>
    constexpr auto with_layout() const
    {
        return ConvInputBiasLayouts<InputBiasLayouts..., ConvInputBiasLayout{Layout}>{};
    }
};

template <ConvOutputBiasLayout... OutputBiasLayouts>
struct ConvOutputBiasLayouts
{
    std::array<ConvOutputBiasLayout, sizeof...(OutputBiasLayouts)> output_bias_layout{OutputBiasLayouts...};

    template<OutputBiasLayout Layout>
    constexpr auto with_layout() const
    {
        return ConvOutputBiasLayouts<OutputBiasLayouts..., ConvOutputBiasLayout{Layout}>{};
    }
    
    template<ConvOutputLayout2D Layout>
    constexpr auto with_layout() const
    {
        return ConvOutputBiasLayouts<OutputBiasLayouts..., ConvOutputBiasLayout{Layout}>{};
    }
};

template <typename... BiasTensorLayouts>
struct ConvLayout : BiasTensorLayouts...
{
    ConvInputLayout input_layout;
    ConvWeightLayout weight_layout;
    ConvOutputLayout output_layout;

    template<ConvInputLayout2D Layout>
    constexpr auto with_input_layout() const
    {
        auto result = *this;
        result.input_layout = Layout;
        return result;
    }
    
    template<ConvWeightLayout2D Layout>
    constexpr auto with_weight_layout() const
    {
        auto result = *this;
        result.weight_layout = Layout;
        return result;
    }
    
    template<ConvOutputLayout2D Layout>
    constexpr auto with_output_layout() const
    {
        auto result = *this;
        result.output_layout = Layout;
        return result;
    }

    template<typename OutputBiasLayouts>
    constexpr auto with_output_bias_layouts(const OutputBiasLayouts&) const
    {
        return ConvLayout<BiasTensorLayouts..., OutputBiasLayouts>{
            .input_layout = this->input_layout,
            .weight_layout = this->weight_layout,
            .output_layout = this->output_layout
        };
    }
};

struct ElementwiseOperations
{
    ElementwiseOperation input_op{ElementwiseOperation::PASS_THROUGH};
    ElementwiseOperation weight_op{ElementwiseOperation::PASS_THROUGH};
    ElementwiseOperation output_op{ElementwiseOperation::PASS_THROUGH};
};

template <typename GroupConvLayout>
struct ConvSignature
{
    int spatial_dim;
    ConvDirection direction;
    GroupConvLayout layout;
    DataType data_type;
    ElementwiseOperations elementwise_operation;
};

} // namespace ck_tile::builder::test
