// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <variant>
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::test {

using namespace ck_tile::builder;

template <auto... BiasLayouts>
struct ConvLayout
{
    ConvInputLayout input_layout;
    ConvWeightLayout weight_layout;
    ConvOutputLayout output_layout;
    std::array<ConvBiasLayout, sizeof...(BiasLayouts)> bias_layout{ConvBiasLayout{BiasLayouts}...};

    template<auto... Layouts>
    constexpr auto with_bias_layout() const
    {
        return ConvLayout<BiasLayouts..., ConvBiasLayout{Layouts}...>{
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
