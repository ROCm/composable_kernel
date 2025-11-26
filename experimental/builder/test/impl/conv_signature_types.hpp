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
};

template <ConvOutputBiasLayout... OutputBiasLayouts>
struct ConvOutputBiasLayouts
{
    std::array<ConvOutputBiasLayout, sizeof...(OutputBiasLayouts)> output_bias_layout{OutputBiasLayouts...};
};

template <typename... BiasTensorLayouts>
struct ConvLayout : BiasTensorLayouts...
{
    ConvInputLayout input_layout;
    ConvWeightLayout weight_layout;
    ConvOutputLayout output_layout;
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
