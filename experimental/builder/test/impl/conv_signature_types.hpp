// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <variant>
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::test {

using namespace ck_tile::builder;

template <TensorConfig... Configs>
struct TensorOperation
{
    ElementwiseOperation elementwise_operation{ElementwiseOperation::PASS_THROUGH};
    std::array<TensorConfig, sizeof...(Configs)> auxiliary_operand_configs{Configs...};
};

template <typename Op>
struct ConvolutionTensor
{
    ConvolutionTensorType type;
    TensorConfig config;
    Op operation;
};

template <typename InputTensor, typename WeightTensor, typename OutputTensor>
struct ConvSignature
{
    int spatial_dim;
    ConvDirection direction;
    DataType data_type;
    DataType accumulation_data_type;
    InputTensor input;
    WeightTensor weight;
    OutputTensor output;
};

} // namespace ck_tile::builder::test
