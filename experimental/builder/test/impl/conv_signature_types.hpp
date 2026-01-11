// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <variant>
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::test {

using namespace ck_tile::builder;

struct TensorConfig
{
    TensorLayout layout;
    // Optional data types, override the type defined in the signature if provided.
    DataType data_type{DataType::UNDEFINED_DATA_TYPE};
    DataType compute_type{DataType::UNDEFINED_DATA_TYPE};

    constexpr bool operator==(const TensorConfig& other) const
    {
        return layout == other.layout && data_type == other.data_type &&
               compute_type == other.compute_type;
    }
};

template <TensorConfig... Configs>
struct TensorOperation
{
    ElementwiseOperation elementwise_operation{ElementwiseOperation::PASS_THROUGH};
    std::array<TensorConfig, sizeof...(Configs)> auxiliary_operand_configs{Configs...};

    // Add builder to add auxiliary tensor configs
    template <auto... AuxiliaryConfigs>
    constexpr auto with_auxiliary_operand_configs() const
    {
        return TensorOperation<Configs..., TensorConfig{AuxiliaryConfigs}...>{
            .elementwise_operation = this->elementwise_operation};
    }

    constexpr bool operator==(const TensorOperation& other) const
    {
        return elementwise_operation == other.elementwise_operation &&
               auxiliary_operand_configs == other.auxiliary_operand_configs;
    }
};

template <typename Op = TensorOperation<>>
struct ConvolutionTensor
{
    TensorConfig config;
    Op operation{};

    constexpr bool operator==(const ConvolutionTensor& other) const
    {
        return config == other.config && operation == other.operation;
    }
};

template <typename InputTensor  = ConvolutionTensor<>,
          typename WeightTensor = ConvolutionTensor<>,
          typename OutputTensor = ConvolutionTensor<>>
struct ConvSignature
{
    int spatial_dim;
    ConvDirection direction;
    DataType data_type;
    DataType accumulation_data_type;
    InputTensor input;
    WeightTensor weight;
    OutputTensor output;

    constexpr bool operator==(const ConvSignature& other) const
    {
        return spatial_dim == other.spatial_dim && direction == other.direction &&
               data_type == other.data_type &&
               accumulation_data_type == other.accumulation_data_type && input == other.input &&
               weight == other.weight && output == other.output;
    }
};

} // namespace ck_tile::builder::test
