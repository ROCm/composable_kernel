// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {
/**********************************************
 * constexpr helper functions for optional parameters
 **********************************************/

template <auto Sig>
concept ProvidesElementwiseOperation = requires { Sig.elementwise_operation; };

template <auto ConvTensor>
concept ConvTensorHasOp = requires { ConvTensor.operation; };

template <auto Sig>
concept ProvidesConvolutionDirection = requires { Sig.direction; };

// returns elementwise operation for signature. Will default to PASS_THROUGH if not provided by
// signature
template <auto Sig>
constexpr auto getInputElementwiseOperation()
{
    if constexpr(ConvTensorHasOp<Sig.input>)
    {
        return Sig.input.operation.elementwise_operation;
    }
    else if constexpr(ProvidesElementwiseOperation<Sig>)
    {
        return Sig.elementwise_operation;
    }
    else
    {
        return ElementwiseOperation::PASS_THROUGH;
    }
}

template <auto Sig>
constexpr auto getWeightElementwiseOperation()
{
    if constexpr(ConvTensorHasOp<Sig.weight>)
    {
        return Sig.weight.operation.elementwise_operation;
    }
    else if constexpr(ProvidesElementwiseOperation<Sig>)
    {
        return Sig.elementwise_operation;
    }
    else
    {
        return ElementwiseOperation::PASS_THROUGH;
    }
}

template <auto Sig>
constexpr auto getOutputElementwiseOperation()
{
    if constexpr(ConvTensorHasOp<Sig.output>)
    {
        return Sig.output.operation.elementwise_operation;
    }
    else if constexpr(ProvidesElementwiseOperation<Sig>)
    {
        return Sig.elementwise_operation;
    }
    else
    {
        return ElementwiseOperation::PASS_THROUGH;
    }
}

// returns convolution direction for signature. Will default to FORWARD if not provided by signature
template <auto Sig>
constexpr auto getConvDirection()
{
    if constexpr(ProvidesConvolutionDirection<Sig>)
    {
        return Sig.direction;
    }
    else
    {
        return ConvDirection::FORWARD;
    }
}

// return data type of input tensor
template <auto Sig>
    requires ck_tile::builder::ValidConvSignature<Sig>
consteval auto getInputDataType()
{
    return GetTensorDataAndComputeTypes<Sig.input.config, Sig.data_type>().get(0);
}

template <auto Sig>
    requires ck_tile::builder::ValidConvSignature<Sig>
consteval auto getWeightDataType()
{
    return GetTensorDataAndComputeTypes<Sig.weight.config, Sig.data_type>().get(0);
}

template <auto Sig>
    requires ck_tile::builder::ValidConvSignature<Sig>
consteval auto getOutputDataType()
{
    return GetTensorDataAndComputeTypes<Sig.output.config, Sig.data_type>().get(0);
}

// returns data type if and only if all tensors have the same type.
// Otherwise, return DataType::UNDEFINED_DATA_TYPE
template <auto Sig>
    requires ck_tile::builder::ValidConvSignature<Sig>
consteval auto getDataTypeIfCommon()
{

    auto inputDataType  = getInputDataType<Sig>();
    auto weightDataType = getWeightDataType<Sig>();
    auto outputDataType = getOutputDataType<Sig>();

    if(inputDataType == weightDataType == outputDataType)
    {
        return inputDataType;
    }
    else
    {
        return DataType::UNDEFINED_DATA_TYPE;
    }
}
} // namespace ck_tile::builder
