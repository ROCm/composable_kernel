// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory_internal
{
struct CK_PassThroughOp
{
    using Op = ck::tensor_operation::element_wise::PassThrough;
};

struct CK_ScaleOp
{
    using Op = ck::tensor_operation::element_wise::Scale;
};

struct CK_ClampOp
{
    using Op = ck::tensor_operation::element_wise::Clamp;
};

struct CK_ScaleAddScaleAddReluOp
{
    using Op = ck::tensor_operation::element_wise::ScaleAddScaleAddRelu;
};

struct CK_BiasNormalizeInInferClampOp
{
    using Op = ck::tensor_operation::element_wise::BiasNormalizeInInferClamp;
};


template <auto TensorDesc>
consteval auto GetElementwiseOp()
{
    if constexpr (HasTensorOp<decltype(TensorDesc)>)
    {
        if constexpr (TensorDesc.operation.elementwise_operation == ElementwiseOperation::SCALE)
        {
            return CK_ScaleOp{};
        }
        else if constexpr (TensorDesc.operation.elementwise_operation == ElementwiseOperation::SCALEADD_SCALEADD_RELU)
        {
            return CK_ScaleAddScaleAddReluOp{};
        }
        else if constexpr (TensorDesc.operation.elementwise_operation == ElementwiseOperation::BIAS_BNORM_CLAMP)
        {
            return CK_BiasNormalizeInInferClampOp{};
        }
        else if constexpr (TensorDesc.operation.elementwise_operation == ElementwiseOperation::CLAMP)
        {
            return CK_ClampOp{};
        }
        else if constexpr (TensorDesc.operation.elementwise_operation == ElementwiseOperation::PASS_THROUGH)
        {
            return CK_PassThroughOp{};
        }
        else 
        {
            static_assert(false, "Unsupported elementwise operation!");
        }
    }
    return CK_PassThroughOp{};
}

template <auto InputTensor, auto WeightTensor, auto OutputTensor>
struct ElementwiseOps
{
    static const auto input_op = GetElementwiseOp<InputTensor>();
    static const auto weight_op = GetElementwiseOp<WeightTensor>();
    static const auto output_op = GetElementwiseOp<OutputTensor>();
    using AElementwiseOp   = typename decltype(input_op)::Op;
    using BElementwiseOp   = typename decltype(weight_op)::Op;
    using CDEElementwiseOp = typename decltype(output_op)::Op;
};

template <auto Sig>
constexpr auto GetElementwiseOps()
{
    return ElementwiseOps<Sig.elementwise_operation.input_op, Sig.elementwise_operation.weight_op, Sig.elementwise_operation.output_op>{};
}

}
