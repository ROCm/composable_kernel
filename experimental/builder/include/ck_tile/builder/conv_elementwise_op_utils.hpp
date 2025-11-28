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
        constexpr auto op = TensorDesc.operation.elementwise_operation;
        if constexpr (op == ElementwiseOperation::SCALE)
        {
            return CK_ScaleOp{};
        }
        else if constexpr (op == ElementwiseOperation::SCALEADD_SCALEADD_RELU)
        {
            return CK_ScaleAddScaleAddReluOp{};
        }
        else if constexpr (op == ElementwiseOperation::BIAS_BNORM_CLAMP)
        {
            return CK_BiasNormalizeInInferClampOp{};
        }
        else if constexpr (op == ElementwiseOperation::CLAMP)
        {
            return CK_ClampOp{};
        }
        else if constexpr (op == ElementwiseOperation::PASS_THROUGH)
        {
            return CK_PassThroughOp{};
        }
        else 
        {
            static_assert(sizeof(UnsupportedEnumValue<op>) == 0, "Unsupported elementwise operation!");
        }
    }
    else
    {
        return CK_PassThroughOp{};
    }
}

template <auto InputTensor, auto WeightTensor, auto OutputTensor>
struct ElementwiseOps
{
    static constexpr auto input_op = GetElementwiseOp<InputTensor>();
    static constexpr auto weight_op = GetElementwiseOp<WeightTensor>();
    static constexpr auto output_op = GetElementwiseOp<OutputTensor>();
    using AElementwiseOp   = typename decltype(input_op)::Op;
    using BElementwiseOp   = typename decltype(weight_op)::Op;
    using CDEElementwiseOp = typename decltype(output_op)::Op;
};

template <auto Sig>
constexpr auto GetElementwiseOps()
{
    return ElementwiseOps<Sig.input, Sig.weight, Sig.output>{};
}

}
