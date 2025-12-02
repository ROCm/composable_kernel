// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory_internal {

template <ElementwiseOperation Op>
struct ElementwiseOpToCK
{
    static_assert(sizeof(UnsupportedEnumValue<Op>) == 0,
                  "Unsupported elementwise operation conversion to CK.");
};

template <>
struct ElementwiseOpToCK<ElementwiseOperation::PASS_THROUGH>
{
    using Op = ck::tensor_operation::element_wise::PassThrough;
};

template <>
struct ElementwiseOpToCK<ElementwiseOperation::SCALE>
{
    using Op = ck::tensor_operation::element_wise::Scale;
};

template <>
struct ElementwiseOpToCK<ElementwiseOperation::CLAMP>
{
    using Op = ck::tensor_operation::element_wise::Clamp;
};

template <>
struct ElementwiseOpToCK<ElementwiseOperation::SCALEADD_SCALEADD_RELU>
{
    using Op = ck::tensor_operation::element_wise::ScaleAddScaleAddRelu;
};

template <>
struct ElementwiseOpToCK<ElementwiseOperation::BIAS_BNORM_CLAMP>
{
    using Op = ck::tensor_operation::element_wise::BiasNormalizeInInferClamp;
};

template <auto TensorDesc>
consteval auto GetElementwiseOp()
{
    if constexpr(HasTensorOp<decltype(TensorDesc)>)
    {
        constexpr auto op = TensorDesc.operation.elementwise_operation;
        return ElementwiseOpToCK<op>{};
    }
    else
    {
        return ElementwiseOpToCK<ElementwiseOperation::PASS_THROUGH>{};
    }
}

template <auto InputTensor, auto WeightTensor, auto OutputTensor>
struct ElementwiseOps
{
    static constexpr auto input_op  = GetElementwiseOp<InputTensor>();
    static constexpr auto weight_op = GetElementwiseOp<WeightTensor>();
    static constexpr auto output_op = GetElementwiseOp<OutputTensor>();
    using AElementwiseOp            = typename decltype(input_op)::Op;
    using BElementwiseOp            = typename decltype(weight_op)::Op;
    using CDEElementwiseOp          = typename decltype(output_op)::Op;
};

template <auto Sig>
constexpr auto GetElementwiseOps()
{
    return ElementwiseOps<Sig.input, Sig.weight, Sig.output>{};
}

} // namespace ck_tile::builder::factory_internal
