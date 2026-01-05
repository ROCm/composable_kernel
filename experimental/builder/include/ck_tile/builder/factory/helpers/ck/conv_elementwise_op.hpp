// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory::internal {

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

template <auto Sig>
struct ElementwiseOps
{
    private:
    static constexpr auto input_op  = GetElementwiseOp<Sig.input>();
    static constexpr auto weight_op = GetElementwiseOp<Sig.weight>();
    static constexpr auto output_op = GetElementwiseOp<Sig.output>();

    static constexpr bool is_forward    = ConvDirectionIsForward<Sig>;
    static constexpr bool is_bwd_weight = ConvDirectionIsBackwardWeight<Sig>;

    using InputOp  = typename decltype(input_op)::Op;
    using WeightOp = typename decltype(weight_op)::Op;
    using OutputOp = typename decltype(output_op)::Op;

    public:
    // Forward convolution elementwise ops
    using AElementwiseOp   = std::conditional_t<is_forward, InputOp, void>;
    using BElementwiseOp   = std::conditional_t<is_forward, WeightOp, void>;
    using CDEElementwiseOp = std::conditional_t<is_forward, OutputOp, void>;

    // Backward weight convolution elementwise ops
    using InElementwiseOp  = std::conditional_t<is_bwd_weight, InputOp, void>;
    using WeiElementwiseOp = std::conditional_t<is_bwd_weight, WeightOp, void>;
    using OutElementwiseOp = std::conditional_t<is_bwd_weight, OutputOp, void>;
};

} // namespace ck_tile::builder::factory::internal
