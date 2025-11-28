// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// This file defines the compile-time "signature" for grouped convolution operations.
// A signature is a collection of properties that fully describe a convolution kernel's
// mathematical characteristics. It uses C++20 concepts and enums to specify these
// properties, enabling compile-time validation and specialization.
//
// The core components of a signature are:
//   - Spatial dimensionality (1D, 2D, 3D)
//   - Operational direction (Forward, Backward Data, Backward Weight)
//   - Tensor memory layout (Channels First/Last)
//   - Data type (FP32, FP16, BF16)
//   - Fused element-wise operation (e.g., Bias, Clamp)
//
// The file also provides predicate concepts to query the properties of a given
// signature at compile time.
#pragma once

#include <concepts>
#include <type_traits>

#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {

// Constrains convolution to 1D, 2D, or 3D spatial dimensions.
template <auto N>
concept ConvSpatialDim = std::is_integral_v<decltype(N)> && (N == 1 || N == 2 || N == 3);

// Constrains convolution data types to common floating-point types.
template <DataType T>
concept ValidConvDataType = (T == DataType::FP32) || (T == DataType::FP16) || (T == DataType::BF16) ||
                       (T == DataType::FP8) || (T == DataType::I8) || (T == DataType::U8);

template <typename T>
concept TensorConfigDescriptor = requires(T t) {
    { t.layout } -> std::convertible_to<ConvLayout>;
    // Only require that data type is defined. It might be set to undefined value, in which case the signature's data type is used.
    { t.data_type } -> std::convertible_to<DataType>;
};

template <typename T>
concept HasAuxiliaryOperandConfigs = requires(T t) {
    { t.auxiliary_operand_configs };
};

template <typename T>
concept ConvertibleToArrayOfTensorConfigs = 
    std::is_same_v<std::remove_cvref_t<T>, std::array<TensorConfig, std::tuple_size_v<std::remove_cvref_t<T>>>>;

template <typename T>
concept AuxiliaryOperandConfigsWellDefinedIfProvided = requires(T t) {
    requires !HasAuxiliaryOperandConfigs<T> || requires {
        { t.auxiliary_operand_configs } -> ConvertibleToArrayOfTensorConfigs;
    };
};

template <typename T>
concept TensorOperatorDescriptor = requires(T t) {
    { t.elementwise_operation } -> std::convertible_to<ElementwiseOperation>;
    requires AuxiliaryOperandConfigsWellDefinedIfProvided<T>;
};

template <typename T>
concept HasTensorOp = requires(T t) {
    { t.operation };
};

template <typename T>
concept HasConvolutionDirection = requires(T t) {
    { t.direction };
};

// Note: it is not required to provide an ElementwiseOp, but if one is provided, check if well
// defined
template <typename T>
concept ElementwiseOpWellDefinedIfProvided = requires { !HasTensorOp<T> || TensorOperatorDescriptor<T>;};

// Note: it is not required to provide a convolution, but if one is provided, check if well defined
template <typename T>
concept ConvolutionDirectionWellDefinedIfProvided = requires(T t) {
    requires !HasConvolutionDirection<T> || requires {
        { t.direction } -> std::convertible_to<ConvDirection>;
    };
};

// Concept for the convolution tensor
template <typename T>
concept ConvTensorDescriptor = requires(T t) {
    { t.type } -> std::convertible_to<ConvolutionTensorType>;
    { t.config } -> TensorConfigDescriptor;
    requires ElementwiseOpWellDefinedIfProvided<T>;
};

template <typename T>
concept HasElementwiseOpWithAuxiliaryOperands = requires(T t) {
    requires HasTensorOp<T>;
    requires HasAuxiliaryOperandConfigs<decltype(t.operation)>;
};

// Concept for a type that defines a convolution's operational signature.
template <typename T>
concept ConvSignatureDescriptor = requires(T t) {
    { t.spatial_dim } -> std::convertible_to<unsigned int>;
    { t.data_type } -> std::convertible_to<DataType>;
    { t.input } -> ConvTensorDescriptor;
    { t.weight } -> ConvTensorDescriptor;
    { t.output } -> ConvTensorDescriptor;
    requires ConvolutionDirectionWellDefinedIfProvided<T>;
};

// Concept to validate a convolution signature's values.
template <auto Sig>
concept ValidConvSignature = requires {
    requires ConvSpatialDim<Sig.spatial_dim>;
    requires ValidConvDataType<Sig.data_type>;
};

// Predicate for forward convolution (default if direction is not included).
template <auto Sig>
concept ConvDirectionIsForward =
    !requires { Sig.direction; } || (Sig.direction == ConvDirection::FORWARD);

// Predicate for backward data convolution.
template <auto Sig>
concept ConvDirectionIsBackwardData = (Sig.direction == ConvDirection::BACKWARD_DATA);

// Predicate for backward weight convolution.
template <auto Sig>
concept ConvDirectionIsBackwardWeight = (Sig.direction == ConvDirection::BACKWARD_WEIGHT);

// Constraints for forward convolution input layouts.
template <auto LayoutValue, size_t SpatialDim>
concept ValidConvInputLayoutForSpatialDim =
    (SpatialDim == 1 && std::same_as<decltype(LayoutValue._1d), ConvInputLayout1D>) ||
    (SpatialDim == 2 && std::same_as<decltype(LayoutValue._2d), ConvInputLayout2D>) ||
    (SpatialDim == 3 && std::same_as<decltype(LayoutValue._3d), ConvInputLayout3D>);

// Constraints for forward convolution output layouts.
template <auto LayoutValue, size_t SpatialDim>
concept ValidConvOutputLayoutForSpatialDim =
    (SpatialDim == 1 && std::same_as<decltype(LayoutValue._1d), ConvOutputLayout1D>) ||
    (SpatialDim == 2 && std::same_as<decltype(LayoutValue._2d), ConvOutputLayout2D>) ||
    (SpatialDim == 3 && std::same_as<decltype(LayoutValue._3d), ConvOutputLayout3D>);

// Constraints for forward convolution weight layouts.
template <auto LayoutValue, size_t SpatialDim>
concept ValidConvWeightLayoutForSpatialDim =
    (SpatialDim == 1 && std::same_as<decltype(LayoutValue._1d), ConvWeightLayout1D>) ||
    (SpatialDim == 2 && std::same_as<decltype(LayoutValue._2d), ConvWeightLayout2D>) ||
    (SpatialDim == 3 && std::same_as<decltype(LayoutValue._3d), ConvWeightLayout3D>);

} // namespace ck_tile::builder
