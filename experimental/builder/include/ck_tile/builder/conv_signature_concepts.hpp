// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
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
#include "ck_tile/builder/conv_signature_predicates.hpp"

namespace ck_tile::builder {

// Constrains convolution to 1D, 2D, or 3D spatial dimensions.
template <auto N>
concept ConvSpatialDim = std::is_integral_v<decltype(N)> && (N == 1 || N == 2 || N == 3);

// Constraints for forward convolution layouts.
template <auto LayoutValue, size_t SpatialDim>
concept ValidConvLayoutForSpatialDim =
    (SpatialDim == 1 && std::same_as<decltype(LayoutValue), GroupConvLayout1D>) ||
    (SpatialDim == 2 && std::same_as<decltype(LayoutValue), GroupConvLayout2D>) ||
    (SpatialDim == 3 && std::same_as<decltype(LayoutValue), GroupConvLayout3D>);

// Constrains convolution data types to common floating-point types.
template <DataType T>
concept ConvDataType = (T == DataType::FP32) || (T == DataType::FP16) || (T == DataType::BF16) ||
                       (T == DataType::FP8) || (T == DataType::I8) || (T == DataType::U8);

template <typename T>
concept ConvDeviceOp = std::same_as<std::remove_cvref_t<T>, GroupConvDeviceOp>;

template <typename T>
concept ConvLayout = std::same_as<std::remove_cvref_t<T>, GroupConvLayout>;

// Concept for a type that defines a convolution's operational signature.
template <typename T>
concept ConvSignatureDescriptor = requires(T t) {
    { t.spatial_dim } -> std::convertible_to<unsigned int>;
    { t.direction } -> std::convertible_to<ConvDirection>;
    { t.layout } -> ConvLayout;
    { t.data_type } -> std::convertible_to<DataType>;
    { t.elementwise_operation } -> std::convertible_to<ElementwiseOperation>;
    { t.device_operation } -> ConvDeviceOp;
};

// Concept to validate a convolution signature's values.
template <auto Sig>
concept ValidConvSignature = requires {
    requires ConvSpatialDim<Sig.spatial_dim>;
    requires ConvDataType<Sig.data_type>;
    requires IsValidConvDeviceOp<Sig>;
};

} // namespace ck_tile::builder
