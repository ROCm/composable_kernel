// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory::internal {
using ALayout = ck_tile::tensor_layout::convolution::NWGC;
template <TensorLayout LAYOUT>
struct LayoutToCKTile
{
    static_assert(sizeof(UnsupportedEnumValue<LAYOUT>) == 0,
                  "Unsupported layout conversion to CK.");
};

// Bias layouts
template <>
struct LayoutToCKTile<TensorLayout::G_K_strided>
{
    using type = ck_tile::tensor_layout::convolution::G_K;
};
template <>
struct LayoutToCKTile<TensorLayout::GC>
{
    using type = ck_tile::tensor_layout::convolution::GC;
};
template <>
struct LayoutToCKTile<TensorLayout::G_C_strided>
{
    using type = ck_tile::tensor_layout::convolution::G_C;
};

// Input 1D
template <>
struct LayoutToCKTile<TensorLayout::NWGC>
{
    using type = ck_tile::tensor_layout::convolution::NWGC;
};
template <>
struct LayoutToCKTile<TensorLayout::GNWC>
{
    using type = ck_tile::tensor_layout::convolution::GNWC;
};

// Input 2D
template <>
struct LayoutToCKTile<TensorLayout::NHWGC>
{
    using type = ck_tile::tensor_layout::convolution::NHWGC;
};
template <>
struct LayoutToCKTile<TensorLayout::GNHWC>
{
    using type = ck_tile::tensor_layout::convolution::GNHWC;
};

// Input 3D
template <>
struct LayoutToCKTile<TensorLayout::NDHWGC>
{
    using type = ck_tile::tensor_layout::convolution::NDHWGC;
};
template <>
struct LayoutToCKTile<TensorLayout::GNDHWC>
{
    using type = ck_tile::tensor_layout::convolution::GNDHWC;
};

// Weight 1D
template <>
struct LayoutToCKTile<TensorLayout::GKXC>
{
    using type = ck_tile::tensor_layout::convolution::GKXC;
};
template <>
struct LayoutToCKTile<TensorLayout::GKCX>
{
    using type = ck_tile::tensor_layout::convolution::GKCX;
};

// Weight 2D
template <>
struct LayoutToCKTile<TensorLayout::GKYXC>
{
    using type = ck_tile::tensor_layout::convolution::GKYXC;
};
template <>
struct LayoutToCKTile<TensorLayout::GKCYX>
{
    using type = ck_tile::tensor_layout::convolution::GKCYX;
};

// Weight 3D
template <>
struct LayoutToCKTile<TensorLayout::GKCZYX>
{
    using type = ck_tile::tensor_layout::convolution::GKCZYX;
};
template <>
struct LayoutToCKTile<TensorLayout::GKZYXC>
{
    using type = ck_tile::tensor_layout::convolution::GKZYXC;
};

// Output 1D
template <>
struct LayoutToCKTile<TensorLayout::NWGK>
{
    using type = ck_tile::tensor_layout::convolution::NWGK;
};
template <>
struct LayoutToCKTile<TensorLayout::GNWK>
{
    using type = ck_tile::tensor_layout::convolution::GNWK;
};

// Output 2D
template <>
struct LayoutToCKTile<TensorLayout::NHWGK>
{
    using type = ck_tile::tensor_layout::convolution::NHWGK;
};
template <>
struct LayoutToCKTile<TensorLayout::GNHWK>
{
    using type = ck_tile::tensor_layout::convolution::GNHWK;
};

// Output 3D
template <>
struct LayoutToCKTile<TensorLayout::NDHWGK>
{
    using type = ck_tile::tensor_layout::convolution::NDHWGK;
};
template <>
struct LayoutToCKTile<TensorLayout::GNDHWK>
{
    using type = ck_tile::tensor_layout::convolution::GNDHWK;
};

template <TensorLayout Layout>
consteval auto TensorLayoutToCKTile()
{
    return typename LayoutToCKTile<Layout>::type{};
}

struct EmptyAuxiliaryTileTensorLayout
{
    using type = ck_tile::tuple<>;
};

template <auto AUXILIARY_TILE_TENSOR_CONFIGS_ARRAY, size_t... Indices>
consteval auto GetAuxiliaryTileTensorLayoutTuple(std::index_sequence<Indices...>)
{
    return ck_tile::tuple<
        decltype(TensorLayoutToCKTile<AUXILIARY_TILE_TENSOR_CONFIGS_ARRAY[Indices].layout>())...>{};
}

template <auto AUXILIARY_TILE_TENSOR_CONFIGS_VALUE, size_t SPATIAL_DIM>
    requires ConvSpatialDim<SPATIAL_DIM>
struct AuxiliaryTileTensorLayouts
{
    static constexpr auto Size = AUXILIARY_TILE_TENSOR_CONFIGS_VALUE.size();
    using type = decltype(GetAuxiliaryTileTensorLayoutTuple<AUXILIARY_TILE_TENSOR_CONFIGS_VALUE>(
        std::make_index_sequence<Size>{}));
};

// TODO: Currently only the ouput tensor can have auxiliary tensors (e.g., bias).
template <auto SIGNATURE>
    requires HasElementwiseOpWithAuxiliaryOperands<decltype(SIGNATURE.output)>
consteval auto GetAuxiliaryTileTensorLayouts()
{
    return AuxiliaryTileTensorLayouts<SIGNATURE.output.operation.auxiliary_operand_configs,
                                      SIGNATURE.spatial_dim>{};
}

template <auto SIGNATURE>
    requires(!HasElementwiseOpWithAuxiliaryOperands<decltype(SIGNATURE.output)>)
consteval auto GetAuxiliaryTileTensorLayouts()
{
    return EmptyAuxiliaryTileTensorLayout{};
}

template <auto SIGNATURE>
    requires ConvSpatialDim<SIGNATURE.spatial_dim> &&
             ValidConvInputLayoutForSpatialDim<SIGNATURE.input.config.layout,
                                               SIGNATURE.spatial_dim> &&
             ValidConvWeightLayoutForSpatialDim<SIGNATURE.weight.config.layout,
                                                SIGNATURE.spatial_dim> &&
             ValidConvOutputLayoutForSpatialDim<SIGNATURE.output.config.layout,
                                                SIGNATURE.spatial_dim>
struct TileConvTensorLayouts
{
    using ALayout  = decltype(TensorLayoutToCKTile<SIGNATURE.input.config.layout>());
    using BLayout  = decltype(TensorLayoutToCKTile<SIGNATURE.weight.config.layout>());
    using ELayout  = decltype(TensorLayoutToCKTile<SIGNATURE.output.config.layout>());
    using DsLayout = decltype(GetAuxiliaryTileTensorLayouts<SIGNATURE>())::type;
};

} // namespace ck_tile::builder::factory::internal
