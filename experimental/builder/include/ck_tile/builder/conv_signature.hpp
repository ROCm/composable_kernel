#pragma once

#include <concepts>
#include <type_traits>

#include <ck_tile/builder/types.hpp>

namespace ck_tile::builder {

// Layouts for grouped convolutions.
enum class GroupConvLayout
{
    NHWGC_GKYXC_NHWGK,    // Channels-last
    NDHWGC_GKZYXC_NDHWGK, // Channels-last
    NGCHW_GKCYX_NGKHW     // Channels-first
};

// Spatial dimensionalities of grouped convolutions.
// N represents the number of spatial dimensions (e.g., 1 for 1D, 2 for 2D, 3 for 3D).
template <auto N>
concept ConvSpatialDim = std::is_integral_v<decltype(N)> && (N == 1 || N == 2 || N == 3);

// Allowed datatypes for grouped convolutions.
// Currently limited to floating-point types commonly accelerated on GPUs.
template <DataType T>
concept ConvDataType = (T == DataType::FP32) || (T == DataType::FP16) || (T == DataType::BF16);

// Direction of the convolution operation.
enum class ConvDirection
{
    Forward,
    BackwardData,
    BackwardWeight
};

// Elementwise operation to fuse to convolution.
enum class ElementwiseOperation
{
    Bias,
    BiasClamp,
    Bilinear,
    Clamp,
    Scale,
    PassThrough
};

// Operational signature of a convolution.
template <typename T>
concept ConvSignatureDescriptor = requires(T t) {
    { t.spatial_dim } -> std::convertible_to<int>;
    { t.direction } -> std::convertible_to<ConvDirection>;
    { t.layout } -> std::convertible_to<GroupConvLayout>;
    { t.data_type } -> std::convertible_to<DataType>;
};

// Valid values for a convolution signature.
template <auto Sig>
concept ValidConvSignature = requires {
    requires ConvSpatialDim<Sig.spatial_dim>;
    requires ConvDataType<Sig.data_type>;
};

} // namespace ck_tile::builder
