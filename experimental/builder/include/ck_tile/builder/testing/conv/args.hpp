// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <optional>
#include <stdexcept>

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include "ck_tile/builder/testing/testing.hpp"
#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/testing/validation.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

/// This file implements common functionality for invoking/testing grouped
/// forward convolutions created through the CK Builder API. The main item
/// of it is the Args structure - which contains a complete description
/// of a convolution operation.
///
/// It is not intended that this file contains implementation details for
/// actually launching a convolution operation. As this can be done
/// through different APIs depending on the kernel (CK, CK Tile, or a
/// reference implementation), the code dealing with that is split out
/// into a separate header for each implementation. Nor does this file
/// deal with details for defining the data types (`Inputs` and `Outputs`)
/// for different conv directions, that is also split out into separate
/// headers to keep this one small.

namespace ck_tile::builder::test {

/// @brief Convolution tensor dimensions.
///
/// This structure is used to describe lengths of a convolution problem. In
/// fact, this structure is a complete description of ALL inputs and outputs
/// lengths of a convolution problem, as this structure contains all of the
/// combined parameters. Note that we can't also use this structure to describe
/// tensor strides: whereas the lengths are all governed by a common set of
/// parameters, strides of the input, weight, and output tensor are all
/// independent.
template <int SPATIAL_DIM>
struct ConvTensorLengths
{
    size_t batch_size                = 1;  // N
    size_t groups                    = 1;  // G
    size_t input_channels            = 1;  // C
    size_t output_channels           = 1;  // K
    FilterExtent<SPATIAL_DIM> image  = {}; // W, H, D
    FilterExtent<SPATIAL_DIM> filter = {}; // X, Y, Z
};

namespace detail {

/// @brief Calculate memory strides for a tensor with custom dimension ordering.
///
/// Given tensor dimensions and a memory layout order, compute the stride
/// (memory jump size) needed to move by 1 in each dimension.
///
/// @param lengths Tensor dimensions (e.g., {3, 4} for 3 rows × 4 columns)
/// @param outer_to_inner Dimension ordering from outermost to innermost in memory
/// @return Strides for each dimension (e.g., {4, 1} for row-major 3×4 tensor)
///
/// Example: For a 3×4 tensor stored row-major (outer_to_inner = {0, 1}):
///   - Moving 1 row down requires jumping 4 positions → stride[0] = 4
///   - Moving 1 column right requires jumping 1 position → stride[1] = 1
template <size_t RANK>
Extent<RANK> make_packed_strides_for_order(const Extent<RANK>& lengths,
                                           const std::array<size_t, RANK>& outer_to_inner)
{
    Extent<RANK> strides = {};

    size_t stride = 1; // Innermost dimension always has stride 1
    for(size_t i = RANK; i > 0; --i)
    {
        const auto dim = outer_to_inner[i - 1]; // Get dimension at this position
        strides[dim]   = stride;                // Assign current stride
        stride *= lengths[dim];                 // Update stride for next (outer) dimension
    }

    return strides;
}

template <int SPATIAL_DIM>
std::array<long_index_t, SPATIAL_DIM>
compute_output_spatial(const std::array<long_index_t, SPATIAL_DIM>& input_spatial,
                       const std::array<long_index_t, SPATIAL_DIM>& filter_spatial,
                       const std::array<long_index_t, SPATIAL_DIM>& conv_strides,
                       const std::array<long_index_t, SPATIAL_DIM>& conv_dilations,
                       const std::array<long_index_t, SPATIAL_DIM>& left_pads,
                       const std::array<long_index_t, SPATIAL_DIM>& right_pads)
{
    std::array<long_index_t, SPATIAL_DIM> output_spatial = {};

    for(int i = 0; i < SPATIAL_DIM; ++i)
    {
        const auto in  = input_spatial[i];
        const auto fil = filter_spatial[i];
        const auto s   = conv_strides[i];
        const auto d   = conv_dilations[i];
        const auto pl  = left_pads[i];
        const auto pr  = right_pads[i];

        // effective_filter = dilation*(filter-1) + 1
        const auto effective_filter = d * (fil - 1) + 1;
        const auto numerator        = in + pl + pr - effective_filter;

        if(s <= 0)
        {
            throw std::runtime_error("invalid convolution stride (must be > 0)");
        }
        if(numerator < 0)
        {
            throw std::runtime_error("invalid convolution parameters (negative output spatial)");
        }

        output_spatial[i] = numerator / s + 1;
    }

    return output_spatial;
}

/// @brief Get memory layout order for any convolution tensor based on TensorLayout.
///
/// This function maps TensorLayout enums to their corresponding memory dimension ordering.
/// Layouts with the same memory pattern (e.g., GNCW, GKCX, GNKW) share the same order array,
/// demonstrating that memory layout order depends on dimension positions, not semantic meaning.
///
/// @tparam LAYOUT The tensor layout (input, weight, or output)
/// @tparam SPATIAL_DIM The spatial dimensionality (1, 2, or 3)
/// @returns Array indicating dimension order from outermost to innermost in memory
template <TensorLayout LAYOUT, int SPATIAL_DIM>
consteval auto get_layout_order()
{
    if constexpr(SPATIAL_DIM == 1)
    {
        // 1D layouts: Order {0, 1, 2, 3} - G, N/K, C/K, W/X pattern
        if constexpr(LAYOUT == TensorLayout::GNCW || LAYOUT == TensorLayout::GKCX ||
                     LAYOUT == TensorLayout::GNKW)
            return std::array<size_t, 4>{0, 1, 2, 3};
        // 1D layouts: Order {0, 1, 3, 2} - G, N/K, W/X, C/K pattern (channels-last)
        else if constexpr(LAYOUT == TensorLayout::GNWC || LAYOUT == TensorLayout::G_NW_C_strided ||
                          LAYOUT == TensorLayout::GKXC || LAYOUT == TensorLayout::G_K_X_C_strided ||
                          LAYOUT == TensorLayout::GNWK || LAYOUT == TensorLayout::G_NW_K_strided)
            return std::array<size_t, 4>{0, 1, 3, 2};
        // 1D layouts: Order {1, 3, 0, 2} - N/K, W/X, G, C/K pattern (batch-first)
        else if constexpr(LAYOUT == TensorLayout::NWGC || LAYOUT == TensorLayout::KXGC ||
                          LAYOUT == TensorLayout::NWGK)
            return std::array<size_t, 4>{1, 3, 0, 2};
        // 1D layouts: Order {1, 0, 2, 3} - N/K, G, C/K, W/X pattern
        else if constexpr(LAYOUT == TensorLayout::NGCW || LAYOUT == TensorLayout::NGKW)
            return std::array<size_t, 4>{1, 0, 2, 3};
        else
            static_assert(sizeof(UnsupportedEnumValue<LAYOUT>) == 0, "Unsupported 1D layout");
    }
    else if constexpr(SPATIAL_DIM == 2)
    {
        // 2D layouts: Order {0, 1, 2, 3, 4} - G, N/K, C/K, H/Y, W/X pattern
        if constexpr(LAYOUT == TensorLayout::GNCHW || LAYOUT == TensorLayout::GKCYX ||
                     LAYOUT == TensorLayout::GNKHW)
            return std::array<size_t, 5>{0, 1, 2, 3, 4};
        // 2D layouts: Order {0, 1, 3, 4, 2} - G, N/K, H/Y, W/X, C/K pattern (channels-last)
        else if constexpr(LAYOUT == TensorLayout::GNHWC ||
                          LAYOUT == TensorLayout::G_NHW_C_strided ||
                          LAYOUT == TensorLayout::GKYXC ||
                          LAYOUT == TensorLayout::G_K_YX_C_strided ||
                          LAYOUT == TensorLayout::GNHWK || LAYOUT == TensorLayout::G_NHW_K_strided)
            return std::array<size_t, 5>{0, 1, 3, 4, 2};
        // 2D layouts: Order {1, 3, 4, 0, 2} - N/K, H/Y, W/X, G, C/K pattern (batch-first)
        else if constexpr(LAYOUT == TensorLayout::NHWGC || LAYOUT == TensorLayout::KYXGC ||
                          LAYOUT == TensorLayout::NHWGK)
            return std::array<size_t, 5>{1, 3, 4, 0, 2};
        // 2D layouts: Order {1, 0, 2, 3, 4} - N/K, G, C/K, H/Y, W/X pattern
        else if constexpr(LAYOUT == TensorLayout::NGCHW || LAYOUT == TensorLayout::NGKHW)
            return std::array<size_t, 5>{1, 0, 2, 3, 4};
        else
            static_assert(sizeof(UnsupportedEnumValue<LAYOUT>) == 0, "Unsupported 2D layout");
    }
    else
    {
        // 3D layouts: Order {0, 1, 2, 3, 4, 5} - G, N/K, C/K, D/Z, H/Y, W/X pattern
        if constexpr(LAYOUT == TensorLayout::GNCDHW || LAYOUT == TensorLayout::GKCZYX ||
                     LAYOUT == TensorLayout::GNKDHW)
            return std::array<size_t, 6>{0, 1, 2, 3, 4, 5};
        // 3D layouts: Order {0, 1, 3, 4, 5, 2} - G, N/K, D/Z, H/Y, W/X, C/K pattern (channels-last)
        else if constexpr(LAYOUT == TensorLayout::GNDHWC ||
                          LAYOUT == TensorLayout::G_NDHW_C_strided ||
                          LAYOUT == TensorLayout::GKZYXC ||
                          LAYOUT == TensorLayout::G_K_ZYX_C_strided ||
                          LAYOUT == TensorLayout::GNDHWK ||
                          LAYOUT == TensorLayout::G_NDHW_K_strided)
            return std::array<size_t, 6>{0, 1, 3, 4, 5, 2};
        // 3D layouts: Order {1, 3, 4, 5, 0, 2} - N/K, D/Z, H/Y, W/X, G, C/K pattern (batch-first)
        else if constexpr(LAYOUT == TensorLayout::NDHWGC || LAYOUT == TensorLayout::KZYXGC ||
                          LAYOUT == TensorLayout::NDHWGK)
            return std::array<size_t, 6>{1, 3, 4, 5, 0, 2};
        // 3D layouts: Order {1, 0, 2, 3, 4, 5} - N/K, G, C/K, D/Z, H/Y, W/X pattern
        else if constexpr(LAYOUT == TensorLayout::NGCDHW || LAYOUT == TensorLayout::NGKDHW)
            return std::array<size_t, 6>{1, 0, 2, 3, 4, 5};
        else
            static_assert(sizeof(UnsupportedEnumValue<LAYOUT>) == 0, "Unsupported 3D layout");
    }
}

} // namespace detail

/// @brief `Args` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see Args
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE>
struct Args<SIGNATURE>
{
    constexpr static auto SPATIAL_DIM = SIGNATURE.spatial_dim;
    constexpr static auto INPUT_TYPE  = SIGNATURE.data_type;
    constexpr static auto WEIGHT_TYPE = SIGNATURE.data_type;
    constexpr static auto OUTPUT_TYPE = SIGNATURE.data_type;

    constexpr static int INPUT_RANK  = 3 + SPATIAL_DIM;
    constexpr static int WEIGHT_RANK = 3 + SPATIAL_DIM;
    constexpr static int OUTPUT_RANK = 3 + SPATIAL_DIM;

    using InputDescriptor  = TensorDescriptor<INPUT_TYPE, INPUT_RANK>;
    using WeightDescriptor = TensorDescriptor<WEIGHT_TYPE, WEIGHT_RANK>;
    using OutputDescriptor = TensorDescriptor<OUTPUT_TYPE, OUTPUT_RANK>;

    // TODO: We shouldn't need to call into an internal namespace here.
    using Ops = factory::internal::ConvElementwiseOps<SIGNATURE>;

    // NOTE: ConvTensorLayouts removed - not used in this file and causes compilation error
    // TODO: We shouldn't need to call into an internal namespace here.
    // using Layouts = factory::internal::ConvTensorLayouts<SIGNATURE>;

    ConvTensorLengths<SPATIAL_DIM> lengths;

    // Optional explicit tensor-memory strides (in elements), for custom/non-packed tensors.
    // When not set, packed strides are derived automatically from the selected TensorLayout.
    // NOTE: These have explicit default initializers to avoid
    // -Wmissing-designated-field-initializers when `Args` is aggregate-initialized
    // using designated initializers in tests.
    std::optional<typename InputDescriptor::Extent> input_strides   = std::nullopt;
    std::optional<typename WeightDescriptor::Extent> weight_strides = std::nullopt;
    std::optional<typename OutputDescriptor::Extent> output_strides = std::nullopt;

    FilterExtent<SPATIAL_DIM> filter_strides;
    FilterExtent<SPATIAL_DIM> filter_dilation;
    FilterExtent<SPATIAL_DIM> input_left_pad;
    FilterExtent<SPATIAL_DIM> input_right_pad;

    Ops::InElementwiseOp a_elementwise_op;
    Ops::WeiElementwiseOp b_elementwise_op;
    Ops::OutElementwiseOp cde_elementwise_op;

    int k_batch = 1;

    /// @brief Compute output spatial dimensions from convolution parameters.
    ///
    /// @returns FilterExtent with computed output height, width, (and depth for 3D)
    FilterExtent<SPATIAL_DIM> compute_output_spatial() const
    {
        const auto input_spatial_arr  = this->lengths.image.template to_array<long_index_t>();
        const auto filter_spatial_arr = this->lengths.filter.template to_array<long_index_t>();
        const auto conv_strides_arr   = this->filter_strides.template to_array<long_index_t>();
        const auto conv_dilations_arr = this->filter_dilation.template to_array<long_index_t>();
        const auto left_pads_arr      = this->input_left_pad.template to_array<long_index_t>();
        const auto right_pads_arr     = this->input_right_pad.template to_array<long_index_t>();

        const auto output_spatial_arr =
            detail::compute_output_spatial<SPATIAL_DIM>(input_spatial_arr,
                                                        filter_spatial_arr,
                                                        conv_strides_arr,
                                                        conv_dilations_arr,
                                                        left_pads_arr,
                                                        right_pads_arr);

        return filter_extent_from_vector<SPATIAL_DIM>(
            std::vector<std::size_t>(output_spatial_arr.begin(), output_spatial_arr.end()));
    }

    /// This function returns the `TensorDescriptor` corresponding to
    /// the input-tensor of the convolution problem. This can then
    /// be used to, for example, allocate memory.
    InputDescriptor make_input_descriptor() const
    {
        using Extent = typename InputDescriptor::Extent;
        Extent lens  = {};

        lens[0] = this->lengths.groups;
        lens[1] = this->lengths.batch_size;
        lens[2] = this->lengths.input_channels;
        if constexpr(SPATIAL_DIM == 1)
        {
            lens[3] = this->lengths.image.width;
        }
        else if constexpr(SPATIAL_DIM == 2)
        {
            lens[3] = this->lengths.image.height;
            lens[4] = this->lengths.image.width;
        }
        else
        {
            lens[3] = this->lengths.image.depth;
            lens[4] = this->lengths.image.height;
            lens[5] = this->lengths.image.width;
        }

        const auto make_default_strides = [&] {
            constexpr auto layout = SIGNATURE.input.config.layout;
            constexpr auto order  = detail::get_layout_order<layout, SPATIAL_DIM>();
            return detail::make_packed_strides_for_order<INPUT_RANK>(lens, order);
        };

        const Extent strides =
            this->input_strides.has_value() ? *this->input_strides : make_default_strides();

        return InputDescriptor(lens, strides);
    }

    /// This function returns the `TensorDescriptor` corresponding to
    /// the weight-tensor of  the convolution problem. This can then
    /// be used to, for example, allocate memory.
    WeightDescriptor make_weight_descriptor() const
    {
        using Extent = typename WeightDescriptor::Extent;
        Extent lens  = {};

        lens[0] = this->lengths.groups;
        lens[1] = this->lengths.output_channels;
        lens[2] = this->lengths.input_channels;
        if constexpr(SPATIAL_DIM == 1)
        {
            lens[3] = this->lengths.filter.width;
        }
        else if constexpr(SPATIAL_DIM == 2)
        {
            lens[3] = this->lengths.filter.height;
            lens[4] = this->lengths.filter.width;
        }
        else
        {
            lens[3] = this->lengths.filter.depth;
            lens[4] = this->lengths.filter.height;
            lens[5] = this->lengths.filter.width;
        }

        const auto make_default_strides = [&] {
            constexpr auto layout = SIGNATURE.weight.config.layout;
            constexpr auto order  = detail::get_layout_order<layout, SPATIAL_DIM>();
            return detail::make_packed_strides_for_order<WEIGHT_RANK>(lens, order);
        };

        const Extent strides =
            this->weight_strides.has_value() ? *this->weight_strides : make_default_strides();

        return WeightDescriptor(lens, strides);
    }

    /// This function returns the `TensorDescriptor` corresponding to
    /// the output-tensor of the convolution problem. This can then
    /// be used to, for example, allocate memory.
    OutputDescriptor make_output_descriptor() const
    {
        using Extent = typename OutputDescriptor::Extent;
        Extent lens  = {};

        const auto output_spatial = compute_output_spatial();

        lens[0] = this->lengths.groups;
        lens[1] = this->lengths.batch_size;
        lens[2] = this->lengths.output_channels;
        if constexpr(SPATIAL_DIM == 1)
        {
            lens[3] = output_spatial.width;
        }
        else if constexpr(SPATIAL_DIM == 2)
        {
            lens[3] = output_spatial.height;
            lens[4] = output_spatial.width;
        }
        else
        {
            lens[3] = output_spatial.depth;
            lens[4] = output_spatial.height;
            lens[5] = output_spatial.width;
        }

        const auto make_default_strides = [&] {
            constexpr auto layout = SIGNATURE.output.config.layout;
            constexpr auto order  = detail::get_layout_order<layout, SPATIAL_DIM>();
            return detail::make_packed_strides_for_order<OUTPUT_RANK>(lens, order);
        };

        const Extent strides =
            this->output_strides.has_value() ? *this->output_strides : make_default_strides();

        return OutputDescriptor(lens, strides);
    }

    /// Convert the Args structure into a CK Tile conv_param structure.
    /// This function is mainly used to be able to use the existing
    /// CK Tile functionality to obtain tensor descriptors.
    ck_tile::conv::ConvParam to_ck_tile_conv_param() const
    {
        const auto to_vector = [](const auto& extent) {
            if constexpr(SPATIAL_DIM == 1)
                return std::vector<ck_tile::index_t>{ck::index_t(extent.width)};
            else if constexpr(SPATIAL_DIM == 2)
                return std::vector<ck_tile::index_t>{ck::index_t(extent.height),
                                                     ck::index_t(extent.width)};
            else
                return std::vector<ck_tile::index_t>{ck::index_t(extent.depth),
                                                     ck::index_t(extent.height),
                                                     ck::index_t(extent.width)};
        };

        return ck_tile::conv::ConvParam(SPATIAL_DIM,
                                        this->lengths.groups,
                                        this->lengths.batch_size,
                                        this->lengths.output_channels,
                                        this->lengths.input_channels,
                                        to_vector(this->lengths.filter),
                                        to_vector(this->lengths.image),
                                        to_vector(this->filter_strides),
                                        to_vector(this->filter_dilation),
                                        to_vector(this->input_left_pad),
                                        to_vector(this->input_right_pad));
    }
};

} // namespace ck_tile::builder::test
