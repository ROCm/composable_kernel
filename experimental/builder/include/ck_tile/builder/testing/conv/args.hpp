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

            if constexpr(SPATIAL_DIM == 1)
            {
                if constexpr(layout == TensorLayout::GNCW)
                    return detail::make_packed_strides_for_order<4>(lens, {0, 1, 2, 3});
                else if constexpr(layout == TensorLayout::GNWC ||
                                  layout == TensorLayout::G_NW_C_strided)
                    return detail::make_packed_strides_for_order<4>(lens, {0, 1, 3, 2});
                else if constexpr(layout == TensorLayout::NWGC)
                    return detail::make_packed_strides_for_order<4>(lens, {1, 3, 0, 2});
                else if constexpr(layout == TensorLayout::NGCW)
                    return detail::make_packed_strides_for_order<4>(lens, {1, 0, 2, 3});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 1D input layout for descriptor initialization.");
            }
            else if constexpr(SPATIAL_DIM == 2)
            {
                if constexpr(layout == TensorLayout::GNCHW)
                    return detail::make_packed_strides_for_order<5>(lens, {0, 1, 2, 3, 4});
                else if constexpr(layout == TensorLayout::GNHWC ||
                                  layout == TensorLayout::G_NHW_C_strided)
                    return detail::make_packed_strides_for_order<5>(lens, {0, 1, 3, 4, 2});
                else if constexpr(layout == TensorLayout::NHWGC)
                    return detail::make_packed_strides_for_order<5>(lens, {1, 3, 4, 0, 2});
                else if constexpr(layout == TensorLayout::NGCHW)
                    return detail::make_packed_strides_for_order<5>(lens, {1, 0, 2, 3, 4});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 2D input layout for descriptor initialization.");
            }
            else
            {
                if constexpr(layout == TensorLayout::GNCDHW)
                    return detail::make_packed_strides_for_order<6>(lens, {0, 1, 2, 3, 4, 5});
                else if constexpr(layout == TensorLayout::GNDHWC ||
                                  layout == TensorLayout::G_NDHW_C_strided)
                    return detail::make_packed_strides_for_order<6>(lens, {0, 1, 3, 4, 5, 2});
                else if constexpr(layout == TensorLayout::NDHWGC)
                    return detail::make_packed_strides_for_order<6>(lens, {1, 3, 4, 5, 0, 2});
                else if constexpr(layout == TensorLayout::NGCDHW)
                    return detail::make_packed_strides_for_order<6>(lens, {1, 0, 2, 3, 4, 5});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 3D input layout for descriptor initialization.");
            }
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

            if constexpr(SPATIAL_DIM == 1)
            {
                if constexpr(layout == TensorLayout::GKCX)
                    return detail::make_packed_strides_for_order<4>(lens, {0, 1, 2, 3});
                else if constexpr(layout == TensorLayout::GKXC ||
                                  layout == TensorLayout::G_K_X_C_strided)
                    return detail::make_packed_strides_for_order<4>(lens, {0, 1, 3, 2});
                else if constexpr(layout == TensorLayout::KXGC)
                    return detail::make_packed_strides_for_order<4>(lens, {1, 3, 0, 2});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 1D weight layout for descriptor initialization.");
            }
            else if constexpr(SPATIAL_DIM == 2)
            {
                if constexpr(layout == TensorLayout::GKCYX)
                    return detail::make_packed_strides_for_order<5>(lens, {0, 1, 2, 3, 4});
                else if constexpr(layout == TensorLayout::GKYXC ||
                                  layout == TensorLayout::G_K_YX_C_strided)
                    return detail::make_packed_strides_for_order<5>(lens, {0, 1, 3, 4, 2});
                else if constexpr(layout == TensorLayout::KYXGC)
                    return detail::make_packed_strides_for_order<5>(lens, {1, 3, 4, 0, 2});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 2D weight layout for descriptor initialization.");
            }
            else
            {
                if constexpr(layout == TensorLayout::GKCZYX)
                    return detail::make_packed_strides_for_order<6>(lens, {0, 1, 2, 3, 4, 5});
                else if constexpr(layout == TensorLayout::GKZYXC ||
                                  layout == TensorLayout::G_K_ZYX_C_strided)
                    return detail::make_packed_strides_for_order<6>(lens, {0, 1, 3, 4, 5, 2});
                else if constexpr(layout == TensorLayout::KZYXGC)
                    return detail::make_packed_strides_for_order<6>(lens, {1, 3, 4, 5, 0, 2});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 3D weight layout for descriptor initialization.");
            }
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

            if constexpr(SPATIAL_DIM == 1)
            {
                if constexpr(layout == TensorLayout::GNKW)
                    return detail::make_packed_strides_for_order<4>(lens, {0, 1, 2, 3});
                else if constexpr(layout == TensorLayout::GNWK ||
                                  layout == TensorLayout::G_NW_K_strided)
                    return detail::make_packed_strides_for_order<4>(lens, {0, 1, 3, 2});
                else if constexpr(layout == TensorLayout::NWGK)
                    return detail::make_packed_strides_for_order<4>(lens, {1, 3, 0, 2});
                else if constexpr(layout == TensorLayout::NGKW)
                    return detail::make_packed_strides_for_order<4>(lens, {1, 0, 2, 3});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 1D output layout for descriptor initialization.");
            }
            else if constexpr(SPATIAL_DIM == 2)
            {
                if constexpr(layout == TensorLayout::GNKHW)
                    return detail::make_packed_strides_for_order<5>(lens, {0, 1, 2, 3, 4});
                else if constexpr(layout == TensorLayout::GNHWK ||
                                  layout == TensorLayout::G_NHW_K_strided)
                    return detail::make_packed_strides_for_order<5>(lens, {0, 1, 3, 4, 2});
                else if constexpr(layout == TensorLayout::NHWGK)
                    return detail::make_packed_strides_for_order<5>(lens, {1, 3, 4, 0, 2});
                else if constexpr(layout == TensorLayout::NGKHW)
                    return detail::make_packed_strides_for_order<5>(lens, {1, 0, 2, 3, 4});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 2D output layout for descriptor initialization.");
            }
            else
            {
                if constexpr(layout == TensorLayout::GNKDHW)
                    return detail::make_packed_strides_for_order<6>(lens, {0, 1, 2, 3, 4, 5});
                else if constexpr(layout == TensorLayout::GNDHWK ||
                                  layout == TensorLayout::G_NDHW_K_strided)
                    return detail::make_packed_strides_for_order<6>(lens, {0, 1, 3, 4, 5, 2});
                else if constexpr(layout == TensorLayout::NDHWGK)
                    return detail::make_packed_strides_for_order<6>(lens, {1, 3, 4, 5, 0, 2});
                else if constexpr(layout == TensorLayout::NGKDHW)
                    return detail::make_packed_strides_for_order<6>(lens, {1, 0, 2, 3, 4, 5});
                else
                    static_assert(sizeof(UnsupportedEnumValue<layout>) == 0,
                                  "Unsupported 3D output layout for descriptor initialization.");
            }
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
