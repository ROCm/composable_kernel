// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include "ck_tile/builder/testing/testing.hpp"
#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/tensor_descriptor.hpp"
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

    // TODO: We shouldn't need to call into an internal namespace here.
    using Layouts = factory::internal::ConvTensorLayouts<SIGNATURE>;

    ConvTensorLengths<SPATIAL_DIM> lengths;

    // TODO: Tensor strides. This needs a new structure as well as some
    // reworking of the make_*_descriptor() functions, as the current
    // implementation (based on ConvParam in old CK / CK Tile) does not
    // support strides at all.

    FilterExtent<SPATIAL_DIM> filter_strides;
    FilterExtent<SPATIAL_DIM> filter_dilation;
    FilterExtent<SPATIAL_DIM> input_left_pad;
    FilterExtent<SPATIAL_DIM> input_right_pad;

    Ops::InElementwiseOp a_elementwise_op;
    Ops::WeiElementwiseOp b_elementwise_op;
    Ops::OutElementwiseOp cde_elementwise_op;

    int k_batch = 1;

    /// This function returns the `TensorDescriptor` corresponding to
    /// the input-tensor of the convolution problem. This can then
    /// be used to, for example, allocate memory.
    InputDescriptor make_input_descriptor() const
    {
        // TODO: We're using old CK functionality to compute the right
        // values here, mainly because CK tile does not support the
        // right tensor layouts here. We should probably change that
        // because CK currently prints an annoying message about it,
        // plus that would let us get rid of the `to_ck_conv_param()`
        // function.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
             typename Layouts::InLayout>(param);
        using Extent = typename InputDescriptor::Extent;
        return InputDescriptor(Extent::from_vector(desc.GetLengths()),
                               Extent::from_vector(desc.GetStrides()));
    }

    /// This function returns the `TensorDescriptor` corresponding to
    /// the weight-tensor of  the convolution problem. This can then
    /// be used to, for example, allocate memory.
    WeightDescriptor make_weight_descriptor() const
    {
        // See note in implementation of `make_input_descriptor`.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
             typename Layouts::WeiLayout>(param);
        using Extent = typename WeightDescriptor::Extent;
        return WeightDescriptor(Extent::from_vector(desc.GetLengths()),
                                Extent::from_vector(desc.GetStrides()));
    }

    /// This function returns the `TensorDescriptor` corresponding to
    /// the output-tensor of the convolution problem. This can then
    /// be used to, for example, allocate memory.
    OutputDescriptor make_output_descriptor() const
    {
        // See note in implementation of `make_input_descriptor`.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
             typename Layouts::OutLayout>(param);
        using Extent = typename OutputDescriptor::Extent;
        return OutputDescriptor(Extent::from_vector(desc.GetLengths()),
                                Extent::from_vector(desc.GetStrides()));
    }

    /// Convert the Args structure into a CK conv_param structure. This
    /// function is mainly used to be able to use the existing
    /// CK-functionality to obtain tensor descriptors.
    ck::utils::conv::ConvParam to_ck_conv_param() const
    {
        const auto to_vector = [](const auto& extent) {
            if constexpr(SPATIAL_DIM == 1)
                return std::vector<ck::index_t>{ck::index_t(extent.width)};
            else if constexpr(SPATIAL_DIM == 2)
                return std::vector<ck::index_t>{ck::index_t(extent.height),
                                                ck::index_t(extent.width)};
            else
                return std::vector<ck::index_t>{ck::index_t(extent.depth),
                                                ck::index_t(extent.height),
                                                ck::index_t(extent.width)};
        };

        return ck::utils::conv::ConvParam(SPATIAL_DIM,
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
