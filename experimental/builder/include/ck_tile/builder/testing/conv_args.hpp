// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_utils.hpp"
#include "ck_tile/builder/conv_factory.hpp"
#include "ck_tile/builder/testing/tensor_memory_manager.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

/// This file implements common functionality for invoking/testing grouped
/// forward convolutions created through the CK Builder API. The main item
/// of it is the ConvArgs structure - which contains a complete description
/// of a convolution operation.
///
/// It is not intended that this file contains implementation details for
/// actually launching a convolution operation. As this can be done
/// through different APIs depending on the kernel (CK, CK Tile, or a
/// reference implementation), the code dealing with that is split out
/// into a separate header for each implementation.

namespace ck_tile::builder::test {

/// This structure describes a 1-, 2-, or 3-D extent. Its used to
/// communicate 1-, 2- or 3-D sizes and strides of tensors.
template <int SPATIAL_DIM>
struct ConvExtent;

template <>
struct ConvExtent<1>
{
    size_t width = 1;
};

template <>
struct ConvExtent<2>
{
    size_t width  = 1;
    size_t height = 1;
};

template <>
struct ConvExtent<3>
{
    size_t width  = 1;
    size_t height = 1;
    size_t depth  = 1;
};

using ConvExtent1D = ConvExtent<1>;
using ConvExtent2D = ConvExtent<2>;
using ConvExtent3D = ConvExtent<3>;

/// This structure is used to describe lengths of a convolution problem. In fact, this
/// structure is a complete description of ALL inputs and outputs lengths of a convolution
/// problem, as this structure contains all of the combined parameters. Note that we can't
/// also use this structure to describe tensor strides: whereas the lengths are all governed
/// by a common set of parameters, strides of the input, weight, and output tensor are all
/// independent.
template <int SPATIAL_DIM>
struct ConvTensorLengths
{
    size_t batch_size              = 1;  // N
    size_t groups                  = 1;  // G
    size_t input_channels          = 1;  // C
    size_t output_channels         = 1;  // K
    ConvExtent<SPATIAL_DIM> image  = {}; // W, H, D
    ConvExtent<SPATIAL_DIM> filter = {}; // X, Y, Z
};

/// The ConvArgs structure is the runtime counterpart of the `ConvSignature`: it contains the
/// runtime values for a convolution operation, and forms a complete description of such an
/// operation together with the signature.
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE>
struct ConvArgs
{
    constexpr static auto SPATIAL_DIM = SIGNATURE.spatial_dim;
    constexpr static auto INPUT_TYPE  = SIGNATURE.data_type;
    constexpr static auto WEIGHT_TYPE = SIGNATURE.data_type;
    constexpr static auto OUTPUT_TYPE = SIGNATURE.data_type;

    using Ops = factory_internal::ElementwiseOps<get_elementwise_operation<SIGNATURE>()>;

    ConvTensorLengths<SPATIAL_DIM> lengths;
    // TODO(Robin): Tensor strides. This needs a new structure as well as some reworking
    // of the TensorDescriptor, as the current implementation (based on ConvParam in old CK/
    // CK Tile) does not support strides at all.

    ConvExtent<SPATIAL_DIM> filter_strides;
    ConvExtent<SPATIAL_DIM> filter_dilation;
    ConvExtent<SPATIAL_DIM> input_left_pad;
    ConvExtent<SPATIAL_DIM> input_right_pad;

    Ops::AElementwiseOp a_elementwise_op;
    Ops::BElementwiseOp b_elementwise_op;
    Ops::CDEElementwiseOp cde_elementwise_op;

    // TODO(Robin): We shouldn't need to call into an internal namespace here.
    using Layouts =
        decltype(ck_tile::builder::factory_internal::
                     GetTensorLayout<SIGNATURE.layout, SPATIAL_DIM, ConvDirection::FORWARD>());

    /// This function returns the `TensorDescriptor` corresponding to the input-tensor of
    /// the convolution problem. This can then be used to, for example, allocate memory.
    TensorDescriptor<INPUT_TYPE> make_input_descriptor() const
    {
        // TODO: We're using old CK functionality to compute the right values here, mainly
        // because CK tile does not support the right tensor layouts here. We should probably
        // change that because CK currently prints an annoying message about it, plus that
        // would let us get rid of the `to_ck_conv_param()` function.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
             typename Layouts::ALayout>(param);
        return TensorDescriptor<INPUT_TYPE>(desc.GetLengths(), desc.GetStrides());
    }

    /// This function returns the `TensorDescriptor` corresponding to the weight-tensor of
    /// the convolution problem. This can then be used to, for example, allocate memory.
    TensorDescriptor<WEIGHT_TYPE> make_weight_descriptor() const
    {
        // See note in implementation of `make_input_descriptor`.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
             typename Layouts::BLayout>(param);
        return TensorDescriptor<WEIGHT_TYPE>(desc.GetLengths(), desc.GetStrides());
    }

    /// This function returns the `TensorDescriptor` corresponding to the output-tensor of
    /// the convolution problem. This can then be used to, for example, allocate memory.
    TensorDescriptor<OUTPUT_TYPE> make_output_descriptor() const
    {
        // See note in implementation of `make_input_descriptor`.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
             typename Layouts::ELayout>(param);
        return TensorDescriptor<OUTPUT_TYPE>(desc.GetLengths(), desc.GetStrides());
    }

    ck::utils::conv::ConvParam to_ck_conv_param() const
    {
        const auto to_vector = [](const auto& extent) {
            std::vector<ck::index_t> result;
            result.reserve(SPATIAL_DIM);

            if constexpr(SPATIAL_DIM >= 3)
                result.push_back(extent.depth);

            if constexpr(SPATIAL_DIM >= 2)
                result.push_back(extent.height);

            result.push_back(extent.width);
            return result;
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
};

/// This function can be used to directly allocate an input buffer that is compatible
/// with the `args` structure.
template <auto SIGNATURE>
DeviceBuffer alloc_input_buffer(const ConvArgs<SIGNATURE>& args)
{
    return alloc_tensor_buffer(args.make_input_descriptor());
}

/// This function can be used to directly allocate a weight buffer that is compatible
/// with the `args` structure.
template <auto SIGNATURE>
DeviceBuffer alloc_weight_buffer(const ConvArgs<SIGNATURE>& args)
{
    return alloc_tensor_buffer(args.make_weight_descriptor());
}

/// This function can be used to directly allocate an output buffer that is compatible
/// with the `args` structure.
template <auto SIGNATURE>
DeviceBuffer alloc_output_buffer(const ConvArgs<SIGNATURE>& args)
{
    return alloc_tensor_buffer(args.make_output_descriptor());
}

} // namespace ck_tile::builder::test
