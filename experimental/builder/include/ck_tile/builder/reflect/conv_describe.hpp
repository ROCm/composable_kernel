// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file
/// @brief Implementation of the describe() function template for convolution kernels

#pragma once

#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck_tile/builder/reflect/instance_to_conv_traits.hpp"

namespace ck_tile::reflect {

/// @brief Factory function to create ConvDescription from a convolution instance type
/// @tparam Instance The convolution instance type (must have ConvTraits)
/// @return A ConvDescription object populated with the instance's configuration details
template <conv::HasConvTraits Instance>
conv::ConvDescription describe()
{
    const auto traits = conv::instance_to_conv_traits<Instance>();

    return conv::ConvDescription(
        conv::ConvSignatureInfo{
            .spatial_dim       = traits.spatial_dim,
            .direction         = traits.direction,
            .input_layout      = traits.layout[0],
            .weight_layout     = traits.layout[1],
            .output_layout     = traits.layout[2],
            .data_type         = traits.data_type,
            .input_element_op  = traits.input_element_op,
            .weight_element_op = traits.weight_element_op,
            .output_element_op = traits.output_element_op,
        },
        conv::GemmAlgorithmInfo{
            .thread_block_size   = traits.thread_block_size,
            .tile_dims           = traits.tile_dims,
            .warp_gemm           = traits.warp_gemm,
            .a_tile_transfer     = traits.a_tile_transfer,
            .b_tile_transfer     = traits.b_tile_transfer,
            .c_tile_transfer     = traits.c_tile_transfer,
            .pipeline_version    = traits.pipeline_version,
            .pipeline_scheduler  = traits.pipeline_scheduler,
            .conv_specialization = traits.conv_specialization,
            .padding             = traits.gemm_padding,
        },
        []() { return reflect::instance_string<Instance>(); });
}

} // namespace ck_tile::reflect
