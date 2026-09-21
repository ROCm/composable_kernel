// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>

#include "ck_tile/builder/reflect/conv_traits.hpp"
#include "ck_tile/builder/reflect/conv_traits_helpers.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"

namespace ck_tile::reflect::conv {

/// @brief Tag dispatch implementation for DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
template <typename Instance>
    requires HasInstanceTraits<Instance> &&
             std::same_as<typename InstanceTraits<Instance>::device_kernel_tag,
                          DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3_Tag>
constexpr ConvTraits instance_to_conv_traits()
{
    using InstTraits = InstanceTraits<Instance>;

    return ConvTraits{
        .spatial_dim         = InstTraits::kSpatialDim,
        .direction           = conv_direction<Instance>(),
        .layout              = fwd_conv_layout<Instance>(),
        .data_type           = conv_data_type<typename InstTraits::ADataType>(),
        .input_element_op    = elementwise_op<typename InstTraits::AElementwiseOperation>(),
        .weight_element_op   = elementwise_op<typename InstTraits::BElementwiseOperation>(),
        .output_element_op   = elementwise_op<typename InstTraits::CDEElementwiseOperation>(),
        .gemm_padding        = gemm_spec<Instance>(),
        .conv_specialization = conv_spec<Instance>(),
        .thread_block_size   = InstTraits::kBlockSize,
        .tile_dims           = conv_traits_data_tile<InstTraits>(),
        .a_tile_transfer     = conv_traits_a_transfer_params<InstTraits>(InstTraits::kAK1),
        .b_tile_transfer     = conv_traits_b_transfer_params<InstTraits>(InstTraits::kBK1),
        .warp_gemm           = conv_traits_wmma_warp_gemm_params<InstTraits>(),
        .c_tile_transfer     = conv_traits_wmma_c_tile_transfer<InstTraits>(),
        // TODO: Add compute types (AComputeDataType, BComputeDataType) when ConvTraits supports
        // them
        // TODO: Add NumGroupsToMerge when ConvTraits supports it
        .pipeline_version   = get_pipeline_version<InstTraits>(),
        .pipeline_scheduler = get_pipeline_scheduler<InstTraits>(),
    };
}

} // namespace ck_tile::reflect::conv
