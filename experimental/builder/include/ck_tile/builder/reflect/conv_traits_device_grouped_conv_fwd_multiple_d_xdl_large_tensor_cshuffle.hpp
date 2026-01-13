// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>

#include "ck_tile/builder/reflect/conv_traits.hpp"
#include "ck_tile/builder/reflect/conv_traits_helpers.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp"

namespace ck_tile::reflect::conv {

/// @brief Tag dispatch implementation for DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor
template <typename Instance>
    requires HasInstanceTraits<Instance> &&
             std::same_as<typename InstanceTraits<Instance>::device_kernel_tag,
                          DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Tag>
constexpr ConvTraits instance_to_conv_traits()
{
    using InstTraits = InstanceTraits<Instance>;

    return ConvTraits{
        .spatial_dim         = InstTraits::kSpatialDim,
        .direction           = conv_direction<Instance>(),
        .layout              = conv_layout<Instance>(),
        .data_type           = conv_data_type<Instance>(),
        .input_element_op    = elementwise_op<typename InstTraits::AElementwiseOperation>(),
        .weight_element_op   = elementwise_op<typename InstTraits::BElementwiseOperation>(),
        .output_element_op   = elementwise_op<typename InstTraits::CDEElementwiseOperation>(),
        .gemm_padding        = gemm_spec<Instance>(),
        .conv_specialization = conv_spec<Instance>(),
        .thread_block_size   = InstTraits::kBlockSize,
        .tile_dims           = conv_traits_data_tile<InstTraits>(),
        .a_tile_transfer     = conv_traits_xdl_a_transfer_params<InstTraits>(),
        .b_tile_transfer     = conv_traits_xdl_b_transfer_params<InstTraits>(),
        .warp_gemm           = conv_traits_xdl_warp_gemm_params<InstTraits>(),
        .c_tile_transfer     = conv_traits_xdl_c_tile_transfer<InstTraits>(),
        .pipeline_version    = get_pipeline_version<InstTraits>(),
        .pipeline_scheduler  = get_pipeline_scheduler<InstTraits>(),
    };
}

} // namespace ck_tile::reflect::conv
