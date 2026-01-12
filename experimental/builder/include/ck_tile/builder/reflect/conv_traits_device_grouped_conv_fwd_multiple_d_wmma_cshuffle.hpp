// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>

#include "ck_tile/builder/reflect/conv_traits.hpp"
#include "ck_tile/builder/reflect/conv_traits_helpers.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"

namespace ck_tile::reflect::conv {

/// @brief Tag dispatch implementation for DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle
template <typename Instance>
    requires HasInstanceTraits<Instance> &&
             std::same_as<typename InstanceTraits<Instance>::device_kernel_tag,
                          DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_Tag>
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
        .tile_dims           = {.m = InstTraits::kMPerBlock,
                                .n = InstTraits::kNPerBlock,
                                .k = InstTraits::kKPerBlock},
        .a_tile_transfer =
            {.tile_dimensions = {.k0     = InstTraits::kKPerBlock / InstTraits::kK1,
                                 .m_or_n = InstTraits::kMPerBlock,
                                 .k1     = InstTraits::kK1},
             .transfer_params = {.k1                   = InstTraits::kK1,
                                 .thread_cluster_dims  = InstTraits::kAThreadClusterLengths,
                                 .thread_cluster_order = InstTraits::kAThreadClusterArrangeOrder,
                                 .src_access_order     = InstTraits::kABlockTransferSrcAccessOrder,
                                 .src_vector_dim       = InstTraits::kABlockTransferSrcVectorDim,
                                 .src_scalar_per_vector =
                                     InstTraits::kABlockTransferSrcScalarPerVector,
                                 .dst_scalar_per_vector_k1 =
                                     InstTraits::kABlockTransferDstScalarPerVectorK1,
                                 .lds_padding = static_cast<bool>(InstTraits::kABlockLdsExtraM)}},
        .b_tile_transfer =
            {.tile_dimensions = {.k0     = InstTraits::kKPerBlock / InstTraits::kK1,
                                 .m_or_n = InstTraits::kNPerBlock,
                                 .k1     = InstTraits::kK1},
             .transfer_params = {.k1                   = InstTraits::kK1,
                                 .thread_cluster_dims  = InstTraits::kBThreadClusterLengths,
                                 .thread_cluster_order = InstTraits::kBThreadClusterArrangeOrder,
                                 .src_access_order     = InstTraits::kBBlockTransferSrcAccessOrder,
                                 .src_vector_dim       = InstTraits::kBBlockTransferSrcVectorDim,
                                 .src_scalar_per_vector =
                                     InstTraits::kBBlockTransferSrcScalarPerVector,
                                 .dst_scalar_per_vector_k1 =
                                     InstTraits::kBBlockTransferDstScalarPerVectorK1,
                                 .lds_padding = static_cast<bool>(InstTraits::kBBlockLdsExtraN)}},
        .warp_gemm       = {.gemm_m = InstTraits::kMPerWmma,
                            .gemm_n = InstTraits::kNPerWmma,
                            .m_iter = InstTraits::kMRepeat,
                            .n_iter = InstTraits::kNRepeat},
        .c_tile_transfer = {.shuffle_params      = {.m_gemms_per_shuffle =
                                                        InstTraits::kCShuffleMRepeatPerShuffle,
                                                    .n_gemms_per_shuffle =
                                                        InstTraits::kCShuffleNRepeatPerShuffle},
                            .thread_cluster_dims = {InstTraits::kCDEThreadClusterLengths[0],
                                                    InstTraits::kCDEThreadClusterLengths[1],
                                                    InstTraits::kCDEThreadClusterLengths[2],
                                                    InstTraits::kCDEThreadClusterLengths[3]},
                            .scalar_per_vector   = InstTraits::kCDEBlockTransferScalarPerVector},
        //   .num_gemm_prefetch_stage = InstTraits::kNumGemmKPrefetchStage,
        .pipeline_version   = get_pipeline_version<InstTraits>(),
        .pipeline_scheduler = get_pipeline_scheduler<InstTraits>(),
    };
}

} // namespace ck_tile::reflect::conv
