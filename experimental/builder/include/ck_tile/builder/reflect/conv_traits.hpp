// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Runtime-accessible convolution kernel configuration data structure
//
// This file defines ConvTraits, a pure data structure that captures the complete
// configuration of a convolution kernel in a domain-specific abstraction, without
// requiring knowledge of the underlying kernel instance implementation details.
//
// ## Purpose and Design
//
// ConvTraits provides type erasure for convolution kernel configurations, allowing
// for reflection of convolution kernel objects. The struct represents kernel
// traits in terms of convolution-specific concepts for AMD GPUs rather than raw
// template parameters.
//
// ## Architecture and Usage
//
// ConvTraits sits at the center of the reflection system:
//
// 1. **Population**: Values are created by `instance_to_conv_traits()` template
//    specializations that extract configuration from compile-time InstanceTraits
//
// 2. **Consumption**: Used by ConvDescription to provide human-readable descriptions
//    of kernel configurations for debugging, logging, and documentation
//
// ## Structure Organization
//
// The struct separates kernel configuration into two logical categories:
//
// - **Signature Information**: Defines what the kernel computes (direction, layouts,
//   data types, elementwise operations, specializations)
//
// - **Algorithm Information**: Defines how the kernel computes (thread block size,
//   tile dimensions, memory access patterns, pipeline configuration)
//
// ## Evolution and Extensibility
//
// ConvTraits is designed to evolve through composition (not inheritance):
//
// - Currently supports XDL forward convolution kernels
// - Will extend to the other forward convolutions
// - Will be extended to cover backward data and backward weight convolutions
// - Will incorporate fusion operations and additional specializations
// - Uses std::optional and std::variant for optional/variant fields
// - Eventually will generalize to KernelTraits for GEMM, flash attention, etc.

#pragma once

#include "ck_tile/builder/reflect/conv_types.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::reflect::conv {

// Runtime data structure representing a convolution kernel's complete configuration
//
// This pure data struct (no template parameters, no static members) provides
// type erasure for convolution kernel configurations. It can hold the configuration
// from any convolution kernel instance, enabling runtime storage, comparison, and
// manipulation of kernel properties.
//
// The struct is populated by `instance_to_conv_traits()` template specializations
// that extract compile-time configuration from InstanceTraits and convert it to
// this standardized runtime representation.
//
// Members are organized into two categories:
// - **Signature Information**: Defines the computational interface (what to compute)
// - **Algorithm Information**: Defines the implementation strategy (how to compute)
//
// Note: This struct will evolve to support additional convolution variants and
// eventually generalize to other kernel types through composition.
//
// There is a lot we still need to do:
//
// TODO: Generalize type support for all tensors and accumulator.
// TODO: Describe all tensors.
// TODO: Include the full generalization of the signature from the input schema.
// TODO: Include the full generalization of the algorithm from the input schema.
struct ConvTraits
{
    // --- Signature Information ---
    int spatial_dim;
    builder::ConvDirection direction;
    std::array<builder::TensorLayout, 3> layout; // [input, weight, output]
    builder::DataType data_type;

    builder::ElementwiseOperation input_element_op;
    builder::ElementwiseOperation weight_element_op;
    builder::ElementwiseOperation output_element_op;

    std::optional<builder::GemmPadding> gemm_padding = std::nullopt;
    builder::ConvSpecialization conv_specialization;

    // --- Algorithm Information ---
    int thread_block_size;
    DataTileInfo tile_dims;

    InputTileTransferInfo a_tile_transfer;
    InputTileTransferInfo b_tile_transfer;

    WarpGemmParams warp_gemm;

    OutputTileTransferInfo c_tile_transfer;

    std::optional<int> num_gemm_k_prefetch_stage = std::nullopt;

    builder::PipelineVersion pipeline_version;
    builder::PipelineScheduler pipeline_scheduler;

    std::optional<int> max_transpose_transfer_src_scalar_per_vector = std::nullopt;
    std::optional<int> max_transpose_transfer_dst_scalar_per_vector = std::nullopt;
    std::optional<int> num_groups_to_merge                          = std::nullopt;
    std::optional<bool> do_pad_gemm_m                               = std::nullopt;
    std::optional<bool> do_pad_gemm_n                               = std::nullopt;
};

} // namespace ck_tile::reflect::conv
