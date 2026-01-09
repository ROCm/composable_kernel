// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/reflect/conv_traits_helpers.hpp"
#include "ck_tile/builder/reflect/conv_types.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"

namespace ck_tile::reflect::conv {

/// @brief Pure data struct holding convolution traits without template parameters or static
/// members.
/// @details This struct can hold the data from any ConvTraitsImpl class, allowing runtime storage
/// and manipulation of convolution configuration information.
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

    builder::GemmPadding gemm_padding;
    std::variant<builder::ConvFwdSpecialization,
                 builder::ConvBwdDataSpecialization,
                 builder::ConvBwdWeightSpecialization>
        conv_specialization;

    // --- Algorithm Information ---
    int thread_block_size;
    DataTileInfo tile_dims;

    InputTileTransferInfo a_tile_transfer;
    InputTileTransferInfo b_tile_transfer;

    WarpGemmParams warp_gemm;

    OutputTileTransferInfo c_tile_transfer;

    int num_gemm_prefetch_stage = 0;

    builder::PipelineVersion pipeline_version;
    builder::PipelineScheduler pipeline_scheduler;
};

} // namespace ck_tile::reflect::conv
