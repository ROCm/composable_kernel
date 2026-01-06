// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/reflect/conv_traits_helpers.hpp"
#include "ck_tile/builder/reflect/conv_types.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"

namespace ck_tile::reflect::conv {

/// @brief Primary template for extracting convolution traits.
/// @details This struct is the main entry point for reflecting on a convolution
/// kernel's properties. It is specialized to handle different kinds of input types.
template <typename T>
struct ConvTraits;

/// @brief Specialization of `ConvTraits` for a direct device kernel `Instance`.
/// @details This is the primary specialization used to extract a comprehensive
/// set of traits directly from a fully-formed device kernel `Instance` type.
/// It uses `InstanceTraits` to access the kernel's template parameters.
template <HasInstanceTraits Instance>
    requires IsXdlFwdConv<InstanceTraits<Instance>>
struct ConvTraits<Instance>
{
    using InstTraits = InstanceTraits<Instance>;

    // --- Signature Information ---
    /// @brief The number of spatial dimensions in the convolution (1, 2, or 3).
    static constexpr int spatial_dim = InstTraits::kSpatialDim;
    /// @brief The direction of the convolution (Forward, Backward Data, or Backward Weight).
    static constexpr builder::ConvDirection direction = conv_direction<Instance>();
    /// @brief The memory layout of the convolution tensors (e.g., GNHWC_GKYXC_GNHWK).
    static constexpr auto layout = conv_layout<Instance>();
    /// @brief The primary data type used in the computation (e.g., FP16, FP32).
    static constexpr builder::DataType data_type = conv_data_type<Instance>();

    static constexpr builder::ElementwiseOperation input_element_op =
        elementwise_op<typename InstTraits::AElementwiseOperation>();
    static constexpr builder::ElementwiseOperation weight_element_op =
        elementwise_op<typename InstTraits::BElementwiseOperation>();
    static constexpr builder::ElementwiseOperation output_element_op =
        elementwise_op<typename InstTraits::CDEElementwiseOperation>();

    /// @brief The GEMM specialization used by the kernel - padding
    static constexpr auto gemm_padding = gemm_spec<Instance>();
    /// @brief The convolution-specific specialization (e.g., Default, 1x1).
    static constexpr auto conv_specialization = conv_spec<Instance>();

    // --- Algorithm Information ---
    /// @brief The total number of threads in a thread block (workgroup).
    static constexpr int thread_block_size = InstTraits::kBlockSize;
    /// @brief The dimensions of the data tile processed by the thread block.
    static constexpr DataTileInfo tile_dims = {
        .m = InstTraits::kMPerBlock, .n = InstTraits::kNPerBlock, .k = InstTraits::kKPerBlock};

    /// @brief Configuration for the A-matrix (input) tile transfer.
    static constexpr InputTileTransferInfo a_tile_transfer = {
        .tile_dimensions = {.k0     = InstTraits::kKPerBlock / InstTraits::kAK1,
                            .m_or_n = InstTraits::kMPerBlock,
                            .k1     = InstTraits::kAK1},
        .transfer_params = {.k1                    = InstTraits::kAK1,
                            .thread_cluster_dims   = InstTraits::kAThreadClusterLengths,
                            .thread_cluster_order  = InstTraits::kAThreadClusterArrangeOrder,
                            .src_access_order      = InstTraits::kABlockTransferSrcAccessOrder,
                            .src_vector_dim        = InstTraits::kABlockTransferSrcVectorDim,
                            .src_scalar_per_vector = InstTraits::kABlockTransferSrcScalarPerVector,
                            .dst_scalar_per_vector_k1 =
                                InstTraits::kABlockTransferDstScalarPerVectorK1,
                            .lds_padding = static_cast<bool>(InstTraits::kABlockLdsExtraM)}};

    /// @brief Configuration for the B-matrix (weights) tile transfer.
    static constexpr InputTileTransferInfo b_tile_transfer = {
        .tile_dimensions = {.k0     = InstTraits::kKPerBlock / InstTraits::kBK1,
                            .m_or_n = InstTraits::kNPerBlock,
                            .k1     = InstTraits::kBK1},
        .transfer_params = {.k1                    = InstTraits::kBK1,
                            .thread_cluster_dims   = InstTraits::kBThreadClusterLengths,
                            .thread_cluster_order  = InstTraits::kBThreadClusterArrangeOrder,
                            .src_access_order      = InstTraits::kBBlockTransferSrcAccessOrder,
                            .src_vector_dim        = InstTraits::kBBlockTransferSrcVectorDim,
                            .src_scalar_per_vector = InstTraits::kBBlockTransferSrcScalarPerVector,
                            .dst_scalar_per_vector_k1 =
                                InstTraits::kBBlockTransferDstScalarPerVectorK1,
                            .lds_padding = static_cast<bool>(InstTraits::kBBlockLdsExtraN)}};

    /// @brief Parameters for the warp-level GEMM computation.
    static constexpr WarpGemmParams warp_gemm = {.gemm_m = InstTraits::kMPerXDL,
                                                 .gemm_n = InstTraits::kNPerXDL,
                                                 .m_iter = InstTraits::kMXdlPerWave,
                                                 .n_iter = InstTraits::kNXdlPerWave};

    /// @brief Configuration for the C-matrix (output) tile transfer.
    static constexpr OutputTileTransferInfo c_tile_transfer = {
        .shuffle_params      = {.m_gemms_per_shuffle = InstTraits::kCShuffleMXdlPerWavePerShuffle,
                                .n_gemms_per_shuffle = InstTraits::kCShuffleNXdlPerWavePerShuffle},
        .thread_cluster_dims = {InstTraits::kCThreadClusterLengths[0],
                                InstTraits::kCThreadClusterLengths[1],
                                InstTraits::kCThreadClusterLengths[2],
                                InstTraits::kCThreadClusterLengths[3]},
        .scalar_per_vector   = InstTraits::kCBlockTransferScalarPerVector};

    /// @brief The block GEMM pipeline version used by the kernel.
    static constexpr auto pipeline_version = get_pipeline_version<InstTraits>();

    /// @brief The pipeline scheduler used by the kernel.
    static constexpr auto pipeline_scheduler = get_pipeline_scheduler<InstTraits>();
};

} // namespace ck_tile::reflect::conv
