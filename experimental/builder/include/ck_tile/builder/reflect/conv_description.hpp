// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Provides utilities to reflect on convolution kernel instances and generate
 * human-readable descriptions of their configuration.
 *
 * This file contains the necessary components to transform a convolution kernel's
 * compile-time properties into a structured, descriptive format. This is primarily
 * used for debugging, logging, and generating documentation.
 *
 * Key components:
 * - ck_tile::reflect::conv::ConvDescription: A struct that holds the extracted
 *   properties and provides methods to format them into strings.
 * - ck_tile::reflect::conv::Describe(): A factory function that creates a
 *   ConvDescription from a given kernel instance type.
 */

#pragma once

#include <concepts>
#include <string_view>
#include <sstream>
#include <type_traits>
#include <variant>

#include <ck_tile/builder/conv_signature_concepts.hpp>
#include <ck_tile/builder/reflect/conv_traits.hpp>
#include <ck_tile/builder/reflect/tree_formatter.hpp>

/// @brief Provides human-readable descriptions of convolution kernel instances

namespace ck_tile::reflect::conv {

/// @brief Signature information for a convolution operation
/// Contains high-level properties that define the convolution's interface,
/// including dimensionality, data layout, data types, and elementwise operations.
struct ConvSignatureInfo
{
    int spatial_dim;
    builder::ConvDirection direction;
    std::variant<builder::GroupConvLayout1D, builder::GroupConvLayout2D, builder::GroupConvLayout3D>
        layout;
    builder::DataType data_type;
    builder::ElementwiseOperation input_element_op;
    builder::ElementwiseOperation weight_element_op;
    builder::ElementwiseOperation output_element_op;
};

/// @brief Algorithm configuration for a convolution kernel
/// Contains low-level implementation details including thread block configuration,
/// tile dimensions, memory access patterns, and pipeline settings.
struct GemmAlgorithmInfo
{
    int thread_block_size;
    DataTileInfo tile_dims;
    WarpGemmParams warp_gemm;
    InputTileTransferInfo a_tile_transfer;
    InputTileTransferInfo b_tile_transfer;
    OutputTileTransferInfo c_tile_transfer;
    builder::PipelineVersion pipeline_version;
    builder::PipelineScheduler pipeline_scheduler;
    std::variant<builder::ConvFwdSpecialization,
                 builder::ConvBwdDataSpecialization,
                 builder::ConvBwdWeightSpecialization>
        conv_specialization;
    builder::GemmPadding padding;
};

/// @brief Provides human-readable descriptions of convolution kernel instances
/// Generates formatted text descriptions at various levels of detail for
/// understanding and documenting convolution kernel configurations.
struct ConvDescription
{
    ConvSignatureInfo signature;
    GemmAlgorithmInfo algorithm;

    /// @brief Generate a brief one-line summary of the convolution
    /// @return A concise description (e.g., "2D Forward convolution")
    std::string brief() const
    {
        std::ostringstream oss;
        oss << signature.spatial_dim << "D " << signature.direction << " convolution";
        return oss.str();
    }

    /// @brief Generate a detailed hierarchical description of the convolution
    /// @return A multi-line tree-formatted description covering signature and algorithm details
    std::string detailed() const
    {
        TreeFormatter f;
        f.writeLine(0, signature.spatial_dim, "D ", signature.direction, " Convolution Kernel");
        f.writeLine(1, "Signature");
        f.writeLine(2, "Tensor Type: ", signature.data_type);
        f.writeLine(2, "Memory Layout: ", signature.layout);
        f.writeLine(2, "Input elementwise operation: ", signature.input_element_op);
        f.writeLine(2, "Weights elementwise operation: ", signature.weight_element_op);
        f.writeLast(2, "Output elementwise operation: ", signature.output_element_op);

        f.writeLine(1, "Algorithm");
        // Compute Block section
        f.writeLine(2, "Thread block size: ", algorithm.thread_block_size);
        f.writeLine(2,
                    "Data tile size: ",
                    algorithm.tile_dims.m,
                    "×",
                    algorithm.tile_dims.n,
                    "×",
                    algorithm.tile_dims.k);
        f.writeLine(2, "Gemm padding: ", algorithm.padding);
        f.writeLine(2, "Convolution specialization: ", algorithm.conv_specialization);
        // Pipeline section
        f.writeLine(2, "Pipeline version: ", algorithm.pipeline_version);
        f.writeLine(2, "Pipeline scheduler: ", algorithm.pipeline_scheduler);
        f.writeLine(2, "Warp Gemm parameters: ");
        f.writeLine(
            3, "subtile size: ", algorithm.warp_gemm.gemm_m, "×", algorithm.warp_gemm.gemm_n);
        f.writeLast(3,
                    "Number of warp gemm iterations: ",
                    algorithm.warp_gemm.m_iter,
                    "×",
                    algorithm.warp_gemm.n_iter);

        // Memory Access section
        f.writeLine(2, "Memory access:");

        f.writeLine(3, "A Tile transfer: ");
        f.writeLine(4,
                    "Tile dimensions: ",
                    algorithm.a_tile_transfer.tile_dimensions.k0,
                    "×",
                    algorithm.a_tile_transfer.tile_dimensions.m_or_n,
                    "×",
                    algorithm.a_tile_transfer.tile_dimensions.k1,
                    "×");
        f.writeLine(
            4, "The innermost K subdimension size: ", algorithm.a_tile_transfer.transfer_params.k1);
        f.writeLine(4,
                    "Spatial thread distribution over the data tile: ",
                    algorithm.a_tile_transfer.transfer_params.thread_cluster_order[0],
                    "×",
                    algorithm.a_tile_transfer.transfer_params.thread_cluster_order[1],
                    "×",
                    algorithm.a_tile_transfer.transfer_params.thread_cluster_order[2]);
        f.writeLine(4,
                    "The order of accessing data tile axes: ",
                    algorithm.a_tile_transfer.transfer_params.src_access_order[0],
                    "×",
                    algorithm.a_tile_transfer.transfer_params.src_access_order[1],
                    "×",
                    algorithm.a_tile_transfer.transfer_params.src_access_order[2]);
        f.writeLine(4,
                    "Vectorized memory access axis index (with contiguous memory): ",
                    algorithm.a_tile_transfer.transfer_params.src_vector_dim);
        f.writeLine(4,
                    "Vector access (GMEM read) instruction size: ",
                    algorithm.a_tile_transfer.transfer_params.src_scalar_per_vector);
        f.writeLine(4,
                    "Vector access (LDS write) instruction size: ",
                    algorithm.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1);
        f.writeLast(4,
                    "LDS data layout padding (to prevent bank conflicts): ",
                    algorithm.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1);

        f.writeLine(3, "B Tile transfer: ");
        f.writeLine(4,
                    "Tile dimensions: ",
                    algorithm.b_tile_transfer.tile_dimensions.k0,
                    "×",
                    algorithm.b_tile_transfer.tile_dimensions.m_or_n,
                    "×",
                    algorithm.b_tile_transfer.tile_dimensions.k1,
                    "×");
        f.writeLine(
            4, "The innermost K subdimension size: ", algorithm.b_tile_transfer.transfer_params.k1);
        f.writeLine(4,
                    "Spatial thread distribution over the data tile: ",
                    algorithm.b_tile_transfer.transfer_params.thread_cluster_order[0],
                    "×",
                    algorithm.b_tile_transfer.transfer_params.thread_cluster_order[1],
                    "×",
                    algorithm.b_tile_transfer.transfer_params.thread_cluster_order[2]);
        f.writeLine(4,
                    "The order of accessing data tile axes: ",
                    algorithm.b_tile_transfer.transfer_params.src_access_order[0],
                    "×",
                    algorithm.b_tile_transfer.transfer_params.src_access_order[1],
                    "×",
                    algorithm.b_tile_transfer.transfer_params.src_access_order[2]);
        f.writeLine(4,
                    "Vectorized memory access axis index (with contiguous memory): ",
                    algorithm.b_tile_transfer.transfer_params.src_vector_dim);
        f.writeLine(4,
                    "Vector access (GMEM read) instruction size: ",
                    algorithm.b_tile_transfer.transfer_params.src_scalar_per_vector);
        f.writeLine(4,
                    "Vector access (LDS write) instruction size: ",
                    algorithm.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1);
        f.writeLast(4,
                    "LDS data layout padding (to prevent bank conflicts): ",
                    algorithm.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1);

        f.writeLast(3, "C Tile transfer: ");
        f.writeLine(4,
                    "Data shuffle (number of gemm instructions per iteration): ",
                    algorithm.c_tile_transfer.shuffle_params.m_gemms_per_shuffle,
                    "×",
                    algorithm.c_tile_transfer.shuffle_params.n_gemms_per_shuffle);
        f.writeLine(4,
                    "Spatial thread distribution used to store data: ",
                    algorithm.c_tile_transfer.thread_cluster_dims[0],
                    "×",
                    algorithm.c_tile_transfer.thread_cluster_dims[1],
                    "×",
                    algorithm.c_tile_transfer.thread_cluster_dims[2],
                    "×",
                    algorithm.c_tile_transfer.thread_cluster_dims[3]);
        f.writeLast(4,
                    "Vector access (GMEM write) instruction size: ",
                    algorithm.c_tile_transfer.scalar_per_vector);
        f.writeLast(2);
        f.writeLast(1);
        return f.getString();
    }

    /// @brief Generate an educational explanation of optimization choices
    /// @return Educational content explaining why certain algorithm choices were made
    /// @note Currently unimplemented - reserved for future enhancement
    std::string explain() const
    {
        std::ostringstream oss;
        // Placeholder for future implementation
        return oss.str();
    }

    /// @brief Generate performance characteristics and use case guidance
    /// @return Guidance on when this configuration is optimal and expected performance
    /// @note Currently unimplemented - reserved for future enhancement
    std::string suggest() const
    {
        std::ostringstream oss;
        // Placeholder for future implementation
        return oss.str();
    }
};

/// @brief Helper concept to detect if a type has InstanceTraits specialization
template <typename T>
concept HasInstanceTraits = requires { typename InstanceTraits<T>; };

/// @brief Factory function to create ConvDescription from a convolution instance type
/// @tparam Instance The convolution instance type (must have InstanceTraits specialization)
/// @return A ConvDescription object populated with the instance's configuration details
template <typename Instance>
    requires HasInstanceTraits<Instance>
ConvDescription Describe()
{
    using Traits = ConvTraits<Instance>;

    return ConvDescription{
        .signature = ConvSignatureInfo{.spatial_dim       = Traits::spatial_dim,
                                       .direction         = Traits::direction,
                                       .layout            = Traits::layout,
                                       .data_type         = Traits::data_type,
                                       .input_element_op  = Traits::input_element_op,
                                       .weight_element_op = Traits::weight_element_op,
                                       .output_element_op = Traits::output_element_op},
        .algorithm = GemmAlgorithmInfo{.thread_block_size   = Traits::thread_block_size,
                                       .tile_dims           = Traits::tile_dims,
                                       .warp_gemm           = Traits::warp_gemm,
                                       .a_tile_transfer     = Traits::a_tile_transfer,
                                       .b_tile_transfer     = Traits::b_tile_transfer,
                                       .c_tile_transfer     = Traits::c_tile_transfer,
                                       .pipeline_version    = Traits::pipeline_version,
                                       .pipeline_scheduler  = Traits::pipeline_scheduler,
                                       .conv_specialization = Traits::conv_specialization,
                                       .padding             = Traits::gemm_padding}};
}

} // namespace ck_tile::reflect::conv
