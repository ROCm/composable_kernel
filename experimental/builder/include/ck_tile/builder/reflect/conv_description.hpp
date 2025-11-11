// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <concepts>
#include <string_view>
#include <sstream>
#include <type_traits>
#include <variant>

#include <ck_tile/builder/conv_signature_concepts.hpp>
#include <ck_tile/builder/reflect/conv_traits.hpp>
#include <ck_tile/builder/reflect/tree_formatter.hpp>

/// @file conv_description.hpp
/// @brief Provides human-readable descriptions of ConvBuilder configurations

namespace ck_tile::reflect::conv {

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

// Algorithm information - groups all algorithm-related configuration
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

// Provides human-readable descriptions of ConvBuilder configurations.
struct ConvDescription
{
    ConvSignatureInfo signature;
    GemmAlgorithmInfo algorithm;

    // Brief one-line summary
    std::string brief() const
    {
        std::ostringstream oss;
        oss << signature.spatial_dim << "D " << signature.direction << " convolution";
        return oss.str();
    }

    // Detailed hierarchical description
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

    // Educational explanation of optimization choices
    std::string explain() const
    {
        std::ostringstream oss;
        // Placeholder for future implementation
        return oss.str();
    }

    // Performance characteristics and use case guidance
    std::string suggest() const
    {
        std::ostringstream oss;
        // Placeholder for future implementation
        return oss.str();
    }
};

// Helper concept to detect if a type has InstanceTraits specialization
template <typename T>
concept HasInstanceTraits = requires { typename InstanceTraits<T>; };

// Helper concept to detect ConvBuilder types
template <typename T>
concept IsConvBuilder = requires {
    typename T::Factory;
    typename T::Instance;
};

// Primary factory function: Create ConvDescription from Instance type directly
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

// Backward compatibility: Create ConvDescription from Builder type
template <typename Builder>
    requires IsConvBuilder<Builder> && (!HasInstanceTraits<Builder>)
ConvDescription Describe()
{
    // Delegate to Instance-based version
    using Instance = typename Builder::Instance;
    return Describe<Instance>();
}

} // namespace ck_tile::reflect::conv
