// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
/// @brief

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
        oss << signature.spatial_dim << "D " << builder::ConvDirectionToString(signature.direction)
            << " convolution";
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

        f.writeLast(1, "Algorithm");
        // Compute Block section
        f.writeLine(2, "Thread block size: ", algorithm.thread_block_size);
        f.writeLine(2,
                    "Data tile size: " algorithm.tile_dims.m,
                    "×",
                    algorithm.tile_dims.n,
                    "×",
                    algorithm.tile_dims.k);
        f.writeLine(2, "Warp Gemm parameters: "),
            f.writeLine(
                3, "subtile size: ", algorithm.warp_gemm.gemm_m, "×", algorithm.warp_gemm.gemm_n);
        f.writeLine(3,
                    "Number of warp gemm iterations: ",
                    algorithm.warp_gemm.m_iter,
                    "×",
                    algorithm.warp_gemm.n_iter, );

        // Memory Layout section
        f.writeLine(2, "Memory Layout:");
        f.writeLine(3, "A-Transfer: ", );
        f.writeLine(3, "B-Transfer: ", );
        f.writeLast(3, "C-Transfer: ", );

        // Pipeline section
        f.writeLine(2, "Pipeline version: ", algorithm.pipeline_version);
        f.writeLast(2, "Pipeline scheduler: ", algorithm.pipeline_scheduler);

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
        .signature = ConvSignatureInfo{.spatial_dim = Traits::spatial_dim,
                                       .direction   = Traits::direction,
                                       .layout      = Traits::layout,
                                       .data_type   = Traits::data_type},
        .algorithm = GemmAlgorithmInfo{
            .block  = DataTileInfo{.block_size = Traits::block.block_size,
                                   .m          = Traits::block.per_block.m,
                                   .n          = Traits::block.per_block.n,
                                   .k          = Traits::block.per_block.k},
            .tuning = TuningInfo{.ak1            = Traits::tuning.ak1,
                                 .bk1            = Traits::tuning.bk1,
                                 .m_per_xdl      = Traits::tuning.m_per_xdl,
                                 .n_per_xdl      = Traits::tuning.n_per_dxl,
                                 .m_xdl_per_wave = Traits::tuning.m_xdl_per_wave,
                                 .n_xdl_per_wave = Traits::tuning.n_xdl_per_wave},
            .a_transfer =
                TileTransferInfo{.k0     = Traits::a_block_transfer.thread_cluster_dims[0],
                                 .m_or_n = Traits::a_block_transfer.thread_cluster_dims[1],
                                 .k1     = Traits::a_block_transfer.thread_cluster_dims[2]},
            .b_transfer =
                TileTransferInfo{.k0     = Traits::b_block_transfer.thread_cluster_dims[0],
                                 .m_or_n = Traits::b_block_transfer.thread_cluster_dims[1],
                                 .k1     = Traits::b_block_transfer.thread_cluster_dims[2]},
            .c_transfer =
                CBlockTransferInfo{
                    .m_xdl_per_wave_per_shuffle =
                        Traits::c_block_transfer.m_xdl_per_wave_per_shuffle,
                    .n_xdl_per_wave_per_shuffle =
                        Traits::c_block_transfer.n_xdl_per_wave_per_shuffle,
                    .m_block        = Traits::c_block_transfer.thread_cluster_dims[0],
                    .m_wave_per_xdl = Traits::c_block_transfer.thread_cluster_dims[1],
                    .n_block        = Traits::c_block_transfer.thread_cluster_dims[2],
                    .n_wave_per_xdl = Traits::c_block_transfer.thread_cluster_dims[3]},
            .pipeline = Traits::pipeline_version}};
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
