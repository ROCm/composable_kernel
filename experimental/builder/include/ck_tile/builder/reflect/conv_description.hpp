// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file
/// @brief Provides utilities to reflect on convolution kernel instances and generate
/// human-readable descriptions of their configuration.
///
/// This file contains the necessary components to transform a convolution kernel's
/// compile-time properties into a structured, descriptive format. This is primarily
/// used for debugging, logging, and generating documentation.
///
/// Key components:
/// - ck_tile::reflect::conv::ConvDescription: A struct that holds the extracted
///   properties and provides methods to format them into strings.
/// - ck_tile::reflect::conv::Describe(): A factory function that creates a
///   ConvDescription from a given kernel instance type.

#pragma once

#include <concepts>
#include <string_view>
#include <sstream>
#include <type_traits>
#include <variant>
#include <functional>

#include <ck_tile/builder/conv_signature_concepts.hpp>
#include <ck_tile/builder/reflect/conv_types.hpp>
#include <ck_tile/builder/reflect/description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/reflect/tree_formatter.hpp>
#include <ck_tile/builder/reflect/conv_traits.hpp>

namespace ck_tile::reflect {

namespace conv {

/// @brief Provides human-readable descriptions of convolution kernel instances
/// Generates formatted text descriptions at various levels of detail for
/// understanding and documenting convolution kernel configurations.
class ConvDescription : public Description
{
    public:
    /// @brief Constructor for ConvDescription
    /// @param traits The ConvTraits object containing all relevant signature and algorithm
    /// information
    /// @param instance_string_getter A callable that returns a string representation of the
    /// instance
    ConvDescription(ConvTraits traits, std::function<std::string()> instance_string_getter)
        : traits_(std::move(traits)), instance_string_getter_(std::move(instance_string_getter))
    {
    }

    /// @brief Generate a brief one-line summary of the convolution
    /// @return A concise description (e.g., "2D Forward convolution")
    std::string brief() const override
    {
        std::ostringstream oss;
        oss << traits_.spatial_dim << "D " << traits_.direction << " convolution";
        return oss.str();
    }

    /// @brief Generate a detailed hierarchical description of the convolution
    /// @return A multi-line tree-formatted description covering signature and algorithm details
    std::string detailed() const override
    {
        TreeFormatter root(traits_.spatial_dim, "D ", traits_.direction, " Convolution Kernel");

        auto& sig = root.add("Signature");
        sig.add("Tensor Type: ", traits_.data_type);
        sig.add("Input Layout: ", traits_.layout[0]);
        sig.add("Weight Layout: ", traits_.layout[1]);
        sig.add("Output Layout: ", traits_.layout[2]);
        sig.add("Input elementwise operation: ", traits_.input_element_op);
        sig.add("Weights elementwise operation: ", traits_.weight_element_op);
        sig.add("Output elementwise operation: ", traits_.output_element_op);

        auto& algo = root.add("Algorithm");
        // Compute Block section
        algo.add("Thread block size: ", traits_.thread_block_size);
        algo.add("Data tile size: ",
                 traits_.tile_dims.m,
                 "×",
                 traits_.tile_dims.n,
                 "×",
                 traits_.tile_dims.k);
        if(traits_.gemm_padding)
            algo.add("Gemm padding: ", *traits_.gemm_padding);
        if(traits_.do_pad_gemm_m)
            algo.add("Do Pad Gemm M: ", *traits_.do_pad_gemm_m);
        if(traits_.do_pad_gemm_n)
            algo.add("Do Pad Gemm N: ", *traits_.do_pad_gemm_n);
        algo.add("Convolution specialization: ", traits_.conv_specialization);
        // Pipeline section
        algo.add("Pipeline version: ", traits_.pipeline_version);
        algo.add("Pipeline scheduler: ", traits_.pipeline_scheduler);
        auto& warpGemm = algo.add("Warp Gemm parameters:");
        warpGemm.add("subtile size: ", traits_.warp_gemm.gemm_m, "×", traits_.warp_gemm.gemm_n);
        warpGemm.add("Number of warp gemm iterations: ",
                     traits_.warp_gemm.m_iter,
                     "×",
                     traits_.warp_gemm.n_iter);

        // Memory Access section
        auto& memAccess = algo.add("Memory access:");

        auto& aTile = memAccess.add("A Tile transfer:");
        aTile.add("Tile dimensions: ",
                  traits_.a_tile_transfer.tile_dimensions.k0,
                  "×",
                  traits_.a_tile_transfer.tile_dimensions.m_or_n,
                  "×",
                  traits_.a_tile_transfer.tile_dimensions.k1);
        aTile.add("The innermost K subdimension size: ",
                  traits_.a_tile_transfer.transfer_params.k1);
        aTile.add("Thread cluster lengths (threads per axis): ",
                  traits_.a_tile_transfer.transfer_params.thread_cluster_dims[0],
                  "×",
                  traits_.a_tile_transfer.transfer_params.thread_cluster_dims[1],
                  "×",
                  traits_.a_tile_transfer.transfer_params.thread_cluster_dims[2]);
        aTile.add("Spatial thread distribution over the data tile: ",
                  traits_.a_tile_transfer.transfer_params.thread_cluster_order[0],
                  "×",
                  traits_.a_tile_transfer.transfer_params.thread_cluster_order[1],
                  "×",
                  traits_.a_tile_transfer.transfer_params.thread_cluster_order[2]);
        aTile.add("The order of accessing data tile axes: ",
                  traits_.a_tile_transfer.transfer_params.src_access_order[0],
                  "×",
                  traits_.a_tile_transfer.transfer_params.src_access_order[1],
                  "×",
                  traits_.a_tile_transfer.transfer_params.src_access_order[2]);
        aTile.add("Vectorized memory access axis index (with contiguous memory): ",
                  traits_.a_tile_transfer.transfer_params.src_vector_dim);
        aTile.add("Vector access (GMEM read) instruction size: ",
                  traits_.a_tile_transfer.transfer_params.src_scalar_per_vector);
        aTile.add("Vector access (LDS write) instruction size: ",
                  traits_.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1);
        aTile.add("LDS data layout padding (to prevent bank conflicts): ",
                  traits_.a_tile_transfer.transfer_params.lds_padding);

        auto& bTile = memAccess.add("B Tile transfer:");
        bTile.add("Tile dimensions: ",
                  traits_.b_tile_transfer.tile_dimensions.k0,
                  "×",
                  traits_.b_tile_transfer.tile_dimensions.m_or_n,
                  "×",
                  traits_.b_tile_transfer.tile_dimensions.k1);
        bTile.add("The innermost K subdimension size: ",
                  traits_.b_tile_transfer.transfer_params.k1);
        bTile.add("Thread cluster lengths (threads per axis): ",
                  traits_.b_tile_transfer.transfer_params.thread_cluster_dims[0],
                  "×",
                  traits_.b_tile_transfer.transfer_params.thread_cluster_dims[1],
                  "×",
                  traits_.b_tile_transfer.transfer_params.thread_cluster_dims[2]);
        bTile.add("Spatial thread distribution over the data tile: ",
                  traits_.b_tile_transfer.transfer_params.thread_cluster_order[0],
                  "×",
                  traits_.b_tile_transfer.transfer_params.thread_cluster_order[1],
                  "×",
                  traits_.b_tile_transfer.transfer_params.thread_cluster_order[2]);
        bTile.add("The order of accessing data tile axes: ",
                  traits_.b_tile_transfer.transfer_params.src_access_order[0],
                  "×",
                  traits_.b_tile_transfer.transfer_params.src_access_order[1],
                  "×",
                  traits_.b_tile_transfer.transfer_params.src_access_order[2]);
        bTile.add("Vectorized memory access axis index (with contiguous memory): ",
                  traits_.b_tile_transfer.transfer_params.src_vector_dim);
        bTile.add("Vector access (GMEM read) instruction size: ",
                  traits_.b_tile_transfer.transfer_params.src_scalar_per_vector);
        bTile.add("Vector access (LDS write) instruction size: ",
                  traits_.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1);
        bTile.add("LDS data layout padding (to prevent bank conflicts): ",
                  traits_.b_tile_transfer.transfer_params.lds_padding);

        auto& cTile = memAccess.add("C Tile transfer:");
        cTile.add("Data shuffle (number of gemm instructions per iteration): ",
                  traits_.c_tile_transfer.shuffle_params.m_gemms_per_shuffle,
                  "×",
                  traits_.c_tile_transfer.shuffle_params.n_gemms_per_shuffle);
        cTile.add("Spatial thread distribution used to store data: ",
                  traits_.c_tile_transfer.thread_cluster_dims[0],
                  "×",
                  traits_.c_tile_transfer.thread_cluster_dims[1],
                  "×",
                  traits_.c_tile_transfer.thread_cluster_dims[2],
                  "×",
                  traits_.c_tile_transfer.thread_cluster_dims[3]);
        cTile.add("Vector access (GMEM write) instruction size: ",
                  traits_.c_tile_transfer.scalar_per_vector);
        if(traits_.num_gemm_k_prefetch_stage)
            algo.add("Num gemm k prefetch stage: ", *traits_.num_gemm_k_prefetch_stage);
        if(traits_.max_transpose_transfer_src_scalar_per_vector)
            algo.add("Max Transpose transfer src scalar per vector: ",
                     *traits_.max_transpose_transfer_src_scalar_per_vector);
        if(traits_.max_transpose_transfer_dst_scalar_per_vector)
            algo.add("Max Transpose dst scalar per vector: ",
                     *traits_.max_transpose_transfer_dst_scalar_per_vector);
        if(traits_.num_groups_to_merge)
            algo.add("Num groups to merge: ", *traits_.num_groups_to_merge);

        return root.getString();
    }

    /// @brief Generate a string representation of the instance
    /// @return A string that represents the instance
    std::string instance_string() const override { return instance_string_getter_(); }

    private:
    ConvTraits traits_;
    std::function<std::string()> instance_string_getter_;
};

} // namespace conv

} // namespace ck_tile::reflect
