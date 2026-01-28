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
        TreeFormatter f;
        f.writeLine(0, traits_.spatial_dim, "D ", traits_.direction, " Convolution Kernel");
        f.writeLine(1, "Signature");
        f.writeLine(2, "Tensor Type: ", traits_.data_type);
        f.writeLine(2, "Input Layout: ", traits_.layout[0]);
        f.writeLine(2, "Weight Layout: ", traits_.layout[1]);
        f.writeLine(2, "Output Layout: ", traits_.layout[2]);
        f.writeLine(2, "Input elementwise operation: ", traits_.input_element_op);
        f.writeLine(2, "Weights elementwise operation: ", traits_.weight_element_op);
        f.writeLast(2, "Output elementwise operation: ", traits_.output_element_op);

        f.writeLast(1, "Algorithm");
        // Compute Block section
        f.writeLine(2, "Thread block size: ", traits_.thread_block_size);
        f.writeLine(2,
                    "Data tile size: ",
                    traits_.tile_dims.m,
                    "×",
                    traits_.tile_dims.n,
                    "×",
                    traits_.tile_dims.k);
        if(traits_.gemm_padding)
            f.writeLine(
                2, "Gemm padding: ", traits_.gemm_padding.value_or(builder::GemmPadding::DEFAULT));
        else
            f.writeLine(2, "Struct does not contain optional gemm_padding argument");
        f.writeLine(2, "Convolution specialization: ", traits_.conv_specialization);
        // Pipeline section
        f.writeLine(2, "Pipeline version: ", traits_.pipeline_version);
        f.writeLine(2, "Pipeline scheduler: ", traits_.pipeline_scheduler);
        f.writeLine(2, "Warp Gemm parameters: ");
        f.writeLine(3, "subtile size: ", traits_.warp_gemm.gemm_m, "×", traits_.warp_gemm.gemm_n);
        f.writeLast(3,
                    "Number of warp gemm iterations: ",
                    traits_.warp_gemm.m_iter,
                    "×",
                    traits_.warp_gemm.n_iter);

        // Memory Access section
        f.writeLast(2, "Memory access:");

        f.writeLine(3, "A Tile transfer: ");
        f.writeLine(4,
                    "Tile dimensions: ",
                    traits_.a_tile_transfer.tile_dimensions.k0,
                    "×",
                    traits_.a_tile_transfer.tile_dimensions.m_or_n,
                    "×",
                    traits_.a_tile_transfer.tile_dimensions.k1,
                    "×");
        f.writeLine(
            4, "The innermost K subdimension size: ", traits_.a_tile_transfer.transfer_params.k1);
        f.writeLine(4,
                    "Spatial thread distribution over the data tile: ",
                    traits_.a_tile_transfer.transfer_params.thread_cluster_order[0],
                    "×",
                    traits_.a_tile_transfer.transfer_params.thread_cluster_order[1],
                    "×",
                    traits_.a_tile_transfer.transfer_params.thread_cluster_order[2]);
        f.writeLine(4,
                    "The order of accessing data tile axes: ",
                    traits_.a_tile_transfer.transfer_params.src_access_order[0],
                    "×",
                    traits_.a_tile_transfer.transfer_params.src_access_order[1],
                    "×",
                    traits_.a_tile_transfer.transfer_params.src_access_order[2]);
        f.writeLine(4,
                    "Vectorized memory access axis index (with contiguous memory): ",
                    traits_.a_tile_transfer.transfer_params.src_vector_dim);
        f.writeLine(4,
                    "Vector access (GMEM read) instruction size: ",
                    traits_.a_tile_transfer.transfer_params.src_scalar_per_vector);
        f.writeLine(4,
                    "Vector access (LDS write) instruction size: ",
                    traits_.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1);
        f.writeLast(4,
                    "LDS data layout padding (to prevent bank conflicts): ",
                    traits_.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1);

        f.writeLine(3, "B Tile transfer: ");
        f.writeLine(4,
                    "Tile dimensions: ",
                    traits_.b_tile_transfer.tile_dimensions.k0,
                    "×",
                    traits_.b_tile_transfer.tile_dimensions.m_or_n,
                    "×",
                    traits_.b_tile_transfer.tile_dimensions.k1,
                    "×");
        f.writeLine(
            4, "The innermost K subdimension size: ", traits_.b_tile_transfer.transfer_params.k1);
        f.writeLine(4,
                    "Spatial thread distribution over the data tile: ",
                    traits_.b_tile_transfer.transfer_params.thread_cluster_order[0],
                    "×",
                    traits_.b_tile_transfer.transfer_params.thread_cluster_order[1],
                    "×",
                    traits_.b_tile_transfer.transfer_params.thread_cluster_order[2]);
        f.writeLine(4,
                    "The order of accessing data tile axes: ",
                    traits_.b_tile_transfer.transfer_params.src_access_order[0],
                    "×",
                    traits_.b_tile_transfer.transfer_params.src_access_order[1],
                    "×",
                    traits_.b_tile_transfer.transfer_params.src_access_order[2]);
        f.writeLine(4,
                    "Vectorized memory access axis index (with contiguous memory): ",
                    traits_.b_tile_transfer.transfer_params.src_vector_dim);
        f.writeLine(4,
                    "Vector access (GMEM read) instruction size: ",
                    traits_.b_tile_transfer.transfer_params.src_scalar_per_vector);
        f.writeLine(4,
                    "Vector access (LDS write) instruction size: ",
                    traits_.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1);
        f.writeLast(4,
                    "LDS data layout padding (to prevent bank conflicts): ",
                    traits_.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1);

        f.writeLast(3, "C Tile transfer: ");
        f.writeLine(4,
                    "Data shuffle (number of gemm instructions per iteration): ",
                    traits_.c_tile_transfer.shuffle_params.m_gemms_per_shuffle,
                    "×",
                    traits_.c_tile_transfer.shuffle_params.n_gemms_per_shuffle);
        f.writeLine(4,
                    "Spatial thread distribution used to store data: ",
                    traits_.c_tile_transfer.thread_cluster_dims[0],
                    "×",
                    traits_.c_tile_transfer.thread_cluster_dims[1],
                    "×",
                    traits_.c_tile_transfer.thread_cluster_dims[2],
                    "×",
                    traits_.c_tile_transfer.thread_cluster_dims[3]);
        f.writeLine(4,
                    "Vector access (GMEM write) instruction size: ",
                    traits_.c_tile_transfer.scalar_per_vector);
        if(traits_.num_gemm_k_prefetch_stage)
            f.writeLine(
                2, "Num gemm k prefetch stage: ", traits_.num_gemm_k_prefetch_stage.value_or(0));
        else
            f.writeLine(2,
                        "Struct does not contain optional "
                        "num_gemm_k_prefetch_stage parameter");

        if(traits_.max_transpose_transfer_src_scalar_per_vector)
            f.writeLine(2,
                        "Max Transpose transfer scr scalar per vector: ",
                        traits_.max_transpose_transfer_src_scalar_per_vector.value_or(0));
        else
            f.writeLine(2,
                        "Struct does not contain optional "
                        "max_transpose_transfer_src_scalar_per_vector parameter");
        if(traits_.max_transpose_dst_scalar_per_vector)
            f.writeLine(2,
                        "Max Transpose dst scalar per vector: ",
                        traits_.max_transpose_dst_scalar_per_vector.value_or(0));
        else
            f.writeLine(
                2,
                "Struct does not contain optional max_transpose_dst_scalar_per_vector parameter");
        if(traits_.num_groups_to_merge)
            f.writeLast(2, "Num groups to merge: ", traits_.num_groups_to_merge.value_or(0));
        else
            f.writeLast(2, "Struct does not contain optional num_groups_to_merge parameter");

        return f.getString();
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
