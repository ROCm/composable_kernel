// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include <concepts>
#include <array>

#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder 
{

/********************************************************************/
/* Descriptors for individual elements of the algorithm description */
/********************************************************************/

// Concept for thread block dimensions for a GEMM problem.
template <typename T>
concept ThreadBlockDescriptor = requires(T t) {
  { t.block_size }  -> std::convertible_to<size_t>;
  { t.tile_size.m } -> std::convertible_to<size_t>;
  { t.tile_size.n } -> std::convertible_to<size_t>;
  { t.tile_size.k } -> std::convertible_to<size_t>;
};

// Concept for parameters that describe a gridwise GEMM problem.
template <typename T>
concept GridwiseGemmDescriptor = requires(T t) {
    { t.ak1 } -> std::convertible_to<size_t>;
    { t.bk1 } -> std::convertible_to<size_t>;
    { t.m_per_xdl } -> std::convertible_to<size_t>;
    { t.n_per_xdl } -> std::convertible_to<size_t>;
    { t.m_xdl_per_wave } -> std::convertible_to<size_t>;
    { t.n_xdl_per_wave } -> std::convertible_to<size_t>;
};

// Concept for convolution input block transfer.
template <typename T>
concept InputBlockTransferDescriptor = requires(T t) {
    { t.k0 } -> std::convertible_to<size_t>;
    { t.m_n } -> std::convertible_to<size_t>;
    { t.k1 } -> std::convertible_to<size_t>;
};

// Concept for output block transfer.
template <typename T>
concept OutputBlockTransferDescriptor = requires(T t) {
    { t.m_block } -> std::convertible_to<size_t>;
    { t.m_wave_per_xdl } -> std::convertible_to<size_t>;
    { t.n_block } -> std::convertible_to<size_t>;
    { t.n_wave_per_xdl } -> std::convertible_to<size_t>;
};

// Concept for the convolution input vector transfer.
template <typename T>
concept InputVectorTransferDescriptor = requires(T t) {
    { t.src_vector_dim } -> std::convertible_to<size_t>;
    { t.src_scalar_per_vector } -> std::convertible_to<size_t>;
    { t.dest_scalar_per_vector_k1 } -> std::convertible_to<size_t>;
    { t.add_extra } -> std::convertible_to<bool>;
};

// Concepts for the convolution output vector transfer.
template <typename T>
concept OutputVectorTransferDescriptor = requires(T t) {
    { t.m_xdl_per_wave_per_shuffle } -> std::convertible_to<size_t>;
    { t.n_xdl_per_wave_per_shuffle } -> std::convertible_to<size_t>;
    { t.scalar_per_vector } -> std::convertible_to<size_t>; 
}; 

// Concept for the thread cluster access order
template <typename T>
concept AccessOrderDescriptor = requires(T t) {
    { t.order } -> std::convertible_to<std::array<size_t, 3>>;
};

// No requirements yet for a ConvAlogorithm concept.
template <typename T>
concept ConvAlgorithmDescriptor = std::is_class_v<T>;

/******************************************** */
/* Requirements for the algorithm description */
/******************************************** */

// Concept to check if struct specifies thread block info.
template <typename T>
concept SpecifiesThreadBlock = requires {
    { T::thread_block } -> ThreadBlockDescriptor;
};

// Concept to check if a struct specifies gridwise GEMM info.
template <typename T>
concept SpecifiesGridwiseGemm = requires {
    { T::tuning_params } -> GridwiseGemmDescriptor;
};

// Concept to check if a struct specifies convolution input and output block transfer info.
template <typename T>
concept SpecifiesBlockTransfer = requires(T t) {
    { T::block_transfer.thread_cluster_dims_a } -> InputBlockTransferDescriptor;
    { T::block_transfer.thread_cluster_dims_b } -> InputBlockTransferDescriptor;
    { T::block_transfer.thread_cluster_dims_c } -> OutputBlockTransferDescriptor;
};

// Concept to check if a struct specifies block vector transfer info.
template <typename T>
concept SpecifiesBlockVectorTransfer = requires(T t) {
    { T::block_transfer.vector_transfer_a } -> InputVectorTransferDescriptor;
    { T::block_transfer.vector_transfer_b } -> InputVectorTransferDescriptor;
    { T::block_transfer.vector_transfer_c } -> OutputVectorTransferDescriptor;
};

// Concept to check if a struct specifies thread cluster access order info.
template <typename T>
concept SpecifiesThreadClusterAccessOrder = requires(T t) {
    { T::block_transfer.thread_cluster_access_order_a } -> AccessOrderDescriptor;
    { T::block_transfer.thread_cluster_access_order_b } -> AccessOrderDescriptor;
};

// Concept to check if a struct specifies source access order info.
template <typename T>
concept SpecifiesSourceAccessOrder = requires(T t) {
    { T::block_transfer.src_access_order_a } -> AccessOrderDescriptor;
    { T::block_transfer.src_access_order_b } -> AccessOrderDescriptor;
};

// Concept to check if struct specifies block_gemm_pipeline_version.
template <typename T>
concept SpecifiesGemmPipelineVersion = requires {
    { T::pipeline_version } -> std::convertible_to<BlockGemmPipelineVersion>;
};

} // namespace ck_tile::builder
