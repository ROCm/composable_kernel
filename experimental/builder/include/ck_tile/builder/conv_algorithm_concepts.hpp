// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <concepts>
#include <array>

#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {

/********************************************************************/
/* Descriptors for individual elements of the algorithm description */
/********************************************************************/

// Concept for thread block dimensions for a GEMM problem.
template <typename T>
concept ThreadBlockDescriptor = requires(T t) {
    { t.block_size } -> std::convertible_to<size_t>;
    { t.tile_size.m } -> std::convertible_to<size_t>;
    { t.tile_size.n } -> std::convertible_to<size_t>;
    { t.tile_size.k } -> std::convertible_to<size_t>;
};

// Concept for parameters that describe a gridwise XDL GEMM problem.
template <typename T>
concept GridwiseXdlGemmDescriptor = requires(T t) {
    { t.ak1 } -> std::convertible_to<size_t>;
    { t.bk1 } -> std::convertible_to<size_t>;
    { t.m_per_xdl } -> std::convertible_to<size_t>;
    { t.n_per_xdl } -> std::convertible_to<size_t>;
    { t.m_xdl_per_wave } -> std::convertible_to<size_t>;
    { t.n_xdl_per_wave } -> std::convertible_to<size_t>;
};

// Concept for parameter that describe block GEMM problem.
template <typename T>
concept BlockGemmDescriptor = requires(T t) {
    { t.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { t.scheduler } -> std::convertible_to<PipelineScheduler>;
};

// Concept for parameters that describe a gridwise WMMA GEMM problem.
template <typename T>
concept GridwiseWmmaGemmDescriptor = requires(T t) {
    { t.k1 } -> std::convertible_to<size_t>;
    { t.m_per_wmma } -> std::convertible_to<size_t>;
    { t.n_per_wmma } -> std::convertible_to<size_t>;
    { t.m_wmma_per_wave } -> std::convertible_to<size_t>;
    { t.n_wmma_per_wave } -> std::convertible_to<size_t>;
    { t.pipeline_version } -> std::convertible_to<PipelineVersion>;
};

// Concept for vectorized data transfer for convolution input tensors.
template <typename T>
concept BlockTransferDescriptor = requires(T t) {
    { t.k0 } -> std::convertible_to<size_t>;
    { t.m_n } -> std::convertible_to<size_t>;
    { t.k1 } -> std::convertible_to<size_t>;
};

// Concept for thread cluster dimensions for GEMM output tensor.
template <typename T>
concept ThreadClusterDescriptor = requires(T t) {
    { t.m_block } -> std::convertible_to<size_t>;
    { t.m_wave_per_xdl } -> std::convertible_to<size_t>;
    { t.n_block } -> std::convertible_to<size_t>;
    { t.n_wave_per_xdl } -> std::convertible_to<size_t>;
};

// Concept for the LDS transfer for the convolution input tensors.
template <typename T>
concept LdsTransferDescriptor = requires(T t) {
    { t.src_vector_dim } -> std::convertible_to<size_t>;
    { t.src_scalar_per_vector } -> std::convertible_to<size_t>;
    { t.lds_dst_scalar_per_vector } -> std::convertible_to<size_t>;
    { t.is_direct_load } -> std::convertible_to<bool>;
    { t.lds_padding } -> std::convertible_to<bool>;
};

// Concept for the convolution output tensor epilogue (copy from registers to global memory via
// LDS).
template <typename T>
concept EpilogueDescriptor = requires(T t) {
    { t.m_per_wave_per_shuffle } -> std::convertible_to<size_t>;
    { t.n_per_wave_per_shuffle } -> std::convertible_to<size_t>;
    { t.scalar_per_vector } -> std::convertible_to<size_t>;
};

// Concept for the thread cluster access order
template <typename T>
concept AccessOrderDescriptor = requires(T t) {
    { t.order } -> std::convertible_to<std::array<size_t, 3>>;
};

// No requirements yet for a ConvAlgorithm concept.
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

// Concept to check if a struct specifies gridwise XDL GEMM info.
template <typename T>
concept SpecifiesGridwiseXdlGemm = requires {
    { T::gridwise_gemm } -> GridwiseXdlGemmDescriptor;
};

// Concept to check if a struct specifies gridwise WMMA GEMM info.
template <typename T>
concept SpecifiesGridwiseWmmaGemm = requires {
    { T::gridwise_gemm } -> GridwiseWmmaGemmDescriptor;
};

// Concept to check if a struct specifies convolution input and output block transfer info.
template <typename T>
concept SpecifiesBlockTransfer = requires(T t) {
    { T::block_transfer.block_transfer_a } -> BlockTransferDescriptor;
    { T::block_transfer.block_transfer_b } -> BlockTransferDescriptor;
    { T::block_transfer.thread_cluster_dims_c } -> ThreadClusterDescriptor;
};

// Concept to check if a struct specifies LDS transfer info for tensors A, B, and C.
template <typename T>
concept SpecifiesLdsTransfer = requires(T t) {
    { T::block_transfer.lds_transfer_a } -> LdsTransferDescriptor;
    { T::block_transfer.lds_transfer_b } -> LdsTransferDescriptor;
    { T::block_transfer.epilogue_c } -> EpilogueDescriptor;
};

// Concept to check if a struct specifies thread cluster access order info.
template <typename T>
concept SpecifiesThreadClusterAccessOrder = requires(T t) {
    { T::block_transfer.block_transfer_access_order_a } -> AccessOrderDescriptor;
    { T::block_transfer.block_transfer_access_order_b } -> AccessOrderDescriptor;
};

// Concept to check if a struct specifies source access order info.
template <typename T>
concept SpecifiesSourceAccessOrder = requires(T t) {
    { T::block_transfer.src_access_order_a } -> AccessOrderDescriptor;
    { T::block_transfer.src_access_order_b } -> AccessOrderDescriptor;
};

// Concept to check if struct specifies block GEMM.
template <typename T>
concept SpecifiesBlockGemm = requires {
    { T::block_gemm.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { T::block_gemm.scheduler } -> std::convertible_to<PipelineScheduler>;
};

template <typename T>
concept SpecifiesFwdConcSpecialization = requires {
    { T::fwd_specialization } -> std::convertible_to<ConvFwdSpecialization>;
};

template <typename T>
concept SpecifiesGemmSpecialization = requires {
    { T::gemm_specialization } -> std::convertible_to<GemmSpecialization>;
};

template <typename T>
concept SpecifiesNumPrefetchStages = requires {
    { T::num_gemm_k_prefetch_stages } -> std::convertible_to<size_t>;
};

template <typename T>
concept SpecifiesNumGroupsToMerge = requires {
    { T::num_groups_to_merge } -> std::convertible_to<size_t>;
};

template <typename T>
concept SpecifiesLoopScheduler = requires {
    { T::loop_scheduler } -> std::convertible_to<PipelineScheduler>;
};

/******************************************** */
/* DL-specific descriptors and requirements   */
/******************************************** */

// Concept for DL thread configuration
template <typename T>
concept DlThreadConfigDescriptor = requires(T t) {
    { t.k0_per_block } -> std::convertible_to<size_t>;
    { t.k1 } -> std::convertible_to<size_t>;
    { t.m1_per_thread } -> std::convertible_to<size_t>;
    { t.n1_per_thread } -> std::convertible_to<size_t>;
    { t.k_per_thread } -> std::convertible_to<size_t>;
};

// Concept for DL thread cluster
template <typename T>
concept DlThreadClusterDescriptor = requires(T t) {
    { t.m1_xs } -> std::convertible_to<std::array<size_t, 2>>;
    { t.n1_xs } -> std::convertible_to<std::array<size_t, 2>>;
};

// Concept for DL block transfer K0_M0_M1_K1 format
template <typename T>
concept DlBlockTransferK0M0M1K1Descriptor = requires(T t) {
    { t.thread_slice_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.thread_cluster_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.thread_cluster_arrange_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_access_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_vector_tensor_contiguous_dim_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.dst_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>;
};

// Concept for DL block transfer K0_N0_N1_K1 format
template <typename T>
concept DlBlockTransferK0N0N1K1Descriptor = requires(T t) {
    { t.thread_slice_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.thread_cluster_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.thread_cluster_arrange_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_access_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_vector_tensor_contiguous_dim_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.dst_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>;
};

// Concept for DL C thread transfer
template <typename T>
concept DlCThreadTransferDescriptor = requires(T t) {
    { t.src_dst_access_order } -> std::convertible_to<std::array<size_t, 6>>;
    { t.src_dst_vector_dim } -> std::convertible_to<size_t>;
    { t.dst_scalar_per_vector } -> std::convertible_to<size_t>;
};

// Concept to check if algorithm specifies DL thread config
template <typename T>
concept SpecifiesDlThreadConfig = requires {
    { T::dl_thread_config } -> DlThreadConfigDescriptor;
};

// Concept to check if algorithm specifies DL thread cluster
template <typename T>
concept SpecifiesDlThreadCluster = requires {
    { T::dl_thread_cluster } -> DlThreadClusterDescriptor;
};

// Concept to check if algorithm specifies DL A block transfer
template <typename T>
concept SpecifiesDlBlockTransferA = requires {
    { T::dl_block_transfer_a } -> DlBlockTransferK0M0M1K1Descriptor;
};

// Concept to check if algorithm specifies DL B block transfer
template <typename T>
concept SpecifiesDlBlockTransferB = requires {
    { T::dl_block_transfer_b } -> DlBlockTransferK0N0N1K1Descriptor;
};

// Concept to check if algorithm specifies DL C thread transfer
template <typename T>
concept SpecifiesDlCThreadTransfer = requires {
    { T::dl_c_thread_transfer } -> DlCThreadTransferDescriptor;
};

} // namespace ck_tile::builder
