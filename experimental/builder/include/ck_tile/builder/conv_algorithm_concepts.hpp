// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
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

// Common concept for size-related fields
template <typename T>
concept SizeType = std::unsigned_integral<std::remove_cvref_t<T>>;

// Concept for thread block dimensions for a GEMM problem.
template <typename T>
concept ThreadBlockDescriptor = requires(T t) {
    { t.block_size } -> SizeType;
    { t.tile_size.m } -> SizeType;
    { t.tile_size.n } -> SizeType;
    { t.tile_size.k } -> SizeType;
};

// Concept for parameters that describe a gridwise XDL GEMM problem.
template <typename T>
concept GridwiseXdlGemmDescriptor = requires(T t) {
    { t.m_per_xdl } -> SizeType;
    { t.n_per_xdl } -> SizeType;
    { t.m_xdl_per_wave } -> SizeType;
    { t.n_xdl_per_wave } -> SizeType;
};

// Concept for parameter that describe block GEMM problem.
template <typename T>
concept BlockGemmPipelineDescriptor = requires(T t) {
    { t.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { t.scheduler } -> std::convertible_to<PipelineScheduler>;
};

// Concept for parameters that describe a gridwise WMMA GEMM problem.
template <typename T>
concept GridwiseWmmaGemmDescriptor = requires(T t) {
    (
    requires { { T::k1 } -> SizeType; } ||
    (requires { { T::ak1 } -> SizeType; } &&
     requires { { T::bk1 } -> SizeType; })
) &&
requires {
    { T::m_per_wmma } -> SizeType;
    { T::n_per_wmma } -> SizeType;
    { T::m_wmma_per_wave } -> SizeType;
    { T::n_wmma_per_wave } -> SizeType;
};
};

// Concept for vectorized data transfer for convolution input tensors.
template <typename T>
concept BlockTransferDescriptor3D = requires(T t) {
    { t.k0 } -> SizeType;
    { t.m_n } -> SizeType;
    { t.k1 } -> SizeType;
};

template <typename T>
concept BlockTransferDescriptor4D = requires(T t) {
    { t.k0 } -> SizeType;
    { t.m_n } -> SizeType;
    { t.k1 } -> SizeType;
    { t.k_batch_size } -> SizeType;
};

template <typename T, size_t ThreadClusterRank>
concept BlockTransferDescriptor = (ThreadClusterRank == 3 && BlockTransferDescriptor3D<T>) ||
                                  (ThreadClusterRank == 4 && BlockTransferDescriptor4D<T>);

// Concept for thread cluster dimensions for GEMM output tensor.
template <typename T>
concept ThreadClusterDescriptor = requires(T t) {
    { t.m_block } -> SizeType;
    { t.m_wave_per_xdl } -> SizeType;
    { t.n_block } -> SizeType;
    { t.n_wave_per_xdl } -> SizeType;
};

// Concept for the LDS transfer for the convolution input tensors.
template <typename T>
concept LdsTransferDescriptor = requires(T t) {
    { t.src_vector_dim } -> SizeType;
    { t.src_scalar_per_vector } -> SizeType;
    { t.lds_dst_scalar_per_vector } -> SizeType;
    { t.is_direct_load } -> std::convertible_to<bool>;
    { t.lds_padding } -> std::convertible_to<bool>;
};

// Concept for the convolution output tensor epilogue (copy from registers to global memory via
// LDS).
template <typename T>
concept EpilogueDescriptor = requires(T t) {
    { t.m_xdl_per_wave_per_shuffle } -> SizeType;
    { t.n_xdl_per_wave_per_shuffle } -> SizeType;
    { t.scalar_per_vector } -> SizeType;
};

// Concept for the thread cluster access order
template <typename T>
concept ThreadClusterOrderDescriptor = requires(T t) {
    { t.order } -> std::convertible_to<std::array<size_t, 3>>;
} || requires(T t) {
    { t.order } -> std::convertible_to<std::array<size_t, 4>>;
};

// Concept for thread block dimensions for a GEMM problem for CK Tile (Block
// size is deduced from block gemm structure).
template <typename T>
concept TileThreadBlockDescriptor = requires(T t) {
    { t.tile_size.m } -> SizeType;
    { t.tile_size.n } -> SizeType;
    { t.tile_size.k } -> SizeType;
};

// Concept for thread block dimensions for a GEMM problem for CK Tile (Block
// size is deduced from block gemm structure).
template <typename T>
concept TileTransferDescriptor = requires(T t) {
    { t.a_scalar_per_vector } -> SizeType;
    { t.b_scalar_per_vector } -> SizeType;
    { t.c_scalar_per_vector } -> SizeType;
};

// Concept to check if struct specifies block GEMM (CK Tile).
template <typename T>
concept TileBlockGemmDescriptor = requires(T t) {
    { t.warps.m } -> std::convertible_to<int>;
    { t.warps.n } -> std::convertible_to<int>;
    { t.warps.k } -> std::convertible_to<int>;
    { t.warp_tile.m } -> std::convertible_to<int>;
    { t.warp_tile.n } -> std::convertible_to<int>;
    { t.warp_tile.k } -> std::convertible_to<int>;
    { t.double_smem_buffer } -> std::convertible_to<bool>;
    { t.num_wave_groups } -> std::convertible_to<int>;
    { t.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { t.scheduler } -> std::convertible_to<PipelineScheduler>;
};

// Concept to check if struct specifies optimizations (CK Tile).
template <typename T>
concept TileOptimizationsDescriptor = requires(T t) {
    { t.num_groups_to_merge } -> std::convertible_to<int>;
    { t.split_image } -> std::convertible_to<bool>;
    { t.explicit_gemm } -> std::convertible_to<bool>;
    { t.two_stage } -> std::convertible_to<bool>;
};

// Base requirement for all ConvAlgorithm concepts, i.e., all conv algorithm concepts must meet this
// concept.
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

// Concept to check if struct specifies thread block info (CK Tile).
template <typename T>
concept SpecifiesTileThreadBlock = requires {
    { T::thread_block } -> TileThreadBlockDescriptor;
};

// Concept to check if a struct specifies gridwise XDL GEMM info.
template <typename T>
concept GridwiseFwdXdlGemmDescriptor = requires(T t) {
    { t.ak1 } -> SizeType;
    { t.bk1 } -> SizeType;
    { t.xdl_params } -> GridwiseXdlGemmDescriptor;
};

// Concept to check if a struct specifies gridwise XDL GEMM info.
template <typename T>
concept GridwiseBwdXdlGemmDescriptor = requires(T t) {
    { t.k1 } -> SizeType;
    { t.xdl_params } -> GridwiseXdlGemmDescriptor;
};

// Concept to check if a struct specifies gridwise XDL GEMM info.
template <typename T>
concept GridwiseBwdDataXdlGemmDescriptor = requires(T t) {
    { t.ak1 } -> SizeType;
    { t.bk1 } -> SizeType;
    { t.xdl_params } -> GridwiseXdlGemmDescriptor;
};

// Concept to check if a struct specifies gridwise XDL GEMM info.
template <typename T>
concept SpecifiesGridwiseFwdXdlGemm = requires(T t) {
    { t.gridwise_gemm } -> GridwiseFwdXdlGemmDescriptor;
};

// Concept to check if a struct specifies gridwise XDL GEMM info.
template <typename T>
concept SpecifiesGridwiseBwdXdlGemm = requires(T t) {
    { t.gridwise_gemm } -> GridwiseBwdXdlGemmDescriptor;
};

// Concept to check if a struct specifies gridwise XDL GEMM info.
template <typename T>
concept SpecifiesGridwiseBwdDataXdlGemm = requires(T t) {
    { t.gridwise_gemm } -> GridwiseBwdDataXdlGemmDescriptor;
};

// Concept to check if a struct specifies gridwise WMMA GEMM info.
template <typename T>
concept SpecifiesGridwiseWmmaGemm = requires(T t) {
    { t.gridwise_gemm } -> GridwiseWmmaGemmDescriptor;
};

// Concept to check if a struct specifies convolution input and output block transfer info.
template <typename T, size_t ThreadClusterRank = 3>
concept SpecifiesBlockTransfer = requires(T t) {
    { T::transfer.a.block_transfer } -> BlockTransferDescriptor<ThreadClusterRank>;
    { T::transfer.b.block_transfer } -> BlockTransferDescriptor<ThreadClusterRank>;
    { T::transfer.c.thread_cluster_dims } -> ThreadClusterDescriptor;
};

// Concept to check if a struct specifies convolution scalar per vector infor for A, B and C.
template <typename T>
concept SpecifiesTileTransfer = requires(T t) {
    { T::transfer.a_scalar_per_vector } -> SizeType;
    { T::transfer.b_scalar_per_vector } -> SizeType;
    { T::transfer.c_scalar_per_vector } -> SizeType;
};

// Concept to check if a struct specifies LDS transfer info for tensors A, B, and C.
template <typename T>
concept SpecifiesLdsTransfer = requires(T t) {
    { T::transfer.a.lds_transfer } -> LdsTransferDescriptor;
    { T::transfer.b.lds_transfer } -> LdsTransferDescriptor;
    { T::transfer.c.epilogue } -> EpilogueDescriptor;
};

// Concept to check if a struct specifies thread cluster access order info.
template <typename T>
concept SpecifiesThreadClusterArrangeOrder = requires(T t) {
    { T::transfer.a.thread_cluster_arrange_order } -> ThreadClusterOrderDescriptor;
    { T::transfer.b.thread_cluster_arrange_order } -> ThreadClusterOrderDescriptor;
};

// Concept to check if a struct specifies source access order info.
template <typename T>
concept SpecifiesSourceAccessOrder = requires(T t) {
    { T::transfer.a.src_access_order } -> ThreadClusterOrderDescriptor;
    { T::transfer.b.src_access_order } -> ThreadClusterOrderDescriptor;
};

// Concept to check if struct specifies block GEMM.
template <typename T>
concept SpecifiesBlockGemm = requires {
    { T::block_gemm_pipeline } -> BlockGemmPipelineDescriptor;
};

template <typename T>
concept SpecifiesGridwiseGemmPipeline = requires {
    { T::pipeline_version } -> std::convertible_to<PipelineVersion>;
};

// Concept to check if struct specifies block GEMM (CK Tile).
template <typename T>
concept SpecifiesTileBlockGemm = requires {
    { T::block_gemm.warps.m } -> std::convertible_to<int>;
    { T::block_gemm.warps.n } -> std::convertible_to<int>;
    { T::block_gemm.warps.k } -> std::convertible_to<int>;
    { T::block_gemm.warp_tile.m } -> std::convertible_to<int>;
    { T::block_gemm.warp_tile.n } -> std::convertible_to<int>;
    { T::block_gemm.warp_tile.k } -> std::convertible_to<int>;
    { T::block_gemm.double_smem_buffer } -> std::convertible_to<bool>;
    { T::block_gemm.num_wave_groups } -> std::convertible_to<int>;
    { T::block_gemm.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { T::block_gemm.scheduler } -> std::convertible_to<PipelineScheduler>;
};

// Concept to check if struct specifies block GEMM (CK Tile).
template <typename T>
concept SpecifiesTileOptimizations = requires {
    { T::optimizations.num_groups_to_merge } -> std::convertible_to<int>;
    { T::optimizations.split_image } -> std::convertible_to<bool>;
    { T::optimizations.explicit_gemm } -> std::convertible_to<bool>;
    { T::optimizations.two_stage } -> std::convertible_to<bool>;
};

template <typename T>
concept SpecifiesTileConvSpecialization = requires {
    { T::specialization } -> std::convertible_to<TileConvSpecialization>;
};

template <typename T>
concept SpecifiesFwdConvSpecialization = requires {
    { T::fwd_specialization } -> std::convertible_to<ConvSpecialization>;
};

template <typename T>
concept SpecifiesBwdWeightConvSpecialization = requires {
    { T::bwd_weight_specialization } -> std::convertible_to<ConvSpecialization>;
};

template <typename T>
concept SpecifiesBwdDataConvSpecialization = requires {
    { T::bwd_data_specialization } -> std::convertible_to<ConvSpecialization>;
};

template <typename T>
concept SpecifiesGemmSpecialization = requires {
    { T::gemm_specialization } -> std::convertible_to<GemmSpecialization>;
};

template <typename T>
concept SpecifiesNumPrefetchStages = requires {
    { T::num_gemm_k_prefetch_stages } -> SizeType;
};

template <typename T>
concept SpecifiesNumGroupsToMerge = requires {
    { T::num_conv_groups_to_merge } -> SizeType;
};

template <typename T>
concept SpecifiesLoopScheduler = requires {
    { T::loop_scheduler } -> std::convertible_to<PipelineScheduler>;
};

template <typename T>
concept SpecifiesGenericInstance = !requires {
    { T::specialization };
};

template <typename T>
concept SpecifiesTransposeTransfer = requires {
    { T::max_transpose_transfer_src_scalar_per_vector } -> SizeType;
    { T::max_transpose_transfer_dst_scalar_per_vector } -> SizeType;
};

template <typename T>
concept HasTransposeTransfer = requires {
    { T::max_transpose_transfer_src_scalar_per_vector };
    { T::max_transpose_transfer_dst_scalar_per_vector };
};

template <typename T>
concept TransposeTransferWellDefinedIfProvided =
    !HasTransposeTransfer<T> || SpecifiesTransposeTransfer<T>;

template <typename T>
concept SpecifiesGemmBatchOptions = requires {
    { T::num_conv_groups_to_merge } -> SizeType;
};

/******************************************** */
/* Algorithm specialization concepts          */
/******************************************** */
template <typename T>
concept SpecifiesLargeTensorSupport = requires {
    { T::specialization } -> std::convertible_to<ConvAlgorithmSpecialization>;
    requires T::specialization == ConvAlgorithmSpecialization::LARGE_TENSOR;
};

template <typename T>
concept SpecifiesReferenceAlgorithm = requires {
    { T::specialization } -> std::convertible_to<ConvAlgorithmSpecialization>;
    requires T::specialization == ConvAlgorithmSpecialization::REFERENCE;
};

template <typename T>
concept SpecifiesTwoStageSupport = requires {
    { T::specialization } -> std::convertible_to<ConvAlgorithmSpecialization>;
    requires T::specialization == ConvAlgorithmSpecialization::TWO_STAGE;
};

template <typename T>
concept SpecifiesMultipleDSupport = requires {
    { T::specialization } -> std::convertible_to<ConvAlgorithmSpecialization>;
    requires T::specialization == ConvAlgorithmSpecialization::MULTIPLE_D;
};

/******************************************** */
/* DL-specific descriptors and requirements   */
/******************************************** */

// Concept for DL thread configuration
template <typename T>
concept DlThreadConfigDescriptor = requires(T t) {
    { t.k0_per_block } -> SizeType;
    { t.k1 } -> SizeType;
    { t.m1_per_thread } -> SizeType;
    { t.n1_per_thread } -> SizeType;
    { t.k_per_thread } -> SizeType;
};

// Concept for DL thread cluster
template <typename T>
concept DlThreadClusterDescriptor = requires(T t) {
    { t.m1_xs } -> std::convertible_to<std::array<size_t, 2>>;
    { t.n1_xs } -> std::convertible_to<std::array<size_t, 2>>;
};

// Concept for DL block transfer
template <typename T, size_t N>
concept DlBlockTransferDescriptor = requires(T t) {
    { t.thread_slice_lengths } -> std::convertible_to<std::array<size_t, N>>;
    { t.thread_cluster_lengths } -> std::convertible_to<std::array<size_t, N>>;
    { t.thread_cluster_arrange_order } -> std::convertible_to<std::array<size_t, N>>;
    { t.src_access_order } -> std::convertible_to<std::array<size_t, N>>;
    { t.src_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, N>>;
    { t.src_vector_tensor_contiguous_dim_order } -> std::convertible_to<std::array<size_t, N>>;
    { t.dst_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, N>>;
};

template <typename T>
concept DlBlockTransferDescriptor4D = DlBlockTransferDescriptor<T, 4>;

template <typename T>
concept DlBlockTransferDescriptor5D = DlBlockTransferDescriptor<T, 5>;

// Concept for DL epilogue
template <typename T>
concept DlEpilogueDescriptor = requires(T t) {
    { t.src_dst_access_order } -> std::convertible_to<std::array<size_t, 6>>;
    { t.src_dst_vector_dim } -> SizeType;
    { t.dst_scalar_per_vector } -> SizeType;
};

// Concept to check if algorithm specifies DL thread config
template <typename T>
concept SpecifiesDlThreadConfig = requires {
    { T::thread_config } -> DlThreadConfigDescriptor;
};

// Concept to check if algorithm specifies DL thread cluster
template <typename T>
concept SpecifiesDlThreadCluster = requires {
    { T::thread_cluster } -> DlThreadClusterDescriptor;
};

// Concept to check if algorithm specifies DL block transfer
template <typename T>
concept SpecifiesDlFwdBlockTransfer = requires {
    { T::transfer.a } -> DlBlockTransferDescriptor4D;
    { T::transfer.b } -> DlBlockTransferDescriptor4D;
};

template <typename T>
concept SpecifiesDlBwdBlockTransfer = requires {
    { T::transfer.a } -> DlBlockTransferDescriptor5D;
    { T::transfer.b } -> DlBlockTransferDescriptor5D;
};

// Concept to check if algorithm specifies DL C thread transfer
template <typename T>
concept SpecifiesDlEpilogue = requires {
    { T::transfer.c } -> DlEpilogueDescriptor;
};

// Concept to detect StreamK configuration in a tile algorithm descriptor.
template <typename T>
concept StreamKDescriptor = requires(T t) {
    { t.reduction_strategy } -> std::convertible_to<StreamKReductionStrategy>;
    { t.persistent } -> std::convertible_to<bool>;
};

// Concept to check if a tile algorithm specifies StreamK work distribution.
template <typename T>
concept SpecifiesStreamK = requires {
    { T::streamk } -> StreamKDescriptor;
};

} // namespace ck_tile::builder
