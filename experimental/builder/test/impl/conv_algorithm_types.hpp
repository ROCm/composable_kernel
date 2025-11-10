// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::test {

namespace ckb = ck_tile::builder;

// Convenience struct for a tuple of m, n, and k values.
template <typename T>
struct MNK
{
    T m{};
    T n{};
    T k{};
};

// Specify thread block dimensions for a GEMM.
struct ThreadBlock
{
    // Thread block size.
    size_t block_size;
    // Size of the submatrix problem in a thread block.
    MNK<size_t> tile_size;
};
static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);

// Describe gridwise XDL GEMM parameters.
struct GridwiseXdlGemm
{
    // NOTE: ak1 and bk1 are difficult to verify in the kernel instantiation!!!
    size_t ak1            = 0;
    size_t bk1            = 0;
    size_t m_per_xdl      = 0;
    size_t n_per_xdl      = 0;
    size_t m_xdl_per_wave = 0;
    size_t n_xdl_per_wave = 0;
};
static_assert(ckb::GridwiseXdlGemmDescriptor<GridwiseXdlGemm>);

// Describe gridwise WMMA GEMM parameters.
struct GridwiseWmmaGemm
{
    size_t k1              = 0;
    size_t m_per_wmma      = 0;
    size_t n_per_wmma      = 0;
    size_t m_wmma_per_wave = 0;
    size_t n_wmma_per_wave = 0;
    PipelineVersion pipeline_version;
};
static_assert(ckb::GridwiseWmmaGemmDescriptor<GridwiseWmmaGemm>);

struct BlockGemm
{
    PipelineVersion pipeline_version;
    PipelineScheduler scheduler;
};
static_assert(ckb::BlockGemmDescriptor<BlockGemm>);

// Describe Aand B block transfer thread cluster lengths.
struct BlockTransfer
{
    size_t k0;
    size_t m_n;
    size_t k1;
};
static_assert(ckb::BlockTransferDescriptor<BlockTransfer>);

// Describe C block transfer thread cluster lengths.
struct ThreadCluster
{
    size_t m_block;
    size_t m_wave_per_xdl;
    size_t n_block;
    size_t n_wave_per_xdl;
};
static_assert(ThreadClusterDescriptor<ThreadCluster>);

struct LdsTransfer
{
    size_t src_vector_dim;
    size_t src_scalar_per_vector;
    size_t lds_dst_scalar_per_vector;
    bool is_direct_load;
    bool lds_padding;
};
static_assert(LdsTransferDescriptor<LdsTransfer>);

struct Epilogue
{
    size_t m_per_wave_per_shuffle;
    size_t n_per_wave_per_shuffle;
    size_t scalar_per_vector;
};
static_assert(EpilogueDescriptor<Epilogue>);

struct AccessOrder
{
    std::array<size_t, 3> order;
};
static_assert(AccessOrderDescriptor<AccessOrder>);

struct BlockTransferABC
{
    BlockTransfer block_transfer_a;
    BlockTransfer block_transfer_b;
    ThreadCluster thread_cluster_dims_c;
    LdsTransfer lds_transfer_a;
    LdsTransfer lds_transfer_b;
    Epilogue epilogue_c;
    AccessOrder block_transfer_access_order_a;
    AccessOrder block_transfer_access_order_b;
    AccessOrder src_access_order_a;
    AccessOrder src_access_order_b;
};

struct ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
{
    ThreadBlock thread_block;
    GridwiseXdlGemm gridwise_gemm;
    BlockTransferABC block_transfer;
    ConvFwdSpecialization fwd_specialization;
    GemmSpecialization gemm_specialization;
    BlockGemm block_gemm;
};
static_assert(
    ckb::ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(
    ckb::SpecifiesThreadBlock<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(
    ckb::SpecifiesGridwiseXdlGemm<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(
    ckb::SpecifiesBlockTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(
    ckb::SpecifiesLdsTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesThreadClusterAccessOrder<
              ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(
    ckb::SpecifiesSourceAccessOrder<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesFwdConcSpecialization<
              ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(
    ckb::SpecifiesBlockGemm<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesGemmSpecialization<
              ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);

struct ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
{
    ThreadBlock thread_block;
    GridwiseXdlGemm gridwise_gemm;
    BlockTransferABC block_transfer;
    ConvFwdSpecialization fwd_specialization;
    GemmSpecialization gemm_specialization;
    size_t num_gemm_k_prefetch_stages;
    size_t num_groups_to_merge;
    PipelineScheduler loop_scheduler;
};
static_assert(
    ckb::ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesThreadBlock<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesGridwiseXdlGemm<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesBlockTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesLdsTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesThreadClusterAccessOrder<
              ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesSourceAccessOrder<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesFwdConcSpecialization<
              ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesNumPrefetchStages<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesGemmSpecialization<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesLoopScheduler<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(
    ckb::SpecifiesNumGroupsToMerge<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);

struct ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle
{
    ThreadBlock thread_block;
    GridwiseWmmaGemm gridwise_gemm;
    BlockTransferABC block_transfer;
    ConvFwdSpecialization fwd_specialization;
    GemmSpecialization gemm_specialization;
    size_t num_gemm_k_prefetch_stages;
    PipelineScheduler loop_scheduler;
};
static_assert(
    ckb::ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(ckb::SpecifiesThreadBlock<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(
    ckb::SpecifiesGridwiseWmmaGemm<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(
    ckb::SpecifiesBlockTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(ckb::SpecifiesLdsTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(ckb::SpecifiesThreadClusterAccessOrder<
              ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(
    ckb::SpecifiesSourceAccessOrder<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(
    ckb::SpecifiesFwdConcSpecialization<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(
    ckb::SpecifiesNumPrefetchStages<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(
    ckb::SpecifiesGemmSpecialization<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
static_assert(
    ckb::SpecifiesLoopScheduler<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);

// DL-specific descriptors
struct DlThreadConfig
{
    size_t k0_per_block;
    size_t k1;
    size_t m1_per_thread;
    size_t n1_per_thread;
    size_t k_per_thread;
};
static_assert(ckb::DlThreadConfigDescriptor<DlThreadConfig>);

struct DlThreadCluster
{
    std::array<size_t, 2> m1_xs; // e.g., {8, 2}
    std::array<size_t, 2> n1_xs; // e.g., {8, 2}
};
static_assert(ckb::DlThreadClusterDescriptor<DlThreadCluster>);

struct DlBlockTransferK0M0M1K1
{
    std::array<size_t, 4> thread_slice_lengths;
    std::array<size_t, 4> thread_cluster_lengths;
    std::array<size_t, 4> thread_cluster_arrange_order;
    std::array<size_t, 4> src_access_order;
    std::array<size_t, 4> src_vector_tensor_lengths;
    std::array<size_t, 4> src_vector_tensor_contiguous_dim_order;
    std::array<size_t, 4> dst_vector_tensor_lengths;
};
static_assert(ckb::DlBlockTransferK0M0M1K1Descriptor<DlBlockTransferK0M0M1K1>);

struct DlBlockTransferK0N0N1K1
{
    std::array<size_t, 4> thread_slice_lengths;
    std::array<size_t, 4> thread_cluster_lengths;
    std::array<size_t, 4> thread_cluster_arrange_order;
    std::array<size_t, 4> src_access_order;
    std::array<size_t, 4> src_vector_tensor_lengths;
    std::array<size_t, 4> src_vector_tensor_contiguous_dim_order;
    std::array<size_t, 4> dst_vector_tensor_lengths;
};
static_assert(ckb::DlBlockTransferK0N0N1K1Descriptor<DlBlockTransferK0N0N1K1>);

struct DlCThreadTransfer
{
    std::array<size_t, 6> src_dst_access_order;
    size_t src_dst_vector_dim;
    size_t dst_scalar_per_vector;
};
static_assert(ckb::DlCThreadTransferDescriptor<DlCThreadTransfer>);

struct ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK
{
    ThreadBlock thread_block;
    ConvFwdSpecialization fwd_specialization;
    GemmSpecialization gemm_specialization;
    DlThreadConfig dl_thread_config;
    DlThreadCluster dl_thread_cluster;
    DlBlockTransferK0M0M1K1 dl_block_transfer_a;
    DlBlockTransferK0N0N1K1 dl_block_transfer_b;
    DlCThreadTransfer dl_c_thread_transfer;
};
static_assert(
    ckb::ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);
static_assert(
    ckb::SpecifiesThreadBlock<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);
static_assert(ckb::SpecifiesFwdConcSpecialization<
              ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);
static_assert(
    ckb::SpecifiesGemmSpecialization<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);
static_assert(
    ckb::SpecifiesDlThreadConfig<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);
static_assert(
    ckb::SpecifiesDlThreadCluster<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);
static_assert(
    ckb::SpecifiesDlBlockTransferA<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);
static_assert(
    ckb::SpecifiesDlBlockTransferB<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);
static_assert(
    ckb::SpecifiesDlCThreadTransfer<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>);

} // namespace ck_tile::builder::test
