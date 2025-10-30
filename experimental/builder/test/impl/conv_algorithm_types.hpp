// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

// Describe gridwise GEMM parameters.
struct GridwiseGemm
{
    // NOTE: ak1 and bk1 are difficult to verify in the kernel instantiation!!!
    size_t ak1            = 0;
    size_t bk1            = 0;
    size_t m_per_xdl      = 0;
    size_t n_per_xdl      = 0;
    size_t m_xdl_per_wave = 0;
    size_t n_xdl_per_wave = 0;
};
static_assert(ckb::GridwiseGemmDescriptor<GridwiseGemm>);

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
    size_t m_xdl_per_wave_per_shuffle;
    size_t n_xdl_per_wave_per_shuffle;
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
    GridwiseGemm gridwise_gemm;
    BlockTransferABC block_transfer;
    ConvFwdSpecialization fwd_specialization;
    BlockGemmPipelineVersion pipeline_version;
};
static_assert(ckb::ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesThreadBlock<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesGridwiseGemm<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesBlockTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesLdsTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesThreadClusterAccessOrder<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesSourceAccessOrder<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesFwdConcSpecialization<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
static_assert(ckb::SpecifiesGemmPipelineVersion<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);

struct ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
{
    ThreadBlock thread_block;
    GridwiseGemm gridwise_gemm;
    BlockTransferABC block_transfer;
    ConvFwdSpecialization fwd_specialization;
    size_t num_gemm_k_prefetch_stages;
};
static_assert(ckb::ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesThreadBlock<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesGridwiseGemm<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesBlockTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesLdsTransfer<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesThreadClusterAccessOrder<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesSourceAccessOrder<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesFwdConcSpecialization<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
static_assert(ckb::SpecifiesNumPrefetchStages<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);

} // namespace ck_tile::builder::test
