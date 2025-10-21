// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::test 
{

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
    int block_size;
    // Size of the submatrix problem in a thread block.
    MNK<int> tile_size;
};
static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);

// Describe some convolution tuning parameters.
struct ConvTuningParams
{
    // NOTE: ak1 and bk1 are difficult to verify in the kernel instantiation!!!
    int ak1            = 0;
    int bk1            = 0;
    int m_per_xdl      = 0;
    int n_per_xdl      = 0;
    int m_xdl_per_wave = 0;
    int n_xdl_per_wave = 0;
};
static_assert(ckb::GridwiseGemmDescriptor<ConvTuningParams>);

// Describe A block transfer thread cluster lengths.
struct InputBlockTransferLengths
{
    int k0;
    int m;
    int k1;
};
static_assert(ckb::InputBlockTransferDescriptor<InputBlockTransferLengths>);

// Describe C block transfer thread cluster lengths.
struct OutputBlockTransferLengths
{
    int m_block;
    int m_wave_per_xdl;
    int n_block;
    int n_wave_per_xdl;
};
static_assert(OutputBlockTransferDescriptor<OutputBlockTransferLengths>);

struct InputVectorTransfer
{
    size_t src_vector_dim;
    size_t src_scalar_per_vector;
    size_t dest_scalar_per_vector_k1;
    bool add_extra; 
};
static_assert(InputVectorTransferDescriptor<InputVectorTransfer>);

struct OutputVectorTransfer
{
    size_t m_xdl_per_wave_per_shuffle;
    size_t n_xdl_per_wave_per_shuffle;
    size_t scalar_per_vector;
};
static_assert(OutputVectorTransferDescriptor<OutputVectorTransfer>);

struct AccessOrder
{
    std::array<int, 3> order;
};
static_assert(AccessOrderDescriptor<AccessOrder>);


struct BlockTransfer
{
    InputBlockTransferLengths thread_cluster_dims_a;
    InputBlockTransferLengths thread_cluster_dims_b;
    OutputBlockTransferLengths thread_cluster_dims_c;
    InputVectorTransfer vector_transfer_a;
    InputVectorTransfer vector_transfer_b;
    OutputVectorTransfer vector_transfer_c;
    AccessOrder thread_cluster_access_order_a;
    AccessOrder thread_cluster_access_order_b;
    AccessOrder src_access_order_a;
    AccessOrder src_access_order_b;
};

struct ConvAlgorithm
{
    ThreadBlock thread_block;
    ConvTuningParams tuning_params;
    BlockTransfer block_transfer;
    BlockGemmPipelineVersion pipeline_version;
};
static_assert(ckb::ConvAlgorithmDescriptor<ConvAlgorithm>);
static_assert(ckb::SpecifiesThreadBlock<ConvAlgorithm>);
static_assert(ckb::SpecifiesGridwiseGemm<ConvAlgorithm>);
static_assert(ckb::SpecifiesGemmPipelineVersion<ConvAlgorithm>);
static_assert(ckb::SpecifiesBlockTransfer<ConvAlgorithm>);
static_assert(ckb::SpecifiesBlockVectorTransfer<ConvAlgorithm>);
static_assert(ckb::SpecifiesThreadClusterAccessOrder<ConvAlgorithm>);
static_assert(ckb::SpecifiesSourceAccessOrder<ConvAlgorithm>);

} // namespace ck_tile::builder::test
