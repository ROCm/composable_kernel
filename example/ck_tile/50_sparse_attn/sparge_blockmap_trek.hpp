// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/sparge_blockmap_pipeline.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/sparge_kstats_pipeline.hpp"
#include "ck_tile/ops/sparse_attn/kernel/sparge_blockmap_kernel.hpp"
#include "ck_tile/ops/sparse_attn/kernel/sparge_kstats_kernel.hpp"

#include "fmha_fwd_trek.hpp"

#include <string>
#include <type_traits>

// ============================================================================
// Args and traits for sparge block map GPU kernel
// ============================================================================
struct sparge_blockmap_args
{
    const void* q_ptr;
    const void* k_ptr;

    ck_tile::index_t batch;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;

    float simthreshd1;
    float cdfthreshd;
    float topk;
    float scale;

    void* block_map_ptr;
    void* lut_ptr;
    void* valid_block_num_ptr;

    // Caller-owned K-stats workspace; size from sparge_blockmap_get_workspace_size.
    // Internal layout (pooled_k then sim_k) given by sparge_blockmap_workspace_layout.
    void* workspace_ptr = nullptr;

    // size = nhead_q to match SpargeAttn upstream hyperparameter_check
    const float* simthreshd1_per_head_ptr = nullptr;
    const float* cdfthreshd_per_head_ptr  = nullptr;
    const float* topk_per_head_ptr        = nullptr;
};

struct sparge_blockmap_workspace_layout
{
    size_t pooled_k_offset; // bytes from workspace_ptr
    size_t pooled_k_bytes;
    size_t sim_k_offset; // bytes from workspace_ptr
    size_t sim_k_bytes;
    size_t total_bytes;
};

struct sparge_blockmap_traits
{
    std::string data_type;
    int hdim_q;
};

// ============================================================================
// Create kernel args and grid dimensions
// ============================================================================
template <typename BlockMapKernel>
auto sparge_blockmap_create_kargs_and_grids(sparge_blockmap_args args,
                                            const void* pooled_k_ws_ptr,
                                            const void* sim_k_ws_ptr)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = BlockMapKernel::MakeKargs(args.q_ptr,
                                           args.k_ptr,
                                           args.seqlen_q,
                                           args.seqlen_k,
                                           args.hdim_q,
                                           args.nhead_q,
                                           args.nhead_q / args.nhead_k,
                                           args.stride_q,
                                           args.stride_k,
                                           args.nhead_stride_q,
                                           args.nhead_stride_k,
                                           args.batch_stride_q,
                                           args.batch_stride_k,
                                           args.simthreshd1,
                                           args.cdfthreshd,
                                           args.topk,
                                           args.scale,
                                           args.block_map_ptr,
                                           args.lut_ptr,
                                           args.valid_block_num_ptr,
                                           pooled_k_ws_ptr,
                                           sim_k_ws_ptr,
                                           args.topk_per_head_ptr,
                                           args.cdfthreshd_per_head_ptr);

    dim3 grids = BlockMapKernel::GridSize(args.batch, args.nhead_q, args.seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename KStatsKernel>
auto sparge_kstats_create_kargs_and_grids(sparge_blockmap_args args,
                                          void* pooled_k_ws_ptr,
                                          void* sim_k_ws_ptr)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = KStatsKernel::MakeKargs(args.k_ptr,
                                         args.seqlen_k,
                                         args.hdim_q,
                                         args.nhead_k,
                                         args.stride_k,
                                         args.nhead_stride_k,
                                         args.batch_stride_k,
                                         args.simthreshd1,
                                         pooled_k_ws_ptr,
                                         sim_k_ws_ptr,
                                         args.simthreshd1_per_head_ptr);

    dim3 grids = KStatsKernel::GridSize(args.batch, args.nhead_k, args.seqlen_k);
    return ck_tile::make_tuple(kargs, grids);
}

// ============================================================================
// Hand-written template instantiation dispatch
// ============================================================================

// Workspace sizing helpers (host, no template instantiation needed).
sparge_blockmap_workspace_layout
sparge_blockmap_compute_workspace_layout(sparge_blockmap_traits traits, sparge_blockmap_args args);

inline size_t sparge_blockmap_get_workspace_size(sparge_blockmap_traits traits,
                                                 sparge_blockmap_args args)
{
    return sparge_blockmap_compute_workspace_layout(traits, args).total_bytes;
}

// Stage 1: K-stats only. Writes pooled_k + sim_k into args.workspace_ptr.
void sparge_kstats_fwd_oneshot(sparge_blockmap_traits traits,
                               sparge_blockmap_args args,
                               const ck_tile::stream_config& stream_config);

// Stage 2: block_map only. Reads pooled_k + sim_k from args.workspace_ptr.
void sparge_blockmap_only_fwd_oneshot(sparge_blockmap_traits traits,
                                      sparge_blockmap_args args,
                                      const ck_tile::stream_config& stream_config);

// Combined functions: kstats + blockmap + attention with unified timing.
float sparge_jenga_fwd(sparge_blockmap_traits,
                       sparge_blockmap_args,
                       fmha_jenga_fwd_traits,
                       fmha_jenga_fwd_args,
                       const ck_tile::stream_config&);

float sparge_vsa_fwd_combined(sparge_blockmap_traits,
                              sparge_blockmap_args,
                              fmha_vsa_fwd_traits,
                              fmha_vsa_fwd_args,
                              const ck_tile::stream_config&);
