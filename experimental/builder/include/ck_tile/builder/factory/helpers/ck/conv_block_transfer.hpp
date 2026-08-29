// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/array.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory::internal {

// Block transfer parameters for A or B tensor.
template <size_t ThreadClusterRank = 3>
struct BlockTransfer
{
    ck::Array<size_t, ThreadClusterRank> thread_cluster_dims{};
    ck::Array<size_t, ThreadClusterRank> thread_cluster_order{};
    ck::Array<size_t, ThreadClusterRank> src_access_order{};
    size_t src_vector_dim            = 0;
    size_t src_scalar_per_vector     = 0;
    size_t lds_dst_scalar_per_vector = 0;
    bool is_direct_load              = false;
    bool lds_padding                 = false;
};

template <auto TRANSFER>
constexpr BlockTransfer<> SetFwdConvBlockTransfer()
{
    auto& block_xfer  = TRANSFER.block_transfer;
    auto& block_order = TRANSFER.thread_cluster_arrange_order;
    auto& src_order   = TRANSFER.src_access_order;
    auto& lds_cfg     = TRANSFER.lds_transfer;

    return BlockTransfer<>{
        .thread_cluster_dims   = {block_xfer.k0, block_xfer.m_n, block_xfer.k1},
        .thread_cluster_order  = {block_order.order[0], block_order.order[1], block_order.order[2]},
        .src_access_order      = {src_order.order[0], src_order.order[1], src_order.order[2]},
        .src_vector_dim        = lds_cfg.src_vector_dim,
        .src_scalar_per_vector = lds_cfg.src_scalar_per_vector,
        .lds_dst_scalar_per_vector = lds_cfg.lds_dst_scalar_per_vector,
        .is_direct_load            = lds_cfg.is_direct_load,
        .lds_padding               = lds_cfg.lds_padding,
    };
}

template <auto TRANSFER>
constexpr auto SetBwdConvBlockTransfer()
{
    auto& block_xfer  = TRANSFER.block_transfer;
    auto& block_order = TRANSFER.thread_cluster_arrange_order;
    auto& src_order   = TRANSFER.src_access_order;
    auto& lds_cfg     = TRANSFER.lds_transfer;

    constexpr auto array_length = block_order.order.size();
    static_assert(block_order.order.size() == src_order.order.size(),
                  "Mismatched size between block order and src order");

    if constexpr(array_length == 3)
    {
        return BlockTransfer<3>{
            .thread_cluster_dims   = {block_xfer.k0, block_xfer.m_n, block_xfer.k1},
            .thread_cluster_order  = {block_order.order[0],
                                      block_order.order[1],
                                      block_order.order[2]},
            .src_access_order      = {src_order.order[0], src_order.order[1], src_order.order[2]},
            .src_vector_dim        = lds_cfg.src_vector_dim,
            .src_scalar_per_vector = lds_cfg.src_scalar_per_vector,
            .lds_dst_scalar_per_vector = lds_cfg.lds_dst_scalar_per_vector,
            .lds_padding               = lds_cfg.lds_padding,
        };
    }
    else if constexpr(array_length == 4)
    {
        return BlockTransfer<4>{
            .thread_cluster_dims       = {block_xfer.k_batch_size,
                                          block_xfer.k0,
                                          block_xfer.m_n,
                                          block_xfer.k1},
            .thread_cluster_order      = {block_order.order[0],
                                          block_order.order[1],
                                          block_order.order[2],
                                          block_order.order[3]},
            .src_access_order          = {src_order.order[0],
                                          src_order.order[1],
                                          src_order.order[2],
                                          src_order.order[3]},
            .src_vector_dim            = lds_cfg.src_vector_dim,
            .src_scalar_per_vector     = lds_cfg.src_scalar_per_vector,
            .lds_dst_scalar_per_vector = lds_cfg.lds_dst_scalar_per_vector,
            .lds_padding               = lds_cfg.lds_padding,
        };
    }
    else
    {
        static_assert(false, "Internal error: Unsupported array length");
    }
}

// Block transfer parameters for C tensor.
struct CBlockTransfer
{
    size_t m_xdl_per_wave_per_shuffle        = 0;
    size_t n_xdl_per_wave_per_shuffle        = 0;
    ck::Array<size_t, 4> thread_cluster_dims = {0, 0, 0, 0};
    size_t scalar_per_vector                 = 0;
};

template <ConvSignatureDescriptor auto SIGNATURE, ConvAlgorithmDescriptor auto ALGORITHM>
constexpr CBlockTransfer SetCBlockTransfer()
{
    auto& thread_cluster_dims = ALGORITHM.transfer.c.thread_cluster_dims;
    auto& epilogue_config     = ALGORITHM.transfer.c.epilogue;
    return CBlockTransfer{
        .m_xdl_per_wave_per_shuffle = epilogue_config.m_xdl_per_wave_per_shuffle,
        .n_xdl_per_wave_per_shuffle = epilogue_config.n_xdl_per_wave_per_shuffle,
        .thread_cluster_dims =
            {
                thread_cluster_dims.m_block,
                thread_cluster_dims.m_wave_per_xdl,
                thread_cluster_dims.n_block,
                thread_cluster_dims.n_wave_per_xdl,
            },
        .scalar_per_vector = epilogue_config.scalar_per_vector,
    };
}

} // namespace ck_tile::builder::factory::internal
