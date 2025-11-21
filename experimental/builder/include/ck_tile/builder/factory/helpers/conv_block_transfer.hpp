// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/array.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory_internal {

// Block transfer parameters for A or B tensor.
struct BlockTransfer
{
    ck::Array<size_t, 3> thread_cluster_dims  = {0, 0, 0}; // k0, m, k1
    ck::Array<size_t, 3> thread_cluster_order = {0, 0, 0};
    ck::Array<size_t, 3> src_access_order     = {0, 0, 0};
    size_t src_vector_dim                     = 0;
    size_t src_scalar_per_vector              = 0;
    size_t lds_dst_scalar_per_vector          = 0;
    bool is_direct_load                       = false;
    bool lds_padding                          = false;
};

template <auto TRANSFER>
constexpr BlockTransfer SetFwdConvBlockTransfer()
{
    constexpr auto& TCL = TRANSFER.block_transfer;
    constexpr auto& TCO = TRANSFER.block_transfer_access_order;
    constexpr auto& SAO = TRANSFER.src_access_order;
    constexpr auto& LDS = TRANSFER.lds_transfer;

    BlockTransfer block_transfer{.thread_cluster_dims  = {TCL.k0, TCL.m_n, TCL.k1},
                                 .thread_cluster_order = {TCO.order[0], TCO.order[1], TCO.order[2]},
                                 .src_access_order     = {SAO.order[0], SAO.order[1], SAO.order[2]},
                                 .src_vector_dim       = LDS.src_vector_dim,
                                 .src_scalar_per_vector     = LDS.src_scalar_per_vector,
                                 .lds_dst_scalar_per_vector = LDS.lds_dst_scalar_per_vector,
                                 .is_direct_load            = LDS.is_direct_load,
                                 .lds_padding               = LDS.lds_padding};
    return block_transfer;
}

// Block transfer parameters for C tensor.
struct CBlockTransfer
{
    size_t m_per_wave_per_shuffle            = 0;
    size_t n_per_wave_per_shuffle            = 0;
    ck::Array<size_t, 4> thread_cluster_dims = {0, 0, 0, 0};
    size_t scalar_per_vector                 = 0;
};

template <ConvSignatureDescriptor auto SIGNATURE, ConvAlgorithmDescriptor auto ALGORITHM>
constexpr CBlockTransfer SetCBlockTransfer()
{
    constexpr auto& TCL = ALGORITHM.transfer.c.thread_cluster_dims;
    constexpr auto& EPC = ALGORITHM.transfer.c.epilogue;
    CBlockTransfer block_transfer{.m_per_wave_per_shuffle = EPC.m_per_wave_per_shuffle,
                                  .n_per_wave_per_shuffle = EPC.n_per_wave_per_shuffle,
                                  .thread_cluster_dims =
                                      {
                                          TCL.m_block,
                                          TCL.m_wave_per_xdl,
                                          TCL.n_block,
                                          TCL.n_wave_per_xdl,
                                      },
                                  .scalar_per_vector = EPC.scalar_per_vector};
    return block_transfer;
}

} // namespace ck_tile::builder::factory_internal
