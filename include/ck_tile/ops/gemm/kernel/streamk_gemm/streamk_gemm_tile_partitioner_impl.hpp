// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "streamk_gemm_tile_partitioner.hpp"
namespace ck_tile {

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::StreamKTilePartitionerBase(
    index_t m, index_t n, index_t k, index_t max_active_wgs)
    : max_active_wgs_{max_active_wgs}, n_{n}, k_{k}
{
    iters_per_tile_    = integer_divide_ceil(k, KPerBlock);
    num_tiles_         = integer_divide_ceil(m, MPerBlock) * integer_divide_ceil(n_, NPerBlock);
    remainder_along_k_ = k % KPerBlock;

    bool big_enough         = num_tiles_ > max_active_wgs_;
    index_t remainder_tiles = num_tiles_ % max_active_wgs_;

    if(remainder_tiles)
    {
        sk_tiles_ = big_enough ? full_tiles_ * max_active_wgs_ + (num_tiles_ % max_active_wgs_)
                               : num_tiles_;
        sk_tiles_ = min(num_tiles_, sk_tiles_);
        sk_ctas_  = max_active_wgs_;
        total_sk_iters_ = sk_tiles_ * iters_per_tile_;

        // If there still isn't enough work to saturate all CUs, then just revert to DP only.
        if(total_sk_iters_ < max_active_wgs_)
        {
            sk_tiles_       = 0;
            sk_ctas_        = 0;
            total_sk_iters_ = 0;
        }
    }
    else // Full DP (i.e., no Stream-K)
    {
        sk_tiles_       = 0;
        sk_ctas_        = 0;
        total_sk_iters_ = 0;
    }

    iters_per_sk_cta_ = sk_ctas_ ? total_sk_iters_ / sk_ctas_ : 0;
    extra_iters_      = sk_ctas_ ? total_sk_iters_ % sk_ctas_ : 0;

    dp_tiles_       = num_tiles_ - sk_tiles_;
    total_dp_iters_ = dp_tiles_ * iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_partials_buffer_size(
    index_t acc_element_bytes) const noexcept
{
    return MPerBlock * NPerBlock * acc_element_bytes * sk_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_flags_buffer_size()
    const noexcept
{
    constexpr index_t alignment  = 128;
    const index_t required_bytes = sizeof(index_t) * sk_ctas_;
    const index_t padded_bytes   = ck_tile::integer_least_multiple(required_bytes, alignment);
    return padded_bytes;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_start_iter(
    index_t cta_idx) const noexcept
{
    // Compute the number of extra iterations done before this CTA. If the cta_idx is less than
    // extra_iters, the number of extra iterations before the CTA is exactly the cta_idx. Otherwise,
    // it is extra_iters.
    index_t extra_iters_before_me = ck_tile::min(cta_idx, extra_iters_);
    return total_dp_iters_ + cta_idx * iters_per_sk_cta_ + extra_iters_before_me;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE void
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_iter_boundaries(
    index_t& iter, index_t& iter_end, index_t cta_idx) const noexcept
{
    iter     = get_start_iter(cta_idx);
    iter_end = iter + iters_per_sk_cta_ + (cta_idx < extra_iters_);
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_tile_index(
    index_t iter) const noexcept
{
    return iter / iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE void
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_tile_boundaries(
    index_t& tile_iter, index_t& tile_iter_end, index_t tile_idx) const noexcept
{
    tile_iter     = tile_idx * iters_per_tile_;
    tile_iter_end = tile_iter + iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE /* static */ index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_local_iter(
    index_t iter, index_t tile_iter) noexcept
{
    return iter - tile_iter;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE /* static */ index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_local_iter_end(
    index_t tile_iter, index_t iter_end, index_t tile_iter_end) noexcept
{
    return ck_tile::min(iter_end, tile_iter_end) - tile_iter;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_tile_local_cta_index(
    index_t tile_iter_start, index_t cta_idx) const noexcept
{
    tile_iter_start = tile_iter_start - (dp_tiles_ * iters_per_tile_);

    // Compute how many WGs fit before this tile starts assuming each WG does an
    // extra_iter
    const index_t num_extra_iter_ctas = tile_iter_start / (iters_per_sk_cta_ + 1);
    // Compute how many WGs fit before this tile starts excluding extra iters
    const index_t num_non_extra_iter_ctas = (tile_iter_start - extra_iters_) / iters_per_sk_cta_;
    // Compute the CTA idx for the CTA that starts this tile
    const index_t coop_group_start =
        num_extra_iter_ctas < extra_iters_ ? num_extra_iter_ctas : num_non_extra_iter_ctas;
    return cta_idx - coop_group_start;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE auto
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_output_tile_index(
    index_t tile_idx) const noexcept -> tuple<index_t, index_t>
{
    const index_t n_macro_tiles = integer_divide_ceil(n_, NPerBlock);

    const index_t im = amd_wave_read_first_lane(tile_idx / n_macro_tiles);
    const index_t in = amd_wave_read_first_lane(tile_idx - im * n_macro_tiles);
    return make_tuple(im, in);
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_workspace_size(
    index_t acc_element_bytes) const noexcept
{
    if constexpr(ReductionStrategy == StreamKReductionStrategy::Linear ||
                 ReductionStrategy == StreamKReductionStrategy::Tree)
    {

        return get_partials_buffer_size(acc_element_bytes) + get_flags_buffer_size();
    }
    else // ReductionStrategy is Atomics
    {
        return 0;
    }
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_num_tiles()
    const noexcept
{
    return num_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_max_active_wgs()
    const noexcept
{
    return max_active_wgs_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_dp_tiles() const noexcept
{
    return dp_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_sk_tiles() const noexcept
{
    return sk_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_sk_ctas() const noexcept
{
    return sk_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_total_sk_iters()
    const noexcept
{
    return total_sk_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_iters_per_tile()
    const noexcept
{
    return iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_iters_per_sk_cta()
    const noexcept
{
    return iters_per_sk_cta_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_extra_iters()
    const noexcept
{
    return extra_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_total_dp_iters()
    const noexcept
{
    return total_dp_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_n() const noexcept
{
    return n_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_k() const noexcept
{
    return k_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_remainder_along_k()
    const noexcept
{
    return remainder_along_k_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::estimate_num_wgs_per_tile()
    const noexcept
{
    // In the case of non-atomic reduction or data-parallel (DP) only, there will always be 1
    // workgroup writing final results to a given macro tile in C.
    int num_wgs_per_tile = 1;

    // Otherwise, for atomics, multiple workgroups may be writing to the same macro tile in C.
    if(sk_ctas_ > 0 && ReductionStrategy == ck_tile::StreamKReductionStrategy::Atomic)
    {
        //  If we have DP and SK tiles, this is DP+2TSK which guarantees at most 2 workgroups per
        //  tile. We only need to check that dp_tiles is greater than zero since we know we have SK
        //  workgroups.
        if(dp_tiles_ > 0)
        {
            num_wgs_per_tile = 2;
        }
        else
        {
            ck_tile::index_t iters_per_sk_cta_non_zero = ck_tile::max(iters_per_sk_cta_, 1);
            // Estimate the number of workgroups per macro tile.
            num_wgs_per_tile = (iters_per_tile_ / iters_per_sk_cta_non_zero) +
                               ((iters_per_tile_ % iters_per_sk_cta_non_zero) != 0);
        }
    }

    return std::max(num_wgs_per_tile, 1);
}

/**
 * @brief XCDs access ids in round robin format, this function remaps the 1D ids to continguous
 * XCD segments
 *
 * @param block_1d_id       grid 1D id
 * @param total_num_tiles   size of the 1D grid
 * @param num_xcds          number of XCDs
 * @return index_t  The id after XCD remap
 */
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE /* static */ index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::remap_xcd(
    index_t block_1d_id, index_t total_num_tiles, index_t num_xcds) noexcept
{
    if(num_xcds == 1)
    {
        return block_1d_id;
    }
    // Number of ids per XCD in the new arrangement
    index_t ids_per_xcd = (total_num_tiles + num_xcds - 1) / num_xcds;

    // When total_num_tiles cannot divide num_xcds, some xcds will have
    // ids_per_xcd ids, the other will have ids_per_xcd - 1 ids.
    // We calculate the number of xcds that have ids_per_xcd ids as tall_xcds
    index_t tall_xcds = total_num_tiles % num_xcds;
    tall_xcds         = (tall_xcds == 0) ? num_xcds : tall_xcds;

    // Compute current XCD and local id within the XCD
    index_t xcd      = block_1d_id % num_xcds;
    index_t local_id = block_1d_id / num_xcds;

    // Calculate new id based on the new grouping
    if(xcd < tall_xcds)
    {
        block_1d_id = xcd * ids_per_xcd + local_id;
    }
    else
    {
        block_1d_id = tall_xcds * ids_per_xcd + (xcd - tall_xcds) * (ids_per_xcd - 1) + local_id;
    }

    /**
     * original ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
     * XCD 0 gets: [0, 8], XCD 1 gets: [1, 9], ...
     *
     * post-remap ids: [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
     * XCD 0 gets: [0, 1], XCD 1 gets: [2, 3], ...
     *
     * after remap the ids are continguous on each XCD
     */
    return block_1d_id;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_k_size(
    index_t num_macro_tiles, index_t local_iter_end) const noexcept
{
    // Determine if this workgroup is responsible for the last macro tile in the K dimension
    bool last_tile = get_iters_per_tile() == local_iter_end;
    index_t k_size;
    // If there is no remainder or if the workgroup was not assigned the last macro tile along K,
    // then their k_size will be a multiple of KPerBlock.
    if(!remainder_along_k_ || !last_tile)
    {
        k_size = num_macro_tiles * KPerBlock;
    }
    // Otherwise, there's a remainder. So, k_size is not a multiple of KPerBlock.
    else
    {
        k_size = (num_macro_tiles - 1) * KPerBlock + remainder_along_k_;
    }

    return k_size;
}

template <typename BlockGemmShapeType,
          StreamKReductionStrategy ReductionStrategyType,
          bool Persistent>
struct StreamKTilePartitioner;

// child class for Persistent Tile Partitioner
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, true>::StreamKTilePartitioner(
    ck_tile::index_t m, ck_tile::index_t n, ck_tile::index_t k, ck_tile::index_t max_active_wgs)
    : StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>(m, n, k, max_active_wgs)
{ // inherit from base constructor
    dp_tiles_per_cta_ = this->dp_tiles_ / this->max_active_wgs_;
    extra_dp_tiles_   = this->dp_tiles_ % this->max_active_wgs_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE auto
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, true>::grid_size() const noexcept
    -> dim3
{
    if(extra_dp_tiles_ == 0)
    {
        return dim3(this->max_active_wgs_, 1, 1);
    }
    else
    {
        return dim3(this->num_tiles_, 1, 1);
    }
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, true>::get_dp_tiles_per_cta()
    const noexcept
{
    return dp_tiles_per_cta_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, true>::get_extra_dp_tiles()
    const noexcept
{
    return extra_dp_tiles_;
}

// child class for Non-Persistent Tile Partitioner
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, false>::StreamKTilePartitioner(
    ck_tile::index_t m, ck_tile::index_t n, ck_tile::index_t k, ck_tile::index_t max_active_wgs)
    : StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>(m, n, k, max_active_wgs)
{ // inherit from base constructor
    dp_ctas_            = this->dp_tiles_;
    dp_start_block_idx_ = 0;
    sk_start_block_idx_ = this->dp_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE auto
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, false>::grid_size() const noexcept
    -> dim3
{
    return dim3(dp_ctas_ + this->get_sk_ctas(), 1, 1);
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, false>::get_dp_ctas()
    const noexcept
{
    return dp_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, false>::get_dp_start_block_idx()
    const noexcept
{
    return dp_start_block_idx_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, false>::get_sk_start_block_idx()
    const noexcept
{
    return sk_start_block_idx_;
}

} // namespace ck_tile
