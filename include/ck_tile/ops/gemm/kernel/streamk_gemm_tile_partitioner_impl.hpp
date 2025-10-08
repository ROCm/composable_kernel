// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

namespace ck_tile {

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::StreamKTilePartitionerBase(
    index_t m, index_t n, index_t k, index_t grid)
    : grid_{grid}, n_{n}
{
    iters_per_tile_ = integer_divide_ceil(k, KPerBlock);
    num_tiles_      = integer_divide_ceil(m, MPerBlock) * integer_divide_ceil(n_, NPerBlock);

    bool big_enough         = num_tiles_ > grid_;
    index_t remainder_tiles = num_tiles_ % grid_;

    if(remainder_tiles)
    {
        sk_tiles_       = big_enough ? full_tiles_ * grid_ + (num_tiles_ % grid_) : num_tiles_;
        sk_tiles_       = min(num_tiles_, sk_tiles_);
        sk_ctas_        = grid_;
        total_sk_iters_ = sk_tiles_ * iters_per_tile_;

        // If there still isn't enough work to saturate all CUs, then just revert to DP only.
        if(total_sk_iters_ < grid_)
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

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_partials_buffer_size(
    index_t acc_element_bytes) const noexcept
{
    return MPerBlock * NPerBlock * acc_element_bytes * sk_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_flags_buffer_size()
    const noexcept
{
    return sizeof(index_t) * sk_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_DEVICE void
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_iter_boundaries(
    index_t& iter, index_t& iter_end, index_t cta_idx) const noexcept
{
    index_t extra_iters__before_me = ck_tile::min(cta_idx, extra_iters_);
    iter     = total_dp_iters_ + cta_idx * iters_per_sk_cta_ + extra_iters__before_me;
    iter_end = iter + iters_per_sk_cta_ + (cta_idx < extra_iters_);
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_tile_index(
    index_t iter) const noexcept
{
    return iter / iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_DEVICE void
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_tile_boundaries(
    index_t& tile_iter, index_t& tile_iter_end, index_t tile_idx) const noexcept
{
    tile_iter     = tile_idx * iters_per_tile_;
    tile_iter_end = tile_iter + iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_DEVICE /* static */ index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_local_iter(
    index_t iter, index_t tile_iter) noexcept
{
    return iter - tile_iter;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_DEVICE /* static */ index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_local_iter_end(
    index_t tile_iter, index_t iter_end, index_t tile_iter_end) noexcept
{
    return ck_tile::min(iter_end, tile_iter_end) - tile_iter;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_DEVICE auto
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_output_tile_index(
    index_t tile_idx) const noexcept -> tuple<index_t, index_t>
{
    const index_t n_macro_tiles = integer_divide_ceil(n_, NPerBlock);

    const index_t im = amd_wave_read_first_lane(tile_idx / n_macro_tiles);
    const index_t in = amd_wave_read_first_lane(tile_idx - im * n_macro_tiles);
    return make_tuple(im, in);
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_workspace_size(
    index_t acc_element_bytes) const noexcept
{
    if constexpr(StreamKReductionStrategy == StreamKReductionStrategy::Reduction)
    {

        return get_partials_buffer_size(acc_element_bytes) + get_flags_buffer_size();
    }
    else // ReductionStrategy is Atomics
    {
        return 0;
    }
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_num_tiles() const noexcept
{
    return num_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_grid() const noexcept
{
    return grid_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_dp_tiles() const noexcept
{
    return dp_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_sk_tiles() const noexcept
{
    return sk_tiles_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_sk_ctas() const noexcept
{
    return sk_ctas_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_total_sk_iters()
    const noexcept
{
    return total_sk_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_iters_per_tile()
    const noexcept
{
    return iters_per_tile_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_iters_per_sk_cta()
    const noexcept
{
    return iters_per_sk_cta_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_extra_iters() const noexcept
{
    return extra_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_total_dp_iters()
    const noexcept
{
    return total_dp_iters_;
}

template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategy>
CK_TILE_HOST_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategy>::get_n() const noexcept
{
    return n_;
}

} // namespace ck_tile
