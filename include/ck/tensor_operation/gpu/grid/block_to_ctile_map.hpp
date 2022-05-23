#ifndef UTILITY_BLOCK_TO_CTILE_MAP
#define UTILITY_BLOCK_TO_CTILE_MAP

#include "utility/math.hpp"
#include "utility/number.hpp"
#include "tensor_description/tensor_adaptor.hpp"
#include "tensor_description/multi_index_transform_helper.hpp"

namespace ck {

// Blocks of row-vectors
template <index_t MPerBlock,
          index_t NPerBlock,
          typename CGridDesc_M_N,
          bool DeviceCTileIndexCheck = false>
struct BlockToCTileMap_M00_N00_M01_N01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ BlockToCTileMap_M00_N00_M01_N01() = default;

    __host__ __device__ BlockToCTileMap_M00_N00_M01_N01(const CGridDesc_M_N& c_grid_desc_m_n,
                                                        index_t M01 = 1,
                                                        index_t N01 = 1)
        : M01_(M01), N01_(N01), underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01, N01))
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);
        const auto N00 = math::integer_divide_ceil(N0, N01_);

        const index_t grid_size = M00 * M01_ * N00 * N01_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return underlying_map_.CalculateBottomIndex(idx_top);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return DefaultValidCTileIndex(c_tile_idx, c_tile_dim);
        else
            return true;
    }

    __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return true; // validity check moved to kernel

        const index_t M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const index_t N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);
        if(M0 % M01_ == 0 && N0 % N01_ == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private:
    __host__ __device__ static constexpr auto
    GetBlockToCTileMap(const CGridDesc_M_N& c_grid_desc_m_n, index_t M01, index_t N01)
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01);
        const auto N00 = math::integer_divide_ceil(N0, N01);

        const auto m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_insert_transform(1), // swallow the carry from lower dimensions
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto cblockid_to_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(1, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto cblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  cblockid_to_m00_m01_n00_n01_block_cluster_adaptor);

        return cblockid_to_m0_n0_block_cluster_adaptor;
    }

    index_t M01_, N01_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1, 1));
    UnderlyingMap underlying_map_;
};

// 2D slices of row-vectors in 3D space
template <index_t MPerBlock,
          index_t NPerBlock,
          typename CGridDesc_M_N,
          bool DeviceCTileIndexCheck = false>
struct BlockToCTileMap_KSplit_M00_N00_M01_N01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ BlockToCTileMap_KSplit_M00_N00_M01_N01() = default;

    __host__ BlockToCTileMap_KSplit_M00_N00_M01_N01(const CGridDesc_M_N& c_grid_desc_m_n,
                                                    index_t M01    = 1,
                                                    index_t N01    = 1,
                                                    index_t KSplit = 1)
        : M01_(M01),
          N01_(N01),
          KSplit_(KSplit),
          underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01, N01, KSplit))
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);
        const auto N00 = math::integer_divide_ceil(N0, N01_);

        const index_t grid_size = M00 * M01_ * N00 * N01_ * KSplit_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return underlying_map_.CalculateBottomIndex(idx_top);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return DefaultValidCTileIndex(c_tile_idx, c_tile_dim);
        else
            return true;
    }

    __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return true; // validity check moved to kernel

        const index_t M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const index_t N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);
        if(M0 % M01_ == 0 && N0 % N01_ == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private:
    __host__ static constexpr auto GetBlockToCTileMap(const CGridDesc_M_N& c_grid_desc_m_n,
                                                      index_t M01,
                                                      index_t N01,
                                                      index_t KSplit)
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01);
        const auto N00 = math::integer_divide_ceil(N0, N01);

        const auto ksplit_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(KSplit),
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto c_blockid_to_ksplit_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(KSplit, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto c_blockid_to_ksplit_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(ksplit_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  c_blockid_to_ksplit_m00_m01_n00_n01_block_cluster_adaptor);

        return c_blockid_to_ksplit_m0_n0_block_cluster_adaptor;
    }

    index_t M01_, N01_, KSplit_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1, 1, 1));
    UnderlyingMap underlying_map_;
};

template <typename CTileIdx, typename CTileDim>
__host__ __device__ bool DefaultValidCTileIndex(const CTileIdx& c_tile_idx,
                                                const CTileDim& c_tile_dim)
{
    bool is_valid = false;

    const index_t m_block = c_tile_dim[Number<0>{}];
    const index_t n_block = c_tile_dim[Number<1>{}];

    if constexpr(CTileIdx::Size() == 2)
    {
        const index_t m_block_idx = c_tile_idx[Number<0>{}];
        const index_t n_block_idx = c_tile_idx[Number<1>{}];
        if(0 <= m_block_idx && m_block_idx < m_block && 0 <= n_block_idx && n_block_idx < n_block)
        {
            is_valid = true;
        }
    }
    else if constexpr(CTileIdx::Size() == 3)
    {
        const index_t ksplit_idx  = c_tile_idx[Number<0>{}];
        const index_t m_block_idx = c_tile_idx[Number<1>{}];
        const index_t n_block_idx = c_tile_idx[Number<2>{}];
        if(0 <= m_block_idx && m_block_idx < m_block && 0 <= n_block_idx && n_block_idx < n_block)
        {
            is_valid = true;
        }
        ignore = ksplit_idx;
    }

    return is_valid;
}

} // namespace ck

#endif // UTILITY_BLOCK_TO_CTILE_MAP
