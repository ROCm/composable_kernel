// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

template <typename BlockWarps, typename BlockTile, typename WarpTile, typename Vector>
struct LoadAndConvertShape
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});
    static constexpr index_t Block_K = BlockTile::at(number<2>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile::at(number<1>{});
    static constexpr index_t Warp_K = WarpTile::at(number<2>{});

    static constexpr index_t Vector_N = Vector::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = BlockWarps::at(number<1>{});
    static constexpr index_t WarpPerBlock_K = BlockWarps::at(number<2>{});

    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);
    static constexpr index_t Repeat_K = Block_K / (WarpPerBlock_K * Warp_K);

    static constexpr index_t BlockSize =
        ck_tile::get_warp_size() * reduce_on_sequence(BlockWarps{}, multiplies<>{}, number<1>{});
};

template <typename XDataType_, typename YDataType_, typename BlockShape_, typename LoadTranspose_>
struct LoadAndConvertProblem
{
    using XDataType     = remove_cvref_t<XDataType_>;
    using YDataType     = remove_cvref_t<YDataType_>;
    using BlockShape    = remove_cvref_t<BlockShape_>;
    using LoadTranspose = remove_cvref_t<LoadTranspose_>;
};

template <typename Problem_>
struct LoadAndConvertKernel
{
    using Problem       = ck_tile::remove_cvref_t<Problem_>;
    using XDataType     = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using YDataType     = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using LoadTranspose = ck_tile::remove_cvref_t<typename Problem::LoadTranspose>;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    template <index_t NumAccess>
    static constexpr auto get_warp_dstr_encoding()
    {
        using S = typename Problem::BlockShape;

        if constexpr(NumAccess == 1)
            return tile_distribution_encoding<sequence<>,
                                              tuple<sequence<S::Block_N>, sequence<2, S::Vector_N>>,
                                              tuple<sequence<2, 1>>,
                                              tuple<sequence<0, 0>>,
                                              sequence<2>,
                                              sequence<1>>{};
        else
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Block_N>, sequence<NumAccess, 2, S::Vector_N / NumAccess>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<1, 0>>,
                sequence<2, 2>,
                sequence<0, 2>>{};
    }

    template <typename DataType>
    CK_TILE_DEVICE static constexpr auto GetVectorSize()
    {
        return DS_READ_TR_SIZE() / sizeof(DataType);
    }

    template <typename DataType>
    CK_TILE_DEVICE static constexpr auto MakeDRAMDistribution()
    {
        using S                           = typename Problem::BlockShape;
        constexpr index_t thread_elements = S::Warp_N * S::Warp_K / get_warp_size();
        constexpr index_t NumAccess       = thread_elements / GetVectorSize<DataType>();

        constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<S::WarpPerBlock_N>,
            tuple<sequence<S::Repeat_M, S::WarpPerBlock_M>, sequence<S::Repeat_K>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encode, get_warp_dstr_encoding<NumAccess>());

        return make_static_tile_distribution(a_block_dstr_encode);
    }

    template <typename DataType>
    CK_TILE_DEVICE static constexpr auto MakeDRAMTransposedDistribution()
    {
        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                typename decltype(MakeDRAMDistribution<DataType>())::DstrEncode,
                DataType>::TransposedDstrEncode{});
    }

    CK_TILE_DEVICE void
    operator()(const XDataType* a, YDataType* c, index_t M, index_t N, index_t K) const
    {
        using S = typename Problem::BlockShape;

        const index_t kMPerBlock = S::WarpPerBlock_M * S::Repeat_M * S::Block_M;
        const index_t kNPerBlock = S::WarpPerBlock_N * S::Repeat_N * S::Block_N;

        constexpr auto block_dims    = make_tuple(number<kMPerBlock>{}, number<S::Block_K>{});
        constexpr auto block_strides = make_tuple(number<1>{}, number<kMPerBlock>{});
        const index_t num_blocks_n   = N / kNPerBlock;
        const index_t block_m        = get_block_id() / num_blocks_n;

        const index_t m_block_base = block_m * kMPerBlock;

        // LDS buffer
        __shared__ XDataType a_lds[kMPerBlock * S::Block_K];

        auto a_lds_write_view = make_naive_tensor_view<address_space_enum::lds>(
            a_lds, block_dims, block_strides, number<1>{}, number<1>{});

        auto a_block_lds_write_window = make_tile_window(a_lds_write_view, block_dims, {0, 0});

        auto a_block_lds_read_window = [&] {
            if constexpr(LoadTranspose::value)
            {
                constexpr auto block_dims_t =
                    make_tuple(number<S::Block_K>{}, number<kMPerBlock>{});
                constexpr auto block_strides_t = make_tuple(number<kMPerBlock>{}, number<1>{});

                auto view = make_naive_tensor_view<address_space_enum::lds>(
                    a_lds,
                    block_dims_t,
                    block_strides_t,
                    number<GetVectorSize<XDataType>()>{},
                    number<1>{});

                return make_tile_window(
                    view, block_dims_t, {0, 0}, MakeDRAMTransposedDistribution<XDataType>());
            }
            else
            {
                auto view = make_naive_tensor_view<address_space_enum::lds>(
                    a_lds, block_dims, block_strides, number<1>{}, number<1>{});

                return make_tile_window(
                    view, block_dims, {0, 0}, MakeDRAMDistribution<XDataType>());
            }
        }();

        // Input tensor
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            a, make_tuple(M, K), make_tuple(1, M), number<1>{}, number<1>{});

        auto a_block_window = make_tile_window(
            a_tensor, block_dims, {m_block_base, 0}, MakeDRAMDistribution<XDataType>());

        // Output tensor
        auto c_tensor = make_naive_tensor_view<address_space_enum::global>(
            c, make_tuple(M, N), make_tuple(1, M), number<1>{}, number<1>{});

        auto c_block_window = make_tile_window(
            c_tensor, block_dims, {m_block_base, 0}, MakeDRAMDistribution<YDataType>());

        const index_t num_k_loops = K / S::Block_K;
        for(index_t k_iter = 0; k_iter < num_k_loops; ++k_iter)
        {
            auto dram_tile = load_tile(a_block_window);
            store_tile(a_block_lds_write_window, dram_tile);
            block_sync_lds();

            decltype(load_tile(c_block_window)) c_tile;
            load_and_convert_tile<8, LoadTranspose::value>(c_tile, a_block_lds_read_window);
            store_tile(c_block_window, c_tile);

            if(k_iter < num_k_loops - 1)
            {
                move_tile_window(a_block_window, {0, S::Block_K});
                move_tile_window(c_block_window, {0, S::Block_K});
            }
        }
    }
};

} // namespace ck_tile
