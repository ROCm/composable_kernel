// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"



namespace ck_tile {

template <typename BlockWaves, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WaveTile,   // warp size, seq<M, N>
          typename Vector>     // contiguous elements(vector size) along seq<M, N>
struct TileCopyShape
{
    // We split Workgroup waves into two specialized groups.
    // One for reading data from global -> LDS, the other is doing reduction
    static constexpr index_t WaveGroups = 2;

    static constexpr index_t Block_M = BlockTile::at(number<0>{});      // 32, 64, 
    static constexpr index_t Block_N = BlockTile::at(number<1>{});      // 128, 256

    static constexpr index_t Warp_M = WaveTile::at(number<0>{});        // 32
    static constexpr index_t Warp_N = WaveTile::at(number<1>{});        // 128

    static constexpr index_t Vector_M = Vector::at(number<0>{});        // 8
    static constexpr index_t Vector_N = Vector::at(number<1>{});           // 8

    static constexpr index_t WarpPerBlock_M = integer_divide_ceil(BlockWaves::at(number<0>{}), WaveGroups); // 2/2 = 1, 
    static constexpr index_t WarpPerBlock_N = integer_divide_ceil(BlockWaves::at(number<1>{}), WaveGroups); // 1/2 = 1

    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;   // 32 /  = 4, 4, 
    static constexpr index_t ThreadPerWarp_N = Warp_N / Vector_N;   // 128 / 8 = 16, 16

    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M); // 32 / (1 * 32) = 1, 64 / (1 * 32) = 2
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N); // 128 / (1 * 128) = 1, 256 / (1 * 128) = 2

    static constexpr index_t WaveNum = reduce_on_sequence(BlockWaves{}, multiplies{}, number<1>{}); // 2, 2

    static constexpr index_t BlockSize = get_warp_size() * WaveNum; // 64 * 2 = 128, 128
    static constexpr index_t WaveGroupSize = WaveNum / WaveGroups;  // 2 / 2 = 1, 1
    static_assert(WaveGroupSize == WarpPerBlock_M * WarpPerBlock_N, "Inconsisten wave group size!");
};

template <typename XDataType_, 
          typename BlockShape_>
struct TileCopyProblem
{
    using XDataType      = remove_cvref_t<XDataType_>;
    using BlockShape     = remove_cvref_t<BlockShape_>;
};


template <typename Problem_>
struct TileCopy
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXWaveTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::ThreadPerWarp_M, S::Vector_M>,    // 32/8, 8
                      sequence<S::ThreadPerWarp_N, S::Vector_N>>,   // 128/8, 8
                tuple<sequence<1, 2>>,                              // 4 * 16 = 64 threads
                tuple<sequence<0, 0>>,
                sequence<1, 2>,                                     // 8 * 8 = 64 elements per thread.
                sequence<1, 1>>{});
    }

    using XDataType = typename Problem::XDataType;
    using XWaveDstr = remove_cvref_t<decltype(MakeXWaveTileDistribution<Problem>())>;
    using XWaveTensor = static_distributed_tensor<XDataType, XWaveDstr>;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        constexpr auto x_block_outer_distr_enc = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M>,  // 1, 1
                      sequence<S::Repeat_N, S::WarpPerBlock_N>>, // 1, 1
                tuple<sequence<1, 2>>,                           // 1 * 1 = 1
                tuple<sequence<1, 1>>,
                sequence<1, 2>,                                  // 1 * 1 = 1
                sequence<0, 0>>{};
        
        constexpr auto x_block_dstr_enc = detail::make_embed_tile_distribution_encoding(
            x_block_outer_distr_enc, XWaveDstr::get_static_tile_distribution_encoding());
        constexpr auto x_block_dstr = make_static_tile_distribution(x_block_dstr_enc);

        return x_block_dstr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOutputDistribution()
    {
        return MakeXBlockTileDistribution<Problem>();
    }

    static constexpr auto x_wave_y_lengths =
        to_sequence(XWaveDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto x_wave_y_index_zeros = uniform_sequence_gen_t<XWaveDstr::NDimY, 0>{};

    using XTileDstr = remove_cvref_t<decltype(MakeXBlockTileDistribution<Problem>())>;

    CK_TILE_DEVICE void operator()(const XDataType* p_x, XDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        //__shared__ XDataType x_lds[number<S::Block_M>{} * number<S::Block_N>{}];

        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        /* 
        const auto x_lds_view = make_naive_tensor_view<address_space_enum::lds>(
                x_lds, 
                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                make_tuple(number<S::Block_M>{}, 1),
                number<S::Vector_N>{},
                number<1>{});
                */            
        
        const auto y_m = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_block_window = make_tile_window(x_m_n,
                                               make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                               {iM, 0},
                                               MakeXBlockTileDistribution<Problem>());
        /* 
        auto x_block_lds_window = make_tile_window(x_lds_view,
                                                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                                {0, 0},
                                                MakeXBlockTileDistribution<Problem>());   
        */
        auto y_block_window = make_tile_window(y_m, 
                                                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}), 
                                                {iM, 0},
                                                MakeXBlockTileDistribution<Problem>());                

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        auto my_id = get_warp_id();

        using reg_tile = decltype(load_tile(x_block_window));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            reg_tile x;
            if (my_id == 1)
            {
                // load from DRAM to registers
                x = load_tile(x_block_window);

                // store from registers to DRAM
                store_tile(y_block_window, x);
            }
            __syncthreads();
            move_tile_window(x_block_window, {0, S::Block_N});
            move_tile_window(y_block_window, {0, S::Block_N});            

        }
    }
};

} // namespace ck_tile
