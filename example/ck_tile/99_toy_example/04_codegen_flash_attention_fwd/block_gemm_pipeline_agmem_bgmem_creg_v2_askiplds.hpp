// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "block_gemm_pipeline_agmem_bgmem_creg_v2_askiplds_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy>
struct BlockGemmPipelineAGmemBGmemCReg
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t k_loops = Policy::AKDim / kKPerBlock;

    // Move this part into Policy?
    __host__ __device__ static constexpr index_t GetStaticLdsSize()
    {
        return sizeof(BDataType) *
               Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    // Cold A Register Cache
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename ARegBlockTensorTmp>
    __host__ __device__ auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        ARegBlockTensorTmp& a_reg_block_tensor_tmp,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");

        ignore = a_element_func;
        ignore = b_element_func;

        // A tile in Reg，blockTensor
        // This tensor distribution used to construct both distributed tensor for local buffer store
        // and read. without buffer address info
        constexpr auto a_reg_block_dstr = Policy::template MakeARegBlockDescriptor<Problem>();

        // B tile in LDS, blockWindow
        BDataType* p_b_lds =
            static_cast<BDataType*>(static_cast<void*>(static_cast<char*>(p_smem)));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        // This tensor view used to construct both tile window for lds store and read, with buffer
        // address info
        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A Reg tensor for store, also used for block GEMM
        auto a_copy_reg_tensor = make_static_distributed_tensor<ADataType>(a_reg_block_dstr);

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.get_tile_distribution());

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(
            get_slice_tile(a_copy_reg_tensor, sequence<0, 0>{}, sequence<kMPerBlock, kKPerBlock>{}),
            b_lds_gemm_window)){};

        auto a_block_tile = load_tile(a_copy_dram_window);
        auto b_block_tile = load_tile(b_copy_dram_window);
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        if constexpr(k_loops > 1)
        {
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            set_slice_tile(a_copy_reg_tensor,
                           a_block_tile,
                           sequence<0, 0>{},
                           sequence<kMPerBlock, kKPerBlock>{});
            a_block_tile = load_tile(a_copy_dram_window);

            store_tile(b_copy_lds_window, b_block_tile);
            b_block_tile = load_tile(b_copy_dram_window);
        }

        __builtin_amdgcn_sched_barrier(0);

        if constexpr(k_loops > 2)
        {
            static_for<0, k_loops - 2, 1>{}([&](auto i_k0) {
                block_sync_lds();

                block_gemm(c_block_tile,
                           get_slice_tile(a_copy_reg_tensor,
                                          sequence<0, i_k0 * kKPerBlock>{},
                                          sequence<kMPerBlock, (i_k0 + 1) * kKPerBlock>{}),
                           b_copy_lds_window);

                block_sync_lds();

                move_tile_window(a_copy_dram_window, {0, kKPerBlock});
                move_tile_window(b_copy_dram_window, {0, kKPerBlock});

                set_slice_tile(a_copy_reg_tensor,
                               a_block_tile,
                               sequence<0, (i_k0 + 1) * kKPerBlock>{},
                               sequence<kMPerBlock, (i_k0 + 2) * kKPerBlock>{});
                a_block_tile = load_tile(a_copy_dram_window);

                store_tile(b_copy_lds_window, b_block_tile);
                b_block_tile = load_tile(b_copy_dram_window);

                block_gemm.HotLoopScheduler();
                __builtin_amdgcn_sched_barrier(0);
            });
        }

        // tail
        {
            if constexpr(k_loops > 1)
            {
                block_sync_lds();

                block_gemm(c_block_tile,
                           get_slice_tile(a_copy_reg_tensor,
                                          sequence<0, (k_loops - 2) * kKPerBlock>{},
                                          sequence<kMPerBlock, (k_loops - 1) * kKPerBlock>{}),
                           b_copy_lds_window);

                block_sync_lds();
            }

            set_slice_tile(a_copy_reg_tensor,
                           a_block_tile,
                           sequence<0, (k_loops - 1) * kKPerBlock>{},
                           sequence<kMPerBlock, k_loops * kKPerBlock>{});

            store_tile(b_copy_lds_window, b_block_tile);

            block_sync_lds();

            block_gemm(c_block_tile,
                       get_slice_tile(a_copy_reg_tensor,
                                      sequence<0, (k_loops - 1) * kKPerBlock>{},
                                      sequence<kMPerBlock, k_loops * kKPerBlock>{}),
                       b_copy_lds_window);
        }

        set_slice_tile(a_reg_block_tensor_tmp,
                       a_copy_reg_tensor,
                       sequence<0, 0>{},
                       sequence<kMPerBlock, k_loops * kKPerBlock>{});

        return c_block_tile;
    }

    // Hot A Register Cache
    template <typename BDramBlockWindowTmp, typename BElementFunction, typename ARegBlockTensorTmp>
    __host__ __device__ auto operator()(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        const ARegBlockTensorTmp& a_reg_block_tensor_tmp,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        ignore = b_element_func;

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // A tile in Reg，blockTensor
        // This tensor distribution used to construct both distributed tensor for local buffer store
        // and read. without buffer address info
        constexpr auto a_reg_block_dstr = Policy::template MakeARegBlockDescriptor<Problem>();

        // A Reg tensor for store, also used for block GEMM
        auto a_copy_reg_tensor = make_static_distributed_tensor<ADataType>(a_reg_block_dstr);

        set_slice_tile(a_copy_reg_tensor,
                       a_reg_block_tensor_tmp,
                       sequence<0, 0>{},
                       sequence<kMPerBlock, k_loops * kKPerBlock>{});

        // B tile in LDS, blockWindow
        BDataType* p_b_lds =
            static_cast<BDataType*>(static_cast<void*>(static_cast<char*>(p_smem)));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        // This tensor view used to construct both tile window for lds store and read, with buffer
        // address info
        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.get_tile_distribution());

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {0, 0},
            make_static_tile_distribution(block_gemm.MakeBBlockDistributionEncode()));

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(
            get_slice_tile(a_copy_reg_tensor, sequence<0, 0>{}, sequence<kMPerBlock, kKPerBlock>{}),
            b_lds_gemm_window)){};

        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

#if !defined(TOY_FA_FWD_OPT)
        static_for<0, k_loops, 1>{}([&](auto i_k0) {
            auto b_block_tile = load_tile(b_copy_dram_window);
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});
            store_tile(b_copy_lds_window, b_block_tile);
            block_sync_lds();
            block_gemm(c_block_tile,
                       get_slice_tile(a_copy_reg_tensor,
                                      sequence<0, i_k0 * kKPerBlock>{},
                                      sequence<kMPerBlock, (i_k0 + 1) * kKPerBlock>{}),
                       b_copy_lds_window);
            block_sync_lds();
        });
#else
        using BLdsTile = typename decltype(block_gemm)::BLdsTile;
        BLdsTile bWarpTile;

        // Global read 0
        auto b_block_tile = load_tile(b_copy_dram_window);
        move_tile_window(b_copy_dram_window, {0, kKPerBlock});
        if constexpr(k_loops > 1)
        {
            // LDS write 0
            store_tile(b_copy_lds_window, b_block_tile);

            // Global read 1
            b_block_tile = load_tile(b_copy_dram_window);
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            block_sync_lds();

            // LDS read 0
            bWarpTile = load_tile(b_lds_gemm_window);
        }

        if constexpr(k_loops > 2)
        {
            __builtin_amdgcn_sched_barrier(0);
            static_for<0, k_loops - 2, 1>{}([&](auto i_k0) {
                block_sync_lds();

                // LDS write 1
                store_tile(b_copy_lds_window, b_block_tile);

                // Global read 2
                b_block_tile = load_tile(b_copy_dram_window);
                move_tile_window(b_copy_dram_window, {0, kKPerBlock});

                block_gemm(c_block_tile,
                           get_slice_tile(a_copy_reg_tensor,
                                          sequence<0, i_k0 * kKPerBlock>{},
                                          sequence<kMPerBlock, (i_k0 + 1) * kKPerBlock>{}),
                           bWarpTile);

                block_sync_lds();

                // LDS read 1
                bWarpTile = load_tile(b_lds_gemm_window);

                block_gemm.HotLoopScheduler();
                __builtin_amdgcn_sched_barrier(0);
            });
        }
        // tail
        {
            if constexpr(k_loops > 1)
            {
                block_gemm(c_block_tile,
                           get_slice_tile(a_copy_reg_tensor,
                                          sequence<0, (k_loops - 2) * kKPerBlock>{},
                                          sequence<kMPerBlock, (k_loops - 1) * kKPerBlock>{}),
                           bWarpTile);

                block_sync_lds();
            }
            store_tile(b_copy_lds_window, b_block_tile);

            block_sync_lds();

            bWarpTile = load_tile(b_lds_gemm_window);

            block_gemm(c_block_tile,
                       get_slice_tile(a_copy_reg_tensor,
                                      sequence<0, (k_loops - 1) * kKPerBlock>{},
                                      sequence<kMPerBlock, k_loops * kKPerBlock>{}),
                       bWarpTile);
        }
#endif
        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename ARegBlockTensorTmp>
    __device__ auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                               const BDramBlockWindowTmp& b_dram_block_window_tmp,
                               ARegBlockTensorTmp& a_reg_block_tensor_tmp,
                               void* p_smem) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            a_reg_block_tensor_tmp,
            p_smem);
    }

    template <typename BDramBlockWindowTmp, typename ARegBlockTensorTmp>
    __device__ auto operator()(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                               const ARegBlockTensorTmp& a_reg_block_tensor_tmp,
                               void* p_smem) const
    {
        return operator()(
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            a_reg_block_tensor_tmp,
            p_smem);
    }
};

} // namespace ck_tile
