// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2_askiplds_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BlockGemmPipelineAGmemBGmemCRegV2<Problem, BlockGemmPipelineAGmemBGmemCRegV2SkipALdsPolicy>
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using Policy         = BlockGemmPipelineAGmemBGmemCRegV2SkipALdsPolicy;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    // Move this part into Policy?
    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return sizeof(BDataType) *
               Policy::template MakeBLdsBlockDescriptor<Problem>().GetElementSpaceSize();
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    __host__ __device__ auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}],
                      "wrong!");

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
        auto b_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}),
                             a_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A Reg tensor for store, also used for block GEMM
        auto a_copy_reg_tensor = make_static_distributed_tensor<ADataType>(a_reg_block_dstr);

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             b_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.GetTileDistribution());

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(a_copy_reg_tensor, b_lds_gemm_window)){};

        // prefetch
        // global read 0
        auto a_block_tile = load_tile(a_copy_dram_window);
        auto b_block_tile = load_tile(b_copy_dram_window);

        {
            // move to 1
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // Initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // block buffer write 0
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            // store_tile -> shuffle store tile
            store_tile(a_copy_reg_tensor, a_block_tile_tmp);
            // global read 1
            a_block_tile = load_tile(a_copy_dram_window);

            // LDS write 0
            const auto b_block_tile_tmp = tile_elementwise_in(b_element_func, b_block_tile);
            store_tile(b_copy_lds_window, b_block_tile_tmp);
            // global read 1
            b_block_tile = load_tile(b_copy_dram_window);
        }

        index_t iCounter = num_loop - 2;

        do
        {
            block_sync_lds();

            // GEMM i
            block_gemm(c_block_tile, a_copy_reg_tensor, b_lds_gemm_window);

            block_sync_lds();

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_reg_tensor, a_block_tile_tmp);
            // global read i + 2
            a_block_tile = load_tile(a_copy_dram_window);

            // LDS write i + 1
            const auto b_block_tile_tmp = tile_elementwise_in(b_element_func, b_block_tile);
            store_tile(b_copy_lds_window, b_block_tile_tmp);
            // global read i + 2
            b_block_tile = load_tile(b_copy_dram_window);

            iCounter--;

        } while(iCounter > 0);

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            block_gemm(c_block_tile, a_copy_reg_tensor, b_lds_gemm_window);

            block_sync_lds();

            // LDS write num_loop - 1
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_reg_tensor, a_block_tile_tmp);

            const auto b_block_tile_tmp = tile_elementwise_in(b_element_func, b_block_tile);
            store_tile(b_copy_lds_window, b_block_tile_tmp);

            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm(c_block_tile, a_copy_reg_tensor, b_lds_gemm_window);
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    __device__ auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                               const BDramBlockWindowTmp& b_dram_block_window_tmp,
                               index_t num_loop,
                               void* p_smem) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem);
    }
};

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, index_t kHeadDim>
struct BlockGemmPipelineAGmemBGmemCRegV2<
    Problem,
    BlockGemmPipelineAGmemBGmemCRegV2SkipALdsPersistentQRegCachePolicy<kHeadDim>>
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using Policy = BlockGemmPipelineAGmemBGmemCRegV2SkipALdsPersistentQRegCachePolicy<kHeadDim>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t k_loops = Policy::AKDim / kKPerBlock;

    // Move this part into Policy?
    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return sizeof(BDataType) *
               Policy::template MakeBLdsBlockDescriptor<Problem>().GetElementSpaceSize();
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
            is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}],
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
        auto b_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}),
                             a_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A Reg tensor for store, also used for block GEMM
        auto a_copy_reg_tensor = make_static_distributed_tensor<ADataType>(a_reg_block_dstr);

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             b_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.GetTileDistribution());

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(
            get_slice_tile(a_copy_reg_tensor, Sequence<0, 0>{}, Sequence<kMPerBlock, kKPerBlock>{}),
            b_lds_gemm_window)){};

        auto a_block_tile = load_tile(a_copy_dram_window);
        auto b_block_tile = load_tile(b_copy_dram_window);
        {
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            set_slice_tile(a_copy_reg_tensor,
                           a_block_tile,
                           Sequence<0, 0>{},
                           Sequence<kMPerBlock, kKPerBlock>{});
            a_block_tile = load_tile(a_copy_dram_window);

            store_tile(b_copy_lds_window, b_block_tile);
            b_block_tile = load_tile(b_copy_dram_window);
        }
        if constexpr(k_loops > 2)
        {
            static_for<0, k_loops - 2, 1>{}([&](auto i_k0) {
                block_sync_lds();

                block_gemm(c_block_tile,
                           get_slice_tile(a_copy_reg_tensor,
                                          Sequence<0, (i_k0)*kKPerBlock>{},
                                          Sequence<kMPerBlock, (i_k0 + 1) * kKPerBlock>{}),
                           b_copy_lds_window);

                block_sync_lds();

                move_tile_window(a_copy_dram_window, {0, kKPerBlock});
                move_tile_window(b_copy_dram_window, {0, kKPerBlock});

                set_slice_tile(a_copy_reg_tensor,
                               a_block_tile,
                               Sequence<0, (i_k0 + 1) * kKPerBlock>{},
                               Sequence<kMPerBlock, (i_k0 + 2) * kKPerBlock>{});
                a_block_tile = load_tile(a_copy_dram_window);

                store_tile(b_copy_lds_window, b_block_tile);
                b_block_tile = load_tile(b_copy_dram_window);
            });
        }

        // tail
        {
            block_sync_lds();

            block_gemm(c_block_tile,
                       get_slice_tile(a_copy_reg_tensor,
                                      Sequence<0, (k_loops - 2) * kKPerBlock>{},
                                      Sequence<kMPerBlock, (k_loops - 1) * kKPerBlock>{}),
                       b_copy_lds_window);

            block_sync_lds();

            set_slice_tile(a_copy_reg_tensor,
                           a_block_tile,
                           Sequence<0, (k_loops - 1) * kKPerBlock>{},
                           Sequence<kMPerBlock, k_loops * kKPerBlock>{});

            store_tile(b_copy_lds_window, b_block_tile);

            block_sync_lds();

            block_gemm(c_block_tile,
                       get_slice_tile(a_copy_reg_tensor,
                                      Sequence<0, (k_loops - 1) * kKPerBlock>{},
                                      Sequence<kMPerBlock, (k_loops)*kKPerBlock>{}),
                       b_copy_lds_window);
        }

        store_tile(a_reg_block_tensor_tmp, a_copy_reg_tensor);

        return c_block_tile;
    }

    // Hot A Register Cache
    template <typename BDramBlockWindowTmp, typename BElementFunction, typename ARegBlockTensorTmp>
    __host__ __device__ auto operator()(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        const ARegBlockTensorTmp& a_reg_block_tensor_tmp,
                                        void* p_smem) const
    {
        static_assert(is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                      "wrong!");

        static_assert(kNPerBlock == BDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kKPerBlock == BDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}],
                      "wrong!");

        ignore = b_element_func;

        // A tile in Reg，blockTensor
        // This tensor distribution used to construct both distributed tensor for local buffer store
        // and read. without buffer address info
        constexpr auto a_reg_block_dstr = Policy::template MakeARegBlockDescriptor<Problem>();

        // A Reg tensor for store, also used for block GEMM
        auto a_copy_reg_tensor = make_static_distributed_tensor<ADataType>(a_reg_block_dstr);
        store_tile(a_copy_reg_tensor, a_reg_block_tensor_tmp);

        // B tile in LDS, blockWindow
        BDataType* p_b_lds =
            static_cast<BDataType*>(static_cast<void*>(static_cast<char*>(p_smem)));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        // This tensor view used to construct both tile window for lds store and read, with buffer
        // address info
        auto b_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b_lds, b_lds_block_desc);

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             b_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.GetTileDistribution());

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(
            get_slice_tile(a_copy_reg_tensor, Sequence<0, 0>{}, Sequence<kMPerBlock, kKPerBlock>{}),
            b_lds_gemm_window)){};

        auto b_block_tile = load_tile(b_copy_dram_window);
        {
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            store_tile(b_copy_lds_window, b_block_tile);
            b_block_tile = load_tile(b_copy_dram_window);
        }
        if constexpr(k_loops > 2)
        {
            static_for<0, k_loops - 2, 1>{}([&](auto i_k0) {
                block_sync_lds();

                block_gemm(c_block_tile,
                           get_slice_tile(a_copy_reg_tensor,
                                          Sequence<0, (i_k0)*kKPerBlock>{},
                                          Sequence<kMPerBlock, (i_k0 + 1) * kKPerBlock>{}),
                           b_copy_lds_window);

                block_sync_lds();

                move_tile_window(b_copy_dram_window, {0, kKPerBlock});

                store_tile(b_copy_lds_window, b_block_tile);
                b_block_tile = load_tile(b_copy_dram_window);
            });
        }

        // tail
        {
            block_sync_lds();

            block_gemm(c_block_tile,
                       get_slice_tile(a_copy_reg_tensor,
                                      Sequence<0, (k_loops - 2) * kKPerBlock>{},
                                      Sequence<kMPerBlock, (k_loops - 1) * kKPerBlock>{}),
                       b_copy_lds_window);

            block_sync_lds();

            store_tile(b_copy_lds_window, b_block_tile);

            block_sync_lds();

            block_gemm(c_block_tile,
                       get_slice_tile(a_copy_reg_tensor,
                                      Sequence<0, (k_loops - 1) * kKPerBlock>{},
                                      Sequence<kMPerBlock, (k_loops)*kKPerBlock>{}),
                       b_copy_lds_window);
        }

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

} // namespace block
} // namespace tile_program
} // namespace ck
