// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy>
struct GemmPipelineAGmemBGmemCRegV1
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t VectorSizeA = Problem::VectorSizeA;
    static constexpr index_t VectorSizeB = Problem::VectorSizeB;
    static constexpr index_t VectorSizeC = Problem::VectorSizeC;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;
    static constexpr bool kHasHotLoop = Problem::kHasHotLoop;
    static constexpr auto kTailNum    = Problem::kTailNum;

    // CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    // {
    //     return  integer_least_multiple(
    //                 sizeof(ADataType) *
    //                     Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
    //                 16) * 2 +
    //             integer_least_multiple(
    //                 sizeof(BDataType) *
    //                     Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size(),
    //                 16) * 2;
    // }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }
    
    template <typename DstBlockTile, typename SrcTileWindow>
    CK_TILE_DEVICE void GlobalPrefetch(DstBlockTile& dst_block_tile,
                                        SrcTileWindow& dram_tile_window) const
    {
        load_tile(dst_block_tile, dram_tile_window);
        move_tile_window(dram_tile_window, {0, kKPerBlock});
    }

    template <typename DstTileWindow, typename SrcBlockTile, typename ElementFunction>
    CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                        const SrcBlockTile& src_block_tile,
                                        const ElementFunction& element_func) const
    {
        const auto block_tile_tmp = tile_elementwise_in(element_func, src_block_tile);
        store_tile(lds_tile_window, block_tile_tmp);
    }
    
    CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
    {
        // schedule
        // constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(number<0>{});//32
        // constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(number<1>{});//32
        // constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(number<2>{});//8

        // constexpr index_t WaveSize = 64;
        // constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(number<0>{});//2
        // constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(number<1>{});//2

        // constexpr index_t A_LDS_Read_Width = KPerXDL;//8
        // constexpr index_t B_LDS_Read_Width = KPerXDL;//8

        // constexpr index_t num_buffer_load_inst_a =
        //     kMPerBlock * kKPerBlock / (BlockSize * VectorSizeA); // 4
        // constexpr index_t num_buffer_load_inst_b =
        //     kNPerBlock * kKPerBlock / (BlockSize * VectorSizeB); // 4

        // constexpr index_t num_ds_write_inst_a = kMPerBlock * kKPerBlock / (BlockSize * KPerXDL); // 4
        // constexpr index_t num_ds_write_inst_b = kNPerBlock * kKPerBlock / (BlockSize * KPerXDL); // 4

        // constexpr index_t A_LDS_Read_Inst_Num =
        //     WaveNumN * kMPerBlock * kKPerBlock / (BlockSize * KPerXDL); // 8
        // constexpr index_t B_LDS_Read_Inst_Num =
        //     WaveNumM * kMPerBlock * kKPerBlock / (BlockSize * KPerXDL); // 8

        // constexpr index_t num_mfma_inst = kMPerBlock * kNPerBlock * kKPerBlock /
        //                                     (BlockSize / WaveSize) /
        //                                     (MPerXDL * NPerXDL * KPerXDL); // 64

        // // A/B split schedule
        // // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
        // constexpr auto num_ds_read_inst_a = A_LDS_Read_Width * sizeof(ADataType) == 16
        //                                         ? A_LDS_Read_Inst_Num
        //                                         : A_LDS_Read_Inst_Num / 2;
        // constexpr auto num_ds_read_inst_b = B_LDS_Read_Width * sizeof(BDataType) == 16
        //                                         ? B_LDS_Read_Inst_Num
        //                                         : B_LDS_Read_Inst_Num / 2;

        // constexpr auto num_ds_read_inst = num_ds_read_inst_a + num_ds_read_inst_b; // 16
        // constexpr auto num_ds_write_inst = num_ds_write_inst_a + num_ds_write_inst_b; //8
        // constexpr auto num_buffer_load_inst = num_buffer_load_inst_a + num_buffer_load_inst_b; //8

        // constexpr auto num_issue = num_buffer_load_inst; // 8

        // static_for<0, num_issue, 1>{}([&](auto i) {
        //     ignore = i;
        //     __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA : 1
        //     __builtin_amdgcn_sched_group_barrier(
        //         0x100, num_ds_read_inst / num_issue, 0); // DS read : 2
        //     __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);      // MFMA: 1
        //     __builtin_amdgcn_sched_group_barrier(
        //         0x200, num_ds_write_inst / num_issue, 0); // DS write : 1
        //     __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);       // MFMA : 1
        //     __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);       // VMEM read :1
        //     __builtin_amdgcn_sched_group_barrier(
        //         0x008, num_mfma_inst / num_issue - 3, 0); // MFMA : 5
        // });
        // __builtin_amdgcn_sched_barrier(0);
        
        static_for<0, 8, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA : 1
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read : 2
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA: 1
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write : 1
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA : 1
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read :1
            __builtin_amdgcn_sched_group_barrier(0x008, 5, 0); // MFMA : 5
        });
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockSubTile() {
        return Policy::template BlockGemm<Problem>::MakeCBlockSubTile();
    }

    CK_TILE_DEVICE static constexpr auto NumCSubTile() {
        constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(number<0>{});//32
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(number<0>{});//2
        return integer_divide_ceil(kMPerBlock, WaveNumM * MPerXDL);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* __restrict__ p_smem_0,
                                        void* __restrict__ p_smem_1) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");
        ////////////// global window & register /////////////////
        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window_linear(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window_linear(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // A register tile for global load
        constexpr auto ABlockTileDistr = a_copy_dram_window.get_tile_distribution();
        constexpr auto BBlockTileDistr = b_copy_dram_window.get_tile_distribution();
        using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr));
        using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr));
        ABlockTile a_global_load_tile;
        BBlockTile b_global_load_tile;

        // global prefetch 0
        // global read 0
        GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
        GlobalPrefetch(b_global_load_tile, b_copy_dram_window);
        ////////////// LDS desc, window & register /////////////////
        // AB LDS desc
        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        constexpr index_t a_lds_block_space_size_aligned =
            integer_least_multiple(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16);
        // A tile in LDS view
        const ADataType*__restrict__  p_a_lds0 = reinterpret_cast<ADataType*>(p_smem_0);
        const ADataType*__restrict__  p_a_lds1 = reinterpret_cast<ADataType*>(p_smem_1);
        auto a_lds_block0 = make_tensor_view<address_space_enum::lds>(p_a_lds0, a_lds_block_desc);
        auto a_lds_block1 = make_tensor_view<address_space_enum::lds>(p_a_lds1, a_lds_block_desc);
        auto a_lds_ld_block0 = make_tensor_view<address_space_enum::lds>(p_a_lds0, a_lds_block_desc);
        auto a_lds_ld_block1 = make_tensor_view<address_space_enum::lds>(p_a_lds1, a_lds_block_desc);
        // B tile in LDS view
        const BDataType*__restrict__  p_b_lds0 = reinterpret_cast<BDataType*>(reinterpret_cast<char*>(p_smem_0) + a_lds_block_space_size_aligned);
        const BDataType*__restrict__  p_b_lds1 = reinterpret_cast<BDataType*>(reinterpret_cast<char*>(p_smem_1) + a_lds_block_space_size_aligned);
        auto b_lds_block0 = make_tensor_view<address_space_enum::lds>(p_b_lds0, b_lds_block_desc);
        auto b_lds_block1 = make_tensor_view<address_space_enum::lds>(p_b_lds1, b_lds_block_desc);
        auto b_lds_ld_block0 = make_tensor_view<address_space_enum::lds>(p_b_lds0, b_lds_block_desc);
        auto b_lds_ld_block1 = make_tensor_view<address_space_enum::lds>(p_b_lds1, b_lds_block_desc);

        // A LDS tile window for store
        auto a_lds_window0 = make_tile_window_linear(
            a_lds_block0, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0}, ABlockTileDistr);
        auto a_lds_window1 = make_tile_window_linear(
            a_lds_block1, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0}, ABlockTileDistr);

        // B LDS tile window for store
        auto b_lds_window0 = make_tile_window_linear(
            b_lds_block0, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0}, BBlockTileDistr);
        auto b_lds_window1 = make_tile_window_linear(
            b_lds_block1, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0}, BBlockTileDistr);

        // Block GEMM
        auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = Policy::template BlockGemm<Problem>::MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);


        // LDS write 0
        LocalPrefill(a_lds_window0, a_global_load_tile, a_element_func);
        LocalPrefill(b_lds_window0, b_global_load_tile, b_element_func);
        // global read 1
        GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
        GlobalPrefetch(b_global_load_tile, b_copy_dram_window);

        block_sync_lds();
        // local prefetch 0
        // a b register tile for lds prefetch & mfma
        
        constexpr auto ALdsTileDistr = decltype(Policy::template BlockGemm<Problem>::MakeABlockDistribution()){};
        constexpr auto BLdsTileDistr = decltype(Policy::template BlockGemm<Problem>::MakeBBlockDistribution()){};
        using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));
        ALdsTile a_block_tile0;
        BLdsTile b_block_tile0;
        auto a_lds_ld_window0 = make_tile_window_linear(a_lds_ld_block0, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0}, ALdsTileDistr);
        auto a_lds_ld_window1 = make_tile_window_linear(a_lds_ld_block1, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0}, ALdsTileDistr);
        auto b_lds_ld_window0 = make_tile_window_linear(b_lds_ld_block0, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0}, BLdsTileDistr);
        auto b_lds_ld_window1 = make_tile_window_linear(b_lds_ld_block1, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0}, BLdsTileDistr);
    
        load_tile(a_block_tile0, a_lds_ld_window0);
        load_tile(b_block_tile0, b_lds_ld_window0);

        // LDS write 1
        LocalPrefill(a_lds_window1, a_global_load_tile, a_element_func);
        LocalPrefill(b_lds_window1, b_global_load_tile, b_element_func);
        
        // global read 2
        GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
        GlobalPrefetch(b_global_load_tile, b_copy_dram_window);

        index_t iCounter = __builtin_amdgcn_readfirstlane(num_loop - 2);

        ALdsTile a_block_tile1;
        BLdsTile b_block_tile1;
        if (kHasHotLoop) {
            do
            {
                // ping
                {
                    block_sync_lds();
                    //prefetch lds -> vgpr
                    load_tile(a_block_tile1, a_lds_ld_window1);
                    load_tile(b_block_tile1, b_lds_ld_window1);
                    //prefill -> lds
                    LocalPrefill(a_lds_window0, a_global_load_tile, a_element_func);
                    LocalPrefill(b_lds_window0, b_global_load_tile, b_element_func);
                    //prefill global -> vgpr
                    // GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
                    // GlobalPrefetch(b_global_load_tile, b_copy_dram_window);
                    load_tile(a_global_load_tile, a_copy_dram_window);
                    load_tile(b_global_load_tile, b_copy_dram_window);
                    // gemm
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                    move_tile_window(a_copy_dram_window, {0, kKPerBlock});
                    move_tile_window(b_copy_dram_window, {0, kKPerBlock});
                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);
                }
                // pong
                {
                    block_sync_lds();
                    load_tile(a_block_tile0, a_lds_ld_window0);
                    load_tile(b_block_tile0, b_lds_ld_window0);
                    LocalPrefill(a_lds_window1, a_global_load_tile, a_element_func);
                    LocalPrefill(b_lds_window1, b_global_load_tile, b_element_func);
                    GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
                    GlobalPrefetch(b_global_load_tile, b_copy_dram_window);
                    block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);
                }
                iCounter -= 2;
            }while(iCounter > 1);
        }

        //tail 3
        if (kTailNum == 3) {
            // 3
            {
                block_sync_lds();
                // Policy::template BlockGemm<Problem>::PrefetchLds(a_lds_window1, a_block_tile1);
                load_tile(a_block_tile1, a_lds_ld_window1);
                load_tile(b_block_tile1, b_lds_ld_window1);
                LocalPrefill(a_lds_window0, a_global_load_tile, a_element_func);
                LocalPrefill(b_lds_window0, b_global_load_tile, b_element_func);
                block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
            }
            // 2
            {
                block_sync_lds();
                // Policy::template BlockGemm<Problem>::PrefetchLds(a_lds_window0, a_block_tile0);
                load_tile(a_block_tile0, a_lds_ld_window0);
                load_tile(b_block_tile0, b_lds_ld_window0);
                block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
            }
            //1
            {
                block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
            }
        } 
        else 
        {
            // //tail 2
            {
                block_sync_lds();
                // Policy::template BlockGemm<Problem>::PrefetchLds(a_lds_window1, a_block_tile1);
                load_tile(a_block_tile1, a_lds_ld_window1);
                load_tile(b_block_tile1, b_lds_ld_window1);
                block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                static_for<0, 8, 1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(
                        0x008, 8, 0); // MFMA
                });
                __builtin_amdgcn_sched_barrier(0);
            }
            // 2
            {
                block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                __builtin_amdgcn_sched_group_barrier(0x008, 64, 0); // MFMA
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        
        /// cccccccccc
        // constexpr auto c_lds_block_desc = Policy::template MakeCLdsBlockDescriptor<Problem>();
        // auto c_lds_block = make_tensor_view<address_space_enum::lds>(reinterpret_cast<CDataType*>(p_smem), c_lds_block_desc);
        // auto c_lds_window0 = make_tile_window(c_lds_block, make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}), {0, 0});
        // store_tile(c_lds_window0, c_block_tile);
        // block_sync_lds();

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void*__restrict__ p_smem_0,
                                   void*__restrict__ p_smem_1) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem_0,
            p_smem_1);
    }
};

        // if (threadIdx.x == 0) {
        //     constexpr auto span_2d = decltype(a_global_load_tile)::get_distributed_spans();
        //     sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
        //         sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
        //             constexpr auto i_j_idx = make_tuple(idx0, idx1);
        //             printf("%f,", type_convert<float>(a_global_load_tile(i_j_idx)));
        //         });
        //         printf("\n");
        //     });
        // }
        // if (threadIdx.x == 0) {
            // constexpr auto span_2d = decltype(c_block_tile)::get_distributed_spans();
            // sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
            //     sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
            //         constexpr auto i_j_idx = make_tuple(idx0, idx1);
            //         if(abs(type_convert<float>(c_block_tile(i_j_idx)) - 32) > 0.1)
            //             printf("%d %f,", threadIdx.x, type_convert<float>(c_block_tile(i_j_idx)));
            //     });
            //     printf("\n");
            // });
        // }
} // namespace ck_tile
