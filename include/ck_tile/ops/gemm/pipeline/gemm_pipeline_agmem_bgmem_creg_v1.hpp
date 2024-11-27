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

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return  integer_least_multiple(
                    sizeof(ADataType) *
                        Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                    16) * 2 +
                integer_least_multiple(
                    sizeof(BDataType) *
                        Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size(),
                    16) * 2;
    }

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
    
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // A tile in LDS
        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        constexpr index_t a_lds_block_space_size_aligned =
            integer_least_multiple(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16);
        constexpr index_t b_lds_block_space_size_aligned =
            integer_least_multiple(sizeof(BDataType) * b_lds_block_desc.get_element_space_size(), 16);
        ADataType* p_a_lds0 = reinterpret_cast<ADataType*>(p_smem);
        ADataType* p_a_lds1 = reinterpret_cast<ADataType*>(reinterpret_cast<char*>(p_smem) + a_lds_block_space_size_aligned);
        // B tile in LDS
        BDataType* p_b_lds0 = reinterpret_cast<BDataType*>(reinterpret_cast<char*>(p_smem) + a_lds_block_space_size_aligned * 2);
        BDataType* p_b_lds1 = reinterpret_cast<BDataType*>(reinterpret_cast<char*>(p_b_lds0) + b_lds_block_space_size_aligned);


        auto a_lds_block0 = make_tensor_view<address_space_enum::lds>(p_a_lds0, a_lds_block_desc);
        auto b_lds_block0 = make_tensor_view<address_space_enum::lds>(p_b_lds0, b_lds_block_desc);
        auto a_lds_block1 = make_tensor_view<address_space_enum::lds>(p_a_lds1, a_lds_block_desc);
        auto b_lds_block1 = make_tensor_view<address_space_enum::lds>(p_b_lds1, b_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store
        auto a_store_lds_window0 = make_tile_window(
            a_lds_block0, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto a_store_lds_window1 = make_tile_window(
            a_lds_block1, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_store_lds_window0 = make_tile_window(
            b_lds_block0, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto b_store_lds_window1 = make_tile_window(
            b_lds_block1, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // A LDS tile for block GEMM
        auto a_load_lds_window0 = make_tile_window(
            a_lds_block0, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto a_load_lds_window1 = make_tile_window(
            a_lds_block1, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B LDS tile for block GEMM
        auto b_load_lds_window0 = make_tile_window(
            b_lds_block0, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto b_load_lds_window1 = make_tile_window(
            b_lds_block1, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = Policy::template BlockGemm<Problem>::MakeCBlockTile();

        // a b register tile
        auto a_prefetch_tile0 = make_static_distributed_tensor<ADataType>(Policy::template BlockGemm<Problem>::MakeABlockDistribution());
        auto a_prefetch_tile1 = make_static_distributed_tensor<ADataType>(Policy::template BlockGemm<Problem>::MakeABlockDistribution());
        auto b_prefetch_tile0 = make_static_distributed_tensor<BDataType>(Policy::template BlockGemm<Problem>::MakeBBlockDistribution());
        auto b_prefetch_tile1 = make_static_distributed_tensor<BDataType>(Policy::template BlockGemm<Problem>::MakeBBlockDistribution());
        
        using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
        using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

        using ABlockTile =
            decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
        using BBlockTile =
            decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

        ABlockTile a_global_load_tile;
        BBlockTile b_global_load_tile;
        // prefetch
        // global read 0
        GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
        GlobalPrefetch(b_global_load_tile, b_copy_dram_window);

        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
        // LDS write 0
        LocalPrefill(a_store_lds_window0, a_global_load_tile, a_element_func);
        LocalPrefill(b_store_lds_window0, b_global_load_tile, b_element_func);
        
        block_sync_lds();
        // global read 1
        GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
        GlobalPrefetch(b_global_load_tile, b_copy_dram_window);
        // local prefetch 0
        Policy::template BlockGemm<Problem>::PrefetchLds(a_load_lds_window0, a_prefetch_tile0);
        Policy::template BlockGemm<Problem>::PrefetchLds(b_load_lds_window0, b_prefetch_tile0);

        // LDS write 1
        LocalPrefill(a_store_lds_window1, a_global_load_tile, a_element_func);
        LocalPrefill(b_store_lds_window1, b_global_load_tile, b_element_func);
        
        // global read 2
        GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
        GlobalPrefetch(b_global_load_tile, b_copy_dram_window);

        index_t iCounter = num_loop - 1;
        while(iCounter > 2)
        {
            // ping
            {
                block_sync_lds();

                Policy::template BlockGemm<Problem>::PrefetchLds(a_load_lds_window1, a_prefetch_tile1);
                Policy::template BlockGemm<Problem>::PrefetchLds(b_load_lds_window1, b_prefetch_tile1);
                LocalPrefill(a_store_lds_window0, a_global_load_tile, a_element_func);
                LocalPrefill(b_store_lds_window0, b_global_load_tile, b_element_func);
                GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
                GlobalPrefetch(b_global_load_tile, b_copy_dram_window);
                block_gemm(c_block_tile, a_prefetch_tile0, b_prefetch_tile0);

            }
            
            __builtin_amdgcn_sched_barrier(0);
            // pong
            {
                block_sync_lds();
                Policy::template BlockGemm<Problem>::PrefetchLds(a_load_lds_window0, a_prefetch_tile0);
                Policy::template BlockGemm<Problem>::PrefetchLds(b_load_lds_window0, b_prefetch_tile0);
                LocalPrefill(a_store_lds_window1, a_global_load_tile, a_element_func);
                LocalPrefill(b_store_lds_window1, b_global_load_tile, b_element_func);
                GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
                GlobalPrefetch(b_global_load_tile, b_copy_dram_window);
                block_gemm(c_block_tile, a_prefetch_tile1, b_prefetch_tile1);

            }
            
            iCounter -= 2;
        }

        //tail 3
        if (iCounter == 1) {
            // 3
            {
                block_sync_lds();

                Policy::template BlockGemm<Problem>::PrefetchLds(a_load_lds_window1, a_prefetch_tile1);
                Policy::template BlockGemm<Problem>::PrefetchLds(b_load_lds_window1, b_prefetch_tile1);
                LocalPrefill(a_store_lds_window0, a_global_load_tile, a_element_func);
                LocalPrefill(b_store_lds_window0, b_global_load_tile, b_element_func);
                block_gemm(c_block_tile, a_prefetch_tile0, b_prefetch_tile0);
                __builtin_amdgcn_sched_barrier(0);
            }
            // 2
            {
                block_sync_lds();
                Policy::template BlockGemm<Problem>::PrefetchLds(a_load_lds_window0, a_prefetch_tile0);
                Policy::template BlockGemm<Problem>::PrefetchLds(b_load_lds_window0, b_prefetch_tile0);
                block_gemm(c_block_tile, a_prefetch_tile1, b_prefetch_tile1);
                __builtin_amdgcn_sched_barrier(0);
            }
            //1
            {
                block_gemm(c_block_tile, a_prefetch_tile0, b_prefetch_tile0);
            }
        //tail 2
        } else {
            {
                block_sync_lds();
                Policy::template BlockGemm<Problem>::PrefetchLds(a_load_lds_window1, a_prefetch_tile1);
                Policy::template BlockGemm<Problem>::PrefetchLds(b_load_lds_window1, b_prefetch_tile1);
                block_gemm(c_block_tile, a_prefetch_tile0, b_prefetch_tile0);
                __builtin_amdgcn_sched_barrier(0);
            }
            // 2
            {
                block_gemm(c_block_tile, a_prefetch_tile1, b_prefetch_tile1);
            }
        }
        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
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

    // __device__ static constexpr auto HotLoopScheduler()
    // {
    //     // schedule
    //     constexpr auto num_ds_read_inst =
    //         HotLoopInstList::A_LDS_Read_Inst_Num + HotLoopInstList::B_LDS_Read_Inst_Num;
    //     constexpr auto num_ds_write_inst =
    //         HotLoopInstList::A_LDS_Write_Inst_Num + HotLoopInstList::B_LDS_Write_Inst_Num;
    //     ;
    //     constexpr auto num_buffer_load_inst =
    //         HotLoopInstList::A_Buffer_Load_Inst_Num + HotLoopInstList::B_Buffer_Load_Inst_Num;
    //     ;
    //     constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

    //     constexpr auto num_issue = num_buffer_load_inst;

    //     static_for<0, num_issue, 1>{}([&](auto i) {
    //         ignore = i;
    //         __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
    //         __builtin_amdgcn_sched_group_barrier(
    //             0x100, num_ds_read_inst / num_buffer_load_inst, 0); // DS read
    //         __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);      // MFMA
    //         __builtin_amdgcn_sched_group_barrier(
    //             0x200, num_ds_write_inst / num_buffer_load_inst, 0); // DS write
    //         __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);       // MFMA
    //         __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);       // VMEM read
    //         __builtin_amdgcn_sched_group_barrier(
    //             0x008, num_mfma_inst / num_buffer_load_inst - 3, 0); // MFMA
    //     });
    // }
    
    // CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
    // {
    //     constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(number<0>{});
    //     constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(number<1>{});
    //     constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(number<2>{});

    //     constexpr index_t WaveSize = 64;
    //     constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(number<0>{});
    //     constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(number<1>{});

    //     constexpr index_t A_LDS_Read_Width = KPerXDL;
    //     constexpr index_t B_LDS_Read_Width = KPerXDL;

    //     constexpr index_t A_Buffer_Load_Inst_Num =
    //         MPerBlock * KPerBlock / (BlockSize * VectorSizeA);
    //     constexpr index_t B_Buffer_Load_Inst_Num =
    //         NPerBlock * KPerBlock / (BlockSize * VectorSizeB);

    //     constexpr index_t A_LDS_Write_Inst_Num = MPerBlock * KPerBlock / (BlockSize * KPerXDL);
    //     constexpr index_t B_LDS_Write_Inst_Num = NPerBlock * KPerBlock / (BlockSize * KPerXDL);

    //     constexpr index_t A_LDS_Read_Inst_Num =
    //         WaveNumN * MPerBlock * KPerBlock / (BlockSize * KPerXDL);
    //     constexpr index_t B_LDS_Read_Inst_Num =
    //         WaveNumM * MPerBlock * KPerBlock / (BlockSize * KPerXDL);

    //     constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
    //                                         (BlockSize / WaveSize) /
    //                                         (MPerXDL * NPerXDL * KPerXDL);

    //     // A/B split schedule
    //     // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
    //     constexpr auto num_ds_read_inst_a = A_LDS_Read_Width * sizeof(ADataType) == 16
    //                                             ? A_LDS_Read_Inst_Num
    //                                             : A_LDS_Read_Inst_Num / 2;
    //     constexpr auto num_ds_read_inst_b = B_LDS_Read_Width * sizeof(BDataType) == 16
    //                                             ? B_LDS_Read_Inst_Num
    //                                             : B_LDS_Read_Inst_Num / 2;

    //     constexpr auto num_ds_write_inst_a = A_LDS_Write_Inst_Num;
    //     constexpr auto num_ds_write_inst_b = B_LDS_Write_Inst_Num;

    //     constexpr auto num_buffer_load_inst_a = A_Buffer_Load_Inst_Num;
    //     constexpr auto num_buffer_load_inst_b = B_Buffer_Load_Inst_Num;

    //     constexpr auto num_mfma_inst = C_MFMA_Inst_Num;

    //     constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
    //     constexpr auto ds_read_a_issue_cycle =
    //         A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
    //     constexpr auto ds_read_b_issue_cycle =
    //         B_LDS_Read_Width * sizeof(BDataType) == 16 ? 8 : 4;
    //     constexpr auto ds_read_a_mfma_rate =
    //         (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
    //     constexpr auto ds_read_b_mfma_rate =
    //         (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

    //     constexpr auto num_dsread_a_mfma =
    //         (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
    //     constexpr auto num_dsread_b_mfma =
    //         (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

    //     // stage 1
    //     // Separate this part?
    //     // constexpr auto num_mfma_per_ds_read = sizeof(ComputeDataType) / sizeof(ADataType) >
    //     //                                               sizeof(ComputeDataType) /
    //     //                                               sizeof(BDataType)
    //     //                                           ? sizeof(ComputeDataType) /
    //     //                                           sizeof(ADataType) : sizeof(ComputeDataType)
    //     //                                           / sizeof(BDataType);
    //     constexpr auto num_mfma_stage1 =
    //         num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
    //     constexpr auto num_mfma_per_issue =
    //         num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
    //     constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
    //     constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

    //     static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
    //         ignore = i;
    //         static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
    //             ignore = idswrite;
    //             __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
    //             __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
    //         });
    //         __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
    //         __builtin_amdgcn_sched_group_barrier(
    //             0x008, num_mfma_per_issue - num_dswrite_per_issue_a, 0); // MFMA
    //     });
    //     static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
    //         ignore = i;
    //         static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
    //             ignore = idswrite;
    //             __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
    //             __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
    //         });
    //         __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
    //         __builtin_amdgcn_sched_group_barrier(
    //             0x008, num_mfma_per_issue - num_dswrite_per_issue_b, 0); // MFMA
    //     });

    //     // stage 2
    //     static_for<0, num_dsread_a_mfma, 1>{}([&](auto i) {
    //         if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >=
    //                         ds_read_a_mfma_rate)
    //         {
    //             __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
    //         }
    //         else
    //         {
    //             __builtin_amdgcn_sched_group_barrier(
    //                 0x100,
    //                 num_ds_read_inst_a - (num_dsread_a_mfma - 1) * ds_read_a_mfma_rate,
    //                 0); // DS read
    //         }
    //         __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
    //     });

    //     static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
    //         if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
    //                         ds_read_b_mfma_rate)
    //         {
    //             __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
    //         }
    //         else
    //         {
    //             __builtin_amdgcn_sched_group_barrier(
    //                 0x100,
    //                 num_ds_read_inst_b - (num_dsread_b_mfma - 1) * ds_read_b_mfma_rate,
    //                 0); // DS read
    //         }
    //         __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
    //     });
    // }

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
