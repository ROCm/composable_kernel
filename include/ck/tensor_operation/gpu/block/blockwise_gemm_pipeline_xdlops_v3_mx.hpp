// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_mx_pipeline_xdlops_base.hpp"

namespace ck {

// Naive pipeline with lowest resource request per WGP
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
          index_t ThreadBlockSize,
          index_t ScaleBlockSize,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat, // MXdlPerWave
          index_t NRepeat, // NXdlPerWave
          index_t KPack>
struct BlockwiseGemmXdlops_pipeline_v3_mx
{
};

template <index_t ThreadBlockSize,
          index_t ScaleBlockSize,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat, // MXdlPerWave
          index_t NRepeat, // NXdlPerWave
          index_t KPack>
struct BlockwiseGemmXdlops_pipeline_v3_mx<BlockGemmPipelineScheduler::Intrawave,
                                          ThreadBlockSize,
                                          ScaleBlockSize,
                                          ADataType,
                                          AScaleDataType,
                                          BDataType,
                                          BScaleDataType,
                                          ATileDesc,
                                          BTileDesc,
                                          AMmaTileDesc,
                                          BMmaTileDesc,
                                          ABlockTransferSrcScalarPerVector,
                                          BBlockTransferSrcScalarPerVector,
                                          MPerBlock,
                                          NPerBlock,
                                          KPerBlock,
                                          MPerXDL,
                                          NPerXDL,
                                          MRepeat,
                                          NRepeat,
                                          KPack>
    : BlockwiseGemmXdlops_mx_pipeline_base<ThreadBlockSize,
                                           ADataType,
                                           BDataType,
                                           ATileDesc,
                                           BTileDesc,
                                           AMmaTileDesc,
                                           BMmaTileDesc,
                                           ABlockTransferSrcScalarPerVector,
                                           BBlockTransferSrcScalarPerVector,
                                           MPerBlock,
                                           NPerBlock,
                                           KPerBlock,
                                           MPerXDL,
                                           NPerXDL,
                                           MRepeat,
                                           NRepeat,
                                           KPack>

{

    using Base = BlockwiseGemmXdlops_mx_pipeline_base<ThreadBlockSize,
                                                      ADataType,
                                                      BDataType,
                                                      ATileDesc,
                                                      BTileDesc,
                                                      AMmaTileDesc,
                                                      BMmaTileDesc,
                                                      ABlockTransferSrcScalarPerVector,
                                                      BBlockTransferSrcScalarPerVector,
                                                      MPerBlock,
                                                      NPerBlock,
                                                      KPerBlock,
                                                      MPerXDL,
                                                      NPerXDL,
                                                      MRepeat,
                                                      NRepeat,
                                                      KPack>;
    using Base::I0;
    using Base::I1;
    using Base::KRepeat;
    using Base::MWaves;
    using Base::NWaves;
    using Base::WaveSize;
    using Base::xdlops_gemm;
    using typename Base::HotLoopInstList;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetCThreadBuffer;
    using Base::GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetWaveIdx;
    using Base::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;

    using Base::a_block_desc_m0_m1_m2_k;
    using Base::b_block_desc_n0_n1_n2_k;

    using Base::AMmaKStride;
    using Base::BMmaKStride;
    using Base::KThreadChunk;

    using AccType      = typename Base::AccType;
    using Tuple4       = typename Base::Tuple4;
    using ComputeTypeA = typename Base::ComputeTypeA;
    using ComputeTypeB = typename Base::ComputeTypeB;

    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    static constexpr auto ScalesPerKBlockSize =
        KPerBlock / ScaleBlockSize; // How many mx-vectors per K block

    //> How many mx-vectors in each row/col is processed in one call to xdlops_gemm.Run()
    static constexpr auto ScalesPerXdlopsRun = (KPack * xdlops_gemm.K0PerXdlops) / ScaleBlockSize;

    //> How many scales a thread must read to accommodate one call to xdlops_gemm.Run()
    static constexpr auto ScalesPerXdlopsRunPerThread =
        ScalesPerXdlopsRun / xdlops_gemm.mfma_instr.num_input_blks;

    __host__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    __device__ static constexpr auto HotLoopScheduler()
    {
        // A/B split schedule
        // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
        constexpr auto num_ds_read_inst_a =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16
                ? HotLoopInstList::A_LDS_Read_Inst_Num
                : HotLoopInstList::A_LDS_Read_Inst_Num / 2;
        constexpr auto num_ds_read_inst_b =
            HotLoopInstList::B_LDS_Read_Width * sizeof(BDataType) == 16
                ? HotLoopInstList::B_LDS_Read_Inst_Num
                : HotLoopInstList::B_LDS_Read_Inst_Num / 2;

        constexpr auto num_ds_write_inst_a = HotLoopInstList::A_LDS_Write_Inst_Num;
        constexpr auto num_ds_write_inst_b = HotLoopInstList::B_LDS_Write_Inst_Num;

        constexpr auto num_buffer_load_inst_a = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_buffer_load_inst_b = HotLoopInstList::B_Buffer_Load_Inst_Num;

        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto mfma_cycle = HotLoopInstList::C_MFMA_Inst_Cycle;
        constexpr auto ds_read_a_issue_cycle =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
        constexpr auto ds_read_b_issue_cycle =
            HotLoopInstList::B_LDS_Read_Width * sizeof(BDataType) == 16 ? 8 : 4;
        constexpr auto ds_read_a_mfma_rate =
            (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
        constexpr auto ds_read_b_mfma_rate =
            (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

        constexpr auto num_dsread_a_mfma =
            (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
        constexpr auto num_dsread_b_mfma =
            (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

        // stage 1
        // Separate this part?
        // constexpr auto num_mfma_per_ds_read = sizeof(ComputeDataType) / sizeof(ADataType) >
        //                                               sizeof(ComputeDataType) / sizeof(BDataType)
        //                                           ? sizeof(ComputeDataType) / sizeof(ADataType)
        //                                           : sizeof(ComputeDataType) / sizeof(BDataType);
        constexpr auto num_mfma_stage1 = num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
        constexpr auto num_mfma_per_issue =
            num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
        constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
        constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

        static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_mfma_per_issue - num_dswrite_per_issue_a, 0); // MFMA
        });
        static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_mfma_per_issue - num_dswrite_per_issue_b, 0); // MFMA
        });

        // stage 2
        static_for<0, num_dsread_a_mfma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >=
                         ds_read_a_mfma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_a - (num_dsread_a_mfma - 1) *
                                                                              ds_read_a_mfma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });

        static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
                         ds_read_b_mfma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_b - (num_dsread_b_mfma - 1) *
                                                                              ds_read_b_mfma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });
    }

    template <bool HasMainLoop,
              TailNumber TailNum,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename CThreadBuffer,
              typename AScaleGridBuffer,
              typename AScaleGridDesc,
              typename AScaleThreadTransfer,
              typename BScaleGridBuffer,
              typename BScaleGridDesc,
              typename BScaleThreadTransfer>
    __device__ void Run(
        // ABlockCopy
        const AGridDesc& a_grid_desc,
        const ABlockDesc& a_block_desc,
        ABlockTransfer& a_blockwise_copy,
        const AGridBuffer& a_grid_buf,
        ABlockBuffer& a_block_buf,
        const ABlockTransferStep& a_block_copy_step,
        // BBlockCopy
        const BGridDesc& b_grid_desc,
        const BBlockDesc& b_block_desc,
        BBlockTransfer& b_blockwise_copy,
        const BGridBuffer& b_grid_buf,
        BBlockBuffer& b_block_buf,
        const BBlockTransferStep& b_block_copy_step,
        // CThread
        CThreadBuffer& c_thread_buf,
        // A and B scales
        const AScaleGridDesc& a_scale_grid_desc,
        AScaleThreadTransfer& a_scale_thread_copy,
        const AScaleGridBuffer& a_scale_grid_buf,
        const BScaleGridDesc& b_scale_grid_desc,
        BScaleThreadTransfer& b_scale_thread_copy,
        const BScaleGridBuffer& b_scale_grid_buf,
        index_t num_loop) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_thread_desc_.GetElementSpaceSize());

        auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
            a_scale_thread_desc.GetElementSpaceSize());

        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BScaleDataType>(
            b_scale_thread_desc.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(a_scale_thread_buf), Number<2>{}> a_scale_thread_bufs;
        StaticallyIndexedArray<decltype(b_scale_thread_buf), Number<2>{}> b_scale_thread_bufs;

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Prefetch a_scales
        static_for<0, MRepeat, 1>{}([&](auto m0) {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                    constexpr auto a_scale_offset =
                        a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, s));
                    auto a_scale_thread_buf_copy =
                        make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
                            a_scale_thread_desc_copy.GetElementSpaceSize());
                    a_scale_thread_copy.Run(a_scale_grid_desc,
                                            a_scale_grid_buf,
                                            a_scale_thread_desc_copy,
                                            make_tuple(I0, I0),
                                            a_scale_thread_buf_copy);

                    a_scale_thread_bufs(I0)(Number<a_scale_offset>{}) =
                        a_scale_thread_buf_copy[Number<0>{}];
                    a_scale_thread_copy.MoveSrcSliceWindow(
                        a_scale_grid_desc,
                        make_multi_index(0, xdlops_gemm.KPerXdlops / ScaleBlockSize));
                });
            });
            a_scale_thread_copy.MoveSrcSliceWindow(
                a_scale_grid_desc, make_multi_index(MWaves * MPerXDL, -ScalesPerKBlockSize));
        });

        // restore row id and advance to the next set of scales
        a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                               make_multi_index(-MPerBlock, ScalesPerKBlockSize));

        // Prefetch b_scales
        static_for<0, NRepeat, 1>{}([&](auto n0) {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                    constexpr auto b_scale_offset =
                        b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, s));
                    auto b_scale_thread_buf_copy =
                        make_static_buffer<AddressSpaceEnum::Vgpr, BScaleDataType>(
                            b_scale_thread_desc_copy.GetElementSpaceSize());
                    b_scale_thread_copy.Run(b_scale_grid_desc,
                                            b_scale_grid_buf,
                                            b_scale_thread_desc_copy,
                                            make_tuple(I0, I0),
                                            b_scale_thread_buf_copy);

                    b_scale_thread_bufs(I0)(Number<b_scale_offset>{}) =
                        b_scale_thread_buf_copy[Number<0>{}];
                    b_scale_thread_copy.MoveSrcSliceWindow(
                        b_scale_grid_desc,
                        make_multi_index(0, xdlops_gemm.KPerXdlops / ScaleBlockSize));
                });
            });
            b_scale_thread_copy.MoveSrcSliceWindow(
                b_scale_grid_desc, make_multi_index(NWaves * NPerXDL, -ScalesPerKBlockSize));
        });

        // restore col id and advance to the next set of scales
        // NWaves * NPerXDL * NRepeat == NPerBlock
        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                               make_multi_index(-NPerBlock, ScalesPerKBlockSize));

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // Global prefetch 2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Local prefetch 1
        block_sync_lds();
        static_for<0, KRepeat, 1>{}([&](auto k) {
            constexpr auto k_step = k * xdlops_gemm.KPerXdlops * (KPack / xdlops_gemm.K1PerXdlops);
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, xdlops_gemm.K1PerXdlops / KThreadChunk, 1>{}([&](auto chunk) {
                    constexpr auto a_k_step_chunk =
                        k_step + chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<a_k_step_chunk>{}),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, Number<chunk * KThreadChunk>{}),
                                       a_thread_buf);
                });
            });
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                // read block data in chunks to assemble correct thread vectors
                static_for<0, xdlops_gemm.K1PerXdlops / KThreadChunk, 1>{}([&](auto chunk) {
                    constexpr auto b_k_step_chunk =
                        k_step + chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<b_k_step_chunk>{}),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, Number<chunk * KThreadChunk>{}),
                                       b_thread_buf);
                });
            });
        });

        // Initialize C
        c_thread_buf.Clear();
        __builtin_amdgcn_sched_barrier(0);

        // main body
        if constexpr(HasMainLoop)
        {
            // loop over k with the step KPerBlock
            index_t i = 0;
            do
            {
                auto LoopFunc = [&](auto scale_comp_buf, auto scale_mem_buf) {
                    // Prefetch a_scales
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
                            static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                                constexpr auto a_scale_offset =
                                    a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, s));
                                auto a_scale_thread_buf_copy =
                                    make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
                                        a_scale_thread_desc_copy.GetElementSpaceSize());
                                a_scale_thread_copy.Run(a_scale_grid_desc,
                                                        a_scale_grid_buf,
                                                        a_scale_thread_desc_copy,
                                                        make_tuple(I0, I0),
                                                        a_scale_thread_buf_copy);

                                a_scale_thread_bufs(scale_mem_buf)(Number<a_scale_offset>{}) =
                                    a_scale_thread_buf_copy[Number<0>{}];
                                a_scale_thread_copy.MoveSrcSliceWindow(
                                    a_scale_grid_desc,
                                    make_multi_index(0, xdlops_gemm.KPerXdlops / ScaleBlockSize));
                            });
                        });
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc,
                            make_multi_index(MWaves * MPerXDL, -ScalesPerKBlockSize));
                    });

                    // restore row id and advance to the next set of scales
                    a_scale_thread_copy.MoveSrcSliceWindow(
                        a_scale_grid_desc, make_multi_index(-MPerBlock, ScalesPerKBlockSize));

                    // Prefetch b_scales
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
                            static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                                constexpr auto b_scale_offset =
                                    b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, s));
                                auto b_scale_thread_buf_copy =
                                    make_static_buffer<AddressSpaceEnum::Vgpr, BScaleDataType>(
                                        b_scale_thread_desc_copy.GetElementSpaceSize());
                                b_scale_thread_copy.Run(b_scale_grid_desc,
                                                        b_scale_grid_buf,
                                                        b_scale_thread_desc_copy,
                                                        make_tuple(I0, I0),
                                                        b_scale_thread_buf_copy);

                                b_scale_thread_bufs(scale_mem_buf)(Number<b_scale_offset>{}) =
                                    b_scale_thread_buf_copy[Number<0>{}];
                                b_scale_thread_copy.MoveSrcSliceWindow(
                                    b_scale_grid_desc,
                                    make_multi_index(0, xdlops_gemm.KPerXdlops / ScaleBlockSize));
                            });
                        });
                        b_scale_thread_copy.MoveSrcSliceWindow(
                            b_scale_grid_desc,
                            make_multi_index(NWaves * NPerXDL, -ScalesPerKBlockSize));
                    });
                    // restore col id and advance to the next set of scales
                    // NWaves * NPerXDL * NRepeat == NPerBlock
                    b_scale_thread_copy.MoveSrcSliceWindow(
                        b_scale_grid_desc, make_multi_index(-NPerBlock, ScalesPerKBlockSize));

                    // TODO: consider scheduling the scale load
                    // -------------------------------------------------------------------------------------------
                    __builtin_amdgcn_sched_barrier(0);
                    block_sync_lds();
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            static_for<0, KRepeat, 1>{}([&](auto k0) {
                                vector_type<ComputeTypeA, KPack> a_thread_vec; // = vec: pk_i4_t, 32
                                vector_type<ComputeTypeB, KPack> b_thread_vec;

                                static_for<0, KPack / 2, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(m0, I0, k0, ik))>{}];
                                    b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                            make_tuple(n0, I0, k0, ik))>{}];
                                });

                                constexpr index_t a_scale_offset =
                                    a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));
                                constexpr index_t b_scale_offset =
                                    b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                                static_assert(
                                    0 < ScalesPerXdlopsRunPerThread,
                                    "Must have at least one scale per Xdlops per Thread.");

                                vector_type<AScaleDataType, ScalesPerXdlopsRunPerThread>
                                    a_scale_thread_vec;
                                vector_type<BScaleDataType, ScalesPerXdlopsRunPerThread>
                                    b_scale_thread_vec;

                                // Pack scale_thread_buf into scale_thread_vec
                                static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                                    a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                        a_scale_thread_bufs(
                                            scale_comp_buf)[Number<a_scale_offset + s>{}];
                                    b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                        b_scale_thread_bufs(
                                            scale_comp_buf)[Number<b_scale_offset + s>{}];
                                });
                                // CK_PRINT<xdlops_gemm.K1PerXdlops>();
                                // CK_PRINT<decltype(xdlops_gemm)>();
                                using mfma_input_type_a =
                                    typename vector_type<ComputeTypeA,
                                                         xdlops_gemm.K1PerXdlops / 2>::type;
                                // mfma input type = pk_f4_t, 32
                                // CK_PRINT<mfma_input_type_a>();
                                using mfma_input_type_b =
                                    typename vector_type<ComputeTypeB,
                                                         xdlops_gemm.K1PerXdlops / 2>::type;

                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                                // MFMA accumulation
                                xdlops_gemm.template Run<>(
                                    a_thread_vec.template AsType<mfma_input_type_a>(),
                                    a_scale_thread_vec.template AsType<AScaleDataType>(),
                                    b_thread_vec.template AsType<mfma_input_type_b>(),
                                    b_scale_thread_vec.template AsType<BScaleDataType>(),
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                            });
                        });
                    });

                    // k indexes mapping to threads for 32x32x64:
                    // t0 : |0  --> 15 32 --> 47 | 64 --> 79 96  --> 111 | etc.
                    // t32: |16 --> 31 48 --> 63 | 80 --> 95 112 --> 127 | etc.
                    //              k = 0                 k = 1

                    //  k indexes mapping to threads for 16x16x128:
                    // t0 : |0  --> 15 64  --> 79 | 128 --> 143 192 --> 207| etc.
                    // t16: |16 --> 31 80  --> 95 | 144 --> 159 208 --> 223| etc.
                    // t32: |32 --> 47 96  --> 111| 160 --> 175 224 --> 239| etc.
                    // t48: |48 --> 63 112 --> 127| 176 --> 191 240 --> 255| etc.
                    //              k = 0                    k = 1
                    block_sync_lds();
                    static_for<0, KRepeat, 1>{}([&](auto k) {
                        constexpr auto k_step =
                            k * xdlops_gemm.KPerXdlops * (KPack / xdlops_gemm.K1PerXdlops);

                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            static_for<0, xdlops_gemm.K1PerXdlops / KThreadChunk, 1>{}(
                                [&](auto chunk) {
                                    constexpr auto a_k_step_chunk =
                                        k_step + chunk * KThreadChunk *
                                                     xdlops_gemm.mfma_instr.num_input_blks;
                                    a_thread_copy_.Run(
                                        a_block_desc_m0_m1_m2_k,
                                        make_tuple(m0, I0, I0, Number<a_k_step_chunk>{}),
                                        a_block_buf,
                                        a_thread_desc_,
                                        make_tuple(m0, I0, k, Number<chunk * KThreadChunk>{}),
                                        a_thread_buf);
                                });
                        });
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            // read block data in chunks to assemble correct thread vectors
                            static_for<0, xdlops_gemm.K1PerXdlops / KThreadChunk, 1>{}(
                                [&](auto chunk) {
                                    constexpr auto b_k_step_chunk =
                                        k_step + chunk * KThreadChunk *
                                                     xdlops_gemm.mfma_instr.num_input_blks;
                                    b_thread_copy_.Run(
                                        b_block_desc_n0_n1_n2_k,
                                        make_tuple(n0, I0, I0, Number<b_k_step_chunk>{}),
                                        b_block_buf,
                                        b_thread_desc_,
                                        make_tuple(n0, I0, k, Number<chunk * KThreadChunk>{}),
                                        b_thread_buf);
                                });
                        });
                    });

                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);
                };

                LoopFunc(I0, I1);
                LoopFunc(I1, I0);

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
            // Prefetch a_scales
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                        constexpr auto a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, s));
                        auto a_scale_thread_buf_copy =
                            make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
                                a_scale_thread_desc_copy.GetElementSpaceSize());
                        a_scale_thread_copy.Run(a_scale_grid_desc,
                                                a_scale_grid_buf,
                                                a_scale_thread_desc_copy,
                                                make_tuple(I0, I0),
                                                a_scale_thread_buf_copy);

                        a_scale_thread_bufs(I1)(Number<a_scale_offset>{}) =
                            a_scale_thread_buf_copy[Number<0>{}];
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc,
                            make_multi_index(0, xdlops_gemm.KPerXdlops / ScaleBlockSize));
                    });
                });
                a_scale_thread_copy.MoveSrcSliceWindow(
                    a_scale_grid_desc, make_multi_index(MWaves * MPerXDL, -ScalesPerKBlockSize));
            });

            // Prefetch b_scales
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                        constexpr auto b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, s));
                        auto b_scale_thread_buf_copy =
                            make_static_buffer<AddressSpaceEnum::Vgpr, BScaleDataType>(
                                b_scale_thread_desc_copy.GetElementSpaceSize());
                        b_scale_thread_copy.Run(b_scale_grid_desc,
                                                b_scale_grid_buf,
                                                b_scale_thread_desc_copy,
                                                make_tuple(I0, I0),
                                                b_scale_thread_buf_copy);

                        b_scale_thread_bufs(I1)(Number<b_scale_offset>{}) =
                            b_scale_thread_buf_copy[Number<0>{}];
                        b_scale_thread_copy.MoveSrcSliceWindow(
                            b_scale_grid_desc,
                            make_multi_index(0, xdlops_gemm.KPerXdlops / ScaleBlockSize));
                    });
                });
                b_scale_thread_copy.MoveSrcSliceWindow(
                    b_scale_grid_desc, make_multi_index(NWaves * NPerXDL, -ScalesPerKBlockSize));
            });

            block_sync_lds();
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeTypeA, KPack> a_thread_vec;
                        vector_type<ComputeTypeB, KPack> b_thread_vec;

                        static_for<0, KPack / 2, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        constexpr index_t a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));

                        constexpr index_t b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                        vector_type<AScaleDataType, ScalesPerXdlopsRunPerThread> a_scale_thread_vec;
                        vector_type<BScaleDataType, ScalesPerXdlopsRunPerThread> b_scale_thread_vec;

                        // Pack b_scale_thread_buf into b_scale_thread_vec
                        static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                            a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                a_scale_thread_bufs(I0)[Number<a_scale_offset + s>{}];
                            b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                b_scale_thread_bufs(I0)[Number<b_scale_offset + s>{}];
                        });

                        using mfma_input_type_a =
                            typename vector_type<ComputeTypeA, xdlops_gemm.K1PerXdlops / 2>::type;
                        using mfma_input_type_b =
                            typename vector_type<ComputeTypeB, xdlops_gemm.K1PerXdlops / 2>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        // MFMA accumulation
                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type_a>(),
                            a_scale_thread_vec.template AsType<AScaleDataType>(),
                            b_thread_vec.template AsType<mfma_input_type_b>(),
                            b_scale_thread_vec.template AsType<BScaleDataType>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                constexpr auto k_step =
                    k * xdlops_gemm.KPerXdlops * (KPack / xdlops_gemm.K1PerXdlops);

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, xdlops_gemm.K1PerXdlops / KThreadChunk, 1>{}([&](auto chunk) {
                        constexpr auto a_k_step_chunk =
                            k_step + chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<a_k_step_chunk>{}),
                                           a_block_buf,
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k, Number<chunk * KThreadChunk>{}),
                                           a_thread_buf);
                    });
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    // read block data in chunks to assemble correct thread vectors
                    static_for<0, xdlops_gemm.K1PerXdlops / KThreadChunk, 1>{}([&](auto chunk) {
                        constexpr auto b_k_step_chunk =
                            k_step + chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<b_k_step_chunk>{}),
                                           b_block_buf,
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, Number<chunk * KThreadChunk>{}),
                                           b_thread_buf);
                    });
                });
            });

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeTypeA, KPack> a_thread_vec;
                        vector_type<ComputeTypeB, KPack> b_thread_vec;

                        static_for<0, KPack / 2, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        constexpr index_t a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));

                        constexpr index_t b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                        vector_type<AScaleDataType, ScalesPerXdlopsRunPerThread> a_scale_thread_vec;
                        vector_type<BScaleDataType, ScalesPerXdlopsRunPerThread> b_scale_thread_vec;

                        // Pack b_scale_thread_buf into b_scale_thread_vec
                        static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                            a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                a_scale_thread_bufs(I1)[Number<a_scale_offset + s>{}];
                            b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                b_scale_thread_bufs(I1)[Number<b_scale_offset + s>{}];
                        });

                        using mfma_input_type_a =
                            typename vector_type<ComputeTypeA, xdlops_gemm.K1PerXdlops / 2>::type;
                        using mfma_input_type_b =
                            typename vector_type<ComputeTypeB, xdlops_gemm.K1PerXdlops / 2>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        // MFMA accumulation
                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type_a>(),
                            a_scale_thread_vec.template AsType<AScaleDataType>(),
                            b_thread_vec.template AsType<mfma_input_type_b>(),
                            b_scale_thread_vec.template AsType<BScaleDataType>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeTypeA, KPack> a_thread_vec;
                        vector_type<ComputeTypeB, KPack> b_thread_vec;

                        static_for<0, KPack / 2, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        constexpr index_t a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));

                        constexpr index_t b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                        vector_type<AScaleDataType, ScalesPerXdlopsRunPerThread> a_scale_thread_vec;
                        vector_type<BScaleDataType, ScalesPerXdlopsRunPerThread> b_scale_thread_vec;

                        // Pack b_scale_thread_buf into b_scale_thread_vec
                        static_for<0, ScalesPerXdlopsRunPerThread, 1>{}([&](auto s) {
                            a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                a_scale_thread_bufs(I0)[Number<a_scale_offset + s>{}];
                            b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                b_scale_thread_bufs(I0)[Number<b_scale_offset + s>{}];
                        });

                        using mfma_input_type_a =
                            typename vector_type<ComputeTypeA, xdlops_gemm.K1PerXdlops / 2>::type;
                        using mfma_input_type_b =
                            typename vector_type<ComputeTypeB, xdlops_gemm.K1PerXdlops / 2>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        // MFMA accumulation
                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type_a>(),
                            a_scale_thread_vec.template AsType<AScaleDataType>(),
                            b_thread_vec.template AsType<mfma_input_type_b>(),
                            b_scale_thread_vec.template AsType<BScaleDataType>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });
        }
    }

    // TODO: make this field protected when a_scale_thread_copy_ is moved
    // here
    static constexpr auto a_scale_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<KRepeat>{}, Number<ScalesPerXdlopsRunPerThread>{}));

    // Is used to copy data from a_scale_grid to a_scale_thread
    static constexpr auto a_scale_thread_desc_copy =
        make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}, Number<1>{}));

    // TODO: make this field protected when b_scale_thread_copy_ is moved
    // here
    static constexpr auto b_scale_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<NRepeat>{}, Number<KRepeat>{}, Number<ScalesPerXdlopsRunPerThread>{}));

    // Is used to copy data from b_scale_grid to b_scale_thread_buf
    static constexpr auto b_scale_thread_desc_copy =
        make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}, Number<1>{}));

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
};

} // namespace ck
