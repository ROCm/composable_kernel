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
struct BlockwiseGemmXdlops_pipeline_bpreshuffle_mx_moe_v3
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
struct BlockwiseGemmXdlops_pipeline_bpreshuffle_mx_moe_v3<BlockGemmPipelineScheduler::Intrawave,
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
    using Base::I2;
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

    using Base::a_block_desc_m0_m1_m2_m3_k;
    using Base::b_block_desc_n0_n1_n2_n3_k;

    using Base::AMmaKStride;
    using Base::BMmaKStride;
    using Base::KThreadChunk;

    using Base::KXdlPack;
    using Base::MXdlPack;
    using Base::NXdlPack;

    using Base::APackedSize;
    using Base::BPackedSize;

    using AccType      = typename Base::AccType;
    using Tuple5       = typename Base::Tuple5;
    using ComputeTypeA = typename Base::ComputeTypeA;
    using ComputeTypeB = typename Base::ComputeTypeB;

    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    template <typename TileDesc_M0_M1_M2_M3_K>
    __host__ __device__ static constexpr auto
    MakeAGemmMmaTileDescriptor(const TileDesc_M0_M1_M2_M3_K&)
    {
        constexpr index_t M0 = TileDesc_M0_M1_M2_M3_K{}.GetLength(Number<0>{});
        constexpr index_t M1 = TileDesc_M0_M1_M2_M3_K{}.GetLength(Number<1>{});
        constexpr index_t M2 = TileDesc_M0_M1_M2_M3_K{}.GetLength(Number<2>{});
        constexpr index_t M3 = TileDesc_M0_M1_M2_M3_K{}.GetLength(Number<3>{});
        constexpr index_t K2 = KPack;
        constexpr index_t K1 = 64 / NPerXDL;
        constexpr index_t K0 = KRepeat;

        return transform_tensor_descriptor(
            TileDesc_M0_M1_M2_M3_K{},
            make_tuple(
                make_pass_through_transform(Number<M0>{}),
                make_pass_through_transform(Number<M1>{}),
                make_pass_through_transform(Number<M2>{}),
                make_pass_through_transform(Number<M3>{}),
                make_unmerge_transform(make_tuple(Number<K0>{}, Number<K1>{}, Number<K2>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4, 5, 6>{}));
    }

    static constexpr auto a_block_desc_m0_m1_m2_m3_k0_k1_k2 =
        MakeAGemmMmaTileDescriptor(a_block_desc_m0_m1_m2_m3_k);

    static constexpr auto ScalesPerKBlockSize =
        KPerBlock / ScaleBlockSize; // How many mx-vectors per K block

    //> How many mx-vectors in each row/col is processed in one call to xdlops_gemm.Run()
    static constexpr auto ScalesPerXdlopsRun =
        (APackedSize * KPack * xdlops_gemm.K0PerXdlops) / ScaleBlockSize;

    //> How many scales a thread must read to accommodate one call to xdlops_gemm.Run()
    static constexpr auto ScalesPerXdlopsRunPerThread =
        ScalesPerXdlopsRun / xdlops_gemm.mfma_instr.num_input_blks;

    using mx_scale_t                        = e8m0_bexp_t;
    static constexpr auto scale_pack_size_a = sizeof(AScaleDataType) / sizeof(mx_scale_t);
    static constexpr auto scale_pack_size_b = sizeof(BScaleDataType) / sizeof(mx_scale_t);
    static_assert(KXdlPack * MXdlPack % scale_pack_size_a == 0,
                  "A scale pack data type too large!");
    static_assert(KXdlPack * NXdlPack % scale_pack_size_b == 0,
                  "B scale pack data type too large!");
    static constexpr auto a_scale_thread_vec_size = KXdlPack * MXdlPack / scale_pack_size_a;
    static constexpr auto b_scale_thread_vec_size = KXdlPack * NXdlPack / scale_pack_size_b;

    __host__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
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

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
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
        ignore = b_block_desc;
        ignore = b_block_buf;

        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_thread_desc_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0);

        auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
            a_scale_thread_desc.GetElementSpaceSize());
        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BScaleDataType>(
            b_scale_thread_desc.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(a_scale_thread_buf), Number<2>{}> a_scale_thread_bufs;
        StaticallyIndexedArray<decltype(b_scale_thread_buf), Number<2>{}> b_scale_thread_bufs;

        // Global prefetch B1
        b_blockwise_copy.Run(b_grid_desc,
                             b_grid_buf,
                             b_block_desc_n0_n1_n2_k0_k1,
                             b_block_origin_idx,
                             b_thread_bufs(I0));
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Global prefetch A1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        // Prefetch a_scales to buf 0
        static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
            static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                a_scale_thread_copy.Run(a_scale_grid_desc,
                                        a_scale_grid_buf,
                                        a_scale_thread_desc,
                                        make_tuple(m0, k0, I0),
                                        a_scale_thread_bufs(I0));

                a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                       make_multi_index(0, I1, 0));
            });
            a_scale_thread_copy.MoveSrcSliceWindow(
                a_scale_grid_desc, make_multi_index(MWaves, -KRepeat / KXdlPack, 0));
        });

        // restore row id and advance to the next set of scales
        a_scale_thread_copy.MoveSrcSliceWindow(
            a_scale_grid_desc,
            make_multi_index(-MWaves * MRepeat / MXdlPack, KRepeat / KXdlPack, 0));

        // Prefetch b_scales 1
        static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
            static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                b_scale_thread_copy.Run(b_scale_grid_desc,
                                        b_scale_grid_buf,
                                        b_scale_thread_desc,
                                        make_tuple(n0, k0, I0),
                                        b_scale_thread_bufs(I0));

                b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                       make_multi_index(0, I1, 0));
            });
            b_scale_thread_copy.MoveSrcSliceWindow(
                b_scale_grid_desc, make_multi_index(NWaves, -KRepeat / KXdlPack, 0));
        });

        // restore col id and advance to the next set of scales
        // NWaves * NPerXDL * NRepeat == NPerBlock
        b_scale_thread_copy.MoveSrcSliceWindow(
            b_scale_grid_desc,
            make_multi_index(-NWaves * NRepeat / NXdlPack, KRepeat / KXdlPack, 0));

        // Local prefill A1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I0)); // vmem->vgpr-> lds0

        // Global prefetch A2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        // Local prefetch A1
        block_sync_lds();
        static_for<0, KRepeat, 1>{}([&](auto k) {
            constexpr auto k_step = k * xdlops_gemm.KPerXdlops / APackedSize *
                                    (APackedSize * KPack / xdlops_gemm.K1PerXdlops);
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, xdlops_gemm.K1PerXdlops / (APackedSize * KThreadChunk), 1>{}(
                    [&](auto chunk) {
                        constexpr auto a_k_step_chunk =
                            k_step + chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_m3_k,
                                           make_tuple(Number<m0 / MXdlPack>{},
                                                      I0,
                                                      Number<m0 % MXdlPack>{},
                                                      I0,
                                                      Number<a_k_step_chunk>{}),
                                           a_block_buf.At(I0),
                                           a_thread_desc_,
                                           make_tuple(Number<m0 / MXdlPack>{},
                                                      I0,
                                                      Number<m0 % MXdlPack>{},
                                                      k,
                                                      Number<chunk * KThreadChunk>{}),
                                           a_thread_buf);
                    });
            });
        });

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            // loop over k with the step KPerBlock
            index_t i = 0;
            do
            {
                auto LoopFunc = [&](auto scale_comp_buf, auto scale_mem_buf) {
                    // Prefetch a_scales to buf 1
                    static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                        static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                            a_scale_thread_copy.Run(a_scale_grid_desc,
                                                    a_scale_grid_buf,
                                                    a_scale_thread_desc,
                                                    make_tuple(m0, k0, I0),
                                                    a_scale_thread_bufs(scale_mem_buf));

                            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                                   make_multi_index(0, I1, 0));
                        });
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc, make_multi_index(MWaves, -KRepeat / KXdlPack, 0));
                    });

                    // restore row id and advance to the next set of scales
                    a_scale_thread_copy.MoveSrcSliceWindow(
                        a_scale_grid_desc,
                        make_multi_index(-MWaves * MRepeat / MXdlPack, KRepeat / KXdlPack, 0));

                    // Prefetch b_scales 1
                    static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                        static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                            b_scale_thread_copy.Run(b_scale_grid_desc,
                                                    b_scale_grid_buf,
                                                    b_scale_thread_desc,
                                                    make_tuple(n0, k0, I0),
                                                    b_scale_thread_bufs(scale_mem_buf));

                            b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                                   make_multi_index(0, I1, 0));
                        });
                        b_scale_thread_copy.MoveSrcSliceWindow(
                            b_scale_grid_desc, make_multi_index(NWaves, -KRepeat / KXdlPack, 0));
                    });

                    // restore col id and advance to the next set of scales
                    // NWaves * NPerXDL * NRepeat == NPerBlock
                    b_scale_thread_copy.MoveSrcSliceWindow(
                        b_scale_grid_desc,
                        make_multi_index(-NWaves * NRepeat / NXdlPack, KRepeat / KXdlPack, 0));

                    // Local prefill A2
                    block_sync_lds();
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(scale_mem_buf));

                    // Global prefetch A1
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

                    // Global prefetch B2
                    b_blockwise_copy.Run(b_grid_desc,
                                         b_grid_buf,
                                         b_block_desc_n0_n1_n2_k0_k1,
                                         b_block_origin_idx,
                                         b_thread_bufs(scale_mem_buf));
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    // A1 * B1
                    static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                        static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                            static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                                constexpr index_t a_scale_offset =
                                    a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));
                                constexpr index_t b_scale_offset =
                                    b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                                static_assert(0 < ScalesPerXdlopsRunPerThread,
                                              "Must have at least one scale per Xdlops "
                                              "per Thread.");

                                vector_type<AScaleDataType, a_scale_thread_vec_size>
                                    a_scale_thread_vec;
                                vector_type<BScaleDataType, b_scale_thread_vec_size>
                                    b_scale_thread_vec;

                                // Pack scale_thread_buf into scale_thread_vec
                                static_for<0, a_scale_thread_vec_size, 1>{}([&](auto s) {
                                    a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                        a_scale_thread_bufs(
                                            scale_comp_buf)[Number<a_scale_offset + s>{}];
                                });

                                static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                                    b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                        b_scale_thread_bufs(
                                            scale_comp_buf)[Number<b_scale_offset + s>{}];
                                });

                                static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                                    static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                        static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                            constexpr auto kxdl = ikxdl + k0 * KXdlPack;

                                            vector_type<ComputeTypeA, KPack> a_thread_vec;
                                            vector_type<ComputeTypeB, KPack> b_thread_vec;

                                            static_for<0, KPack, 1>{}([&](auto ik) {
                                                a_thread_vec.template AsType<ComputeTypeA>()(
                                                    ik) = a_thread_buf
                                                    [Number<a_thread_desc_.CalculateOffset(
                                                        make_tuple(m0, I0, imxdl, kxdl, ik))>{}];
                                                b_thread_vec.template AsType<ComputeTypeB>()(
                                                    ik) = b_thread_buf
                                                    [Number<b_thread_desc_.CalculateOffset(
                                                        make_tuple(n0, I0, inxdl, kxdl, ik))>{}];
                                            });

                                            using mfma_input_type_a =
                                                typename vector_type<ComputeTypeA,
                                                                     xdlops_gemm.K1PerXdlops /
                                                                         APackedSize>::type;

                                            using mfma_input_type_b =
                                                typename vector_type<ComputeTypeB,
                                                                     xdlops_gemm.K1PerXdlops /
                                                                         BPackedSize>::type;

                                            using mfma_scale_input_type_a =
                                                typename vector_type<AScaleDataType,
                                                                     a_scale_thread_vec_size>::type;
                                            using mfma_scale_input_type_b =
                                                typename vector_type<BScaleDataType,
                                                                     b_scale_thread_vec_size>::type;

                                            constexpr index_t c_offset =
                                                c_thread_desc_.CalculateOffset(
                                                    make_tuple(m0, n0, imxdl, inxdl, 0));

                                            // MFMA accumulation
                                            xdlops_gemm.template Run<ikxdl * MXdlPack + imxdl,
                                                                     ikxdl * NXdlPack + inxdl>(
                                                a_thread_vec.template AsType<mfma_input_type_a>(),
                                                a_scale_thread_vec
                                                    .template AsType<mfma_scale_input_type_a>(),
                                                b_thread_vec.template AsType<mfma_input_type_b>(),
                                                b_scale_thread_vec
                                                    .template AsType<mfma_scale_input_type_b>(),
                                                c_thread_buf.GetVectorTypeReference(
                                                    Number<c_offset>{}));
                                        });
                                    });
                                });
                            });
                        });
                    });

                    // Local prefetch A2
                    block_sync_lds();
                    static_for<0, KRepeat, 1>{}([&](auto k) {
                        constexpr auto k_step = k * xdlops_gemm.KPerXdlops / APackedSize *
                                                (APackedSize * KPack / xdlops_gemm.K1PerXdlops);
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            static_for<0,
                                       xdlops_gemm.K1PerXdlops / (APackedSize * KThreadChunk),
                                       1>{}([&](auto chunk) {
                                constexpr auto a_k_step_chunk =
                                    k_step +
                                    chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                                a_thread_copy_.Run(a_block_desc_m0_m1_m2_m3_k,
                                                   make_tuple(Number<m0 / MXdlPack>{},
                                                              I0,
                                                              Number<m0 % MXdlPack>{},
                                                              I0,
                                                              Number<a_k_step_chunk>{}),
                                                   a_block_buf.At(scale_mem_buf),
                                                   a_thread_desc_,
                                                   make_tuple(Number<m0 / MXdlPack>{},
                                                              I0,
                                                              Number<m0 % MXdlPack>{},
                                                              k,
                                                              Number<chunk * KThreadChunk>{}),
                                                   a_thread_buf);
                            });
                        });
                    });

                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);
                }; // LoopFunc

                LoopFunc(I0, I1);
                LoopFunc(I1, I0);

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
            // Prefetch a_scales
            static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                    a_scale_thread_copy.Run(a_scale_grid_desc,
                                            a_scale_grid_buf,
                                            a_scale_thread_desc,
                                            make_tuple(m0, k0, I0),
                                            a_scale_thread_bufs(I1));

                    a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                           make_multi_index(0, I1, 0));
                });
                a_scale_thread_copy.MoveSrcSliceWindow(
                    a_scale_grid_desc, make_multi_index(MWaves, -KRepeat / KXdlPack, 0));
            });

            // Prefetch b_scales
            static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                    b_scale_thread_copy.Run(b_scale_grid_desc,
                                            b_scale_grid_buf,
                                            b_scale_thread_desc,
                                            make_tuple(n0, k0, I0),
                                            b_scale_thread_bufs(I1));

                    b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                           make_multi_index(0, I1, 0));
                });
                b_scale_thread_copy.MoveSrcSliceWindow(
                    b_scale_grid_desc, make_multi_index(NWaves, -KRepeat / KXdlPack, 0));
            });

            // Local prefill A2
            block_sync_lds();
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I1));

            // Global prefetch B2
            b_blockwise_copy.Run(b_grid_desc,
                                 b_grid_buf,
                                 b_block_desc_n0_n1_n2_k0_k1,
                                 b_block_origin_idx,
                                 b_thread_bufs(I1));

            // A1 * B1
            static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                    static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                        constexpr index_t a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));
                        constexpr index_t b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                        static_assert(0 < ScalesPerXdlopsRunPerThread,
                                      "Must have at least one scale per Xdlops "
                                      "per Thread.");

                        vector_type<AScaleDataType, a_scale_thread_vec_size> a_scale_thread_vec;
                        vector_type<BScaleDataType, b_scale_thread_vec_size> b_scale_thread_vec;

                        // Pack scale_thread_buf into scale_thread_vec
                        static_for<0, a_scale_thread_vec_size, 1>{}([&](auto s) {
                            a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                a_scale_thread_bufs(I0)[Number<a_scale_offset + s>{}];
                        });

                        static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                            b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                b_scale_thread_bufs(I0)[Number<b_scale_offset + s>{}];
                        });

                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    constexpr auto kxdl = ikxdl + k0 * KXdlPack;

                                    vector_type<ComputeTypeA, KPack> a_thread_vec;
                                    vector_type<ComputeTypeB, KPack> b_thread_vec;

                                    static_for<0, KPack, 1>{}([&](auto ik) {
                                        a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                                make_tuple(m0, I0, imxdl, kxdl, ik))>{}];
                                        b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                            b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                                make_tuple(n0, I0, inxdl, kxdl, ik))>{}];
                                    });

                                    using mfma_input_type_a =
                                        typename vector_type<ComputeTypeA,
                                                             xdlops_gemm.K1PerXdlops /
                                                                 APackedSize>::type;

                                    using mfma_input_type_b =
                                        typename vector_type<ComputeTypeB,
                                                             xdlops_gemm.K1PerXdlops /
                                                                 BPackedSize>::type;

                                    using mfma_scale_input_type_a =
                                        typename vector_type<AScaleDataType,
                                                             a_scale_thread_vec_size>::type;
                                    using mfma_scale_input_type_b =
                                        typename vector_type<BScaleDataType,
                                                             b_scale_thread_vec_size>::type;

                                    constexpr index_t c_offset = c_thread_desc_.CalculateOffset(
                                        make_tuple(m0, n0, imxdl, inxdl, 0));

                                    // MFMA accumulation
                                    xdlops_gemm.template Run<ikxdl * MXdlPack + imxdl,
                                                             ikxdl * NXdlPack + inxdl>(
                                        a_thread_vec.template AsType<mfma_input_type_a>(),
                                        a_scale_thread_vec
                                            .template AsType<mfma_scale_input_type_a>(),
                                        b_thread_vec.template AsType<mfma_input_type_b>(),
                                        b_scale_thread_vec
                                            .template AsType<mfma_scale_input_type_b>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                                });
                            });
                        });
                    });
                });
            });

            // Local prefetch A2
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                constexpr auto k_step = k * xdlops_gemm.KPerXdlops / APackedSize *
                                        (APackedSize * KPack / xdlops_gemm.K1PerXdlops);
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, xdlops_gemm.K1PerXdlops / (APackedSize * KThreadChunk), 1>{}(
                        [&](auto chunk) {
                            constexpr auto a_k_step_chunk =
                                k_step +
                                chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                            a_thread_copy_.Run(a_block_desc_m0_m1_m2_m3_k,
                                               make_tuple(Number<m0 / MXdlPack>{},
                                                          I0,
                                                          Number<m0 % MXdlPack>{},
                                                          I0,
                                                          Number<a_k_step_chunk>{}),
                                               a_block_buf.At(I0),
                                               a_thread_desc_,
                                               make_tuple(Number<m0 / MXdlPack>{},
                                                          I0,
                                                          Number<m0 % MXdlPack>{},
                                                          k,
                                                          Number<chunk * KThreadChunk>{}),
                                               a_thread_buf);
                        });
                });
            });

            // A2 * B2
            static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                    static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                        constexpr index_t a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));
                        constexpr index_t b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                        static_assert(0 < ScalesPerXdlopsRunPerThread,
                                      "Must have at least one scale per Xdlops "
                                      "per Thread.");

                        vector_type<AScaleDataType, a_scale_thread_vec_size> a_scale_thread_vec;
                        vector_type<BScaleDataType, b_scale_thread_vec_size> b_scale_thread_vec;

                        // Pack scale_thread_buf into scale_thread_vec
                        static_for<0, a_scale_thread_vec_size, 1>{}([&](auto s) {
                            a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                a_scale_thread_bufs(I1)[Number<a_scale_offset + s>{}];
                        });

                        static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                            b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                b_scale_thread_bufs(I1)[Number<b_scale_offset + s>{}];
                        });

                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    constexpr auto kxdl = ikxdl + k0 * KXdlPack;

                                    vector_type<ComputeTypeA, KPack> a_thread_vec;
                                    vector_type<ComputeTypeB, KPack> b_thread_vec;

                                    static_for<0, KPack, 1>{}([&](auto ik) {
                                        a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                                make_tuple(m0, I0, imxdl, kxdl, ik))>{}];
                                        b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                            b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                                make_tuple(n0, I0, inxdl, kxdl, ik))>{}];
                                    });

                                    using mfma_input_type_a =
                                        typename vector_type<ComputeTypeA,
                                                             xdlops_gemm.K1PerXdlops /
                                                                 APackedSize>::type;

                                    using mfma_input_type_b =
                                        typename vector_type<ComputeTypeB,
                                                             xdlops_gemm.K1PerXdlops /
                                                                 BPackedSize>::type;

                                    using mfma_scale_input_type_a =
                                        typename vector_type<AScaleDataType,
                                                             a_scale_thread_vec_size>::type;
                                    using mfma_scale_input_type_b =
                                        typename vector_type<BScaleDataType,
                                                             b_scale_thread_vec_size>::type;

                                    constexpr index_t c_offset = c_thread_desc_.CalculateOffset(
                                        make_tuple(m0, n0, imxdl, inxdl, 0));

                                    // MFMA accumulation
                                    xdlops_gemm.template Run<ikxdl * MXdlPack + imxdl,
                                                             ikxdl * NXdlPack + inxdl>(
                                        a_thread_vec.template AsType<mfma_input_type_a>(),
                                        a_scale_thread_vec
                                            .template AsType<mfma_scale_input_type_a>(),
                                        b_thread_vec.template AsType<mfma_input_type_b>(),
                                        b_scale_thread_vec
                                            .template AsType<mfma_scale_input_type_b>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                                });
                            });
                        });
                    });
                });
            });
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                    static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                        constexpr index_t a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));
                        constexpr index_t b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                        static_assert(0 < ScalesPerXdlopsRunPerThread,
                                      "Must have at least one scale per Xdlops "
                                      "per Thread.");

                        vector_type<AScaleDataType, a_scale_thread_vec_size> a_scale_thread_vec;
                        vector_type<BScaleDataType, b_scale_thread_vec_size> b_scale_thread_vec;

                        // Pack scale_thread_buf into scale_thread_vec
                        static_for<0, a_scale_thread_vec_size, 1>{}([&](auto s) {
                            a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                a_scale_thread_bufs(I0)[Number<a_scale_offset + s>{}];
                        });

                        static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                            b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                b_scale_thread_bufs(I0)[Number<b_scale_offset + s>{}];
                        });

                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    constexpr auto kxdl = ikxdl + k0 * KXdlPack;

                                    vector_type<ComputeTypeA, KPack> a_thread_vec;
                                    vector_type<ComputeTypeB, KPack> b_thread_vec;

                                    static_for<0, KPack, 1>{}([&](auto ik) {
                                        a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                                make_tuple(m0, I0, imxdl, kxdl, ik))>{}];
                                        // b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                        //     b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        //         make_tuple(n0, I0, inxdl, kxdl, ik))>{}];
                                        b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                            type_convert<ComputeTypeB>(ck::float2_t(1.0));
                                    });

                                    using mfma_input_type_a =
                                        typename vector_type<ComputeTypeA,
                                                             xdlops_gemm.K1PerXdlops /
                                                                 APackedSize>::type;

                                    using mfma_input_type_b =
                                        typename vector_type<ComputeTypeB,
                                                             xdlops_gemm.K1PerXdlops /
                                                                 BPackedSize>::type;

                                    using mfma_scale_input_type_a =
                                        typename vector_type<AScaleDataType,
                                                             a_scale_thread_vec_size>::type;
                                    using mfma_scale_input_type_b =
                                        typename vector_type<BScaleDataType,
                                                             b_scale_thread_vec_size>::type;

                                    constexpr index_t c_offset = c_thread_desc_.CalculateOffset(
                                        make_tuple(m0, n0, imxdl, inxdl, 0));

                                    // MFMA accumulation
                                    xdlops_gemm.template Run<ikxdl * MXdlPack + imxdl,
                                                             ikxdl * NXdlPack + inxdl>(
                                        a_thread_vec.template AsType<mfma_input_type_a>(),
                                        a_scale_thread_vec
                                            .template AsType<mfma_scale_input_type_a>(),
                                        b_thread_vec.template AsType<mfma_input_type_b>(),
                                        b_scale_thread_vec
                                            .template AsType<mfma_scale_input_type_b>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                                });
                            });
                        });
                    });
                });
            });
        }
    }

    // TODO: make this field protected when a_scale_thread_copy_ is moved
    // here
    static constexpr auto a_scale_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat / MXdlPack>{},
                   Number<KRepeat / KXdlPack>{},
                   Number<ScalesPerXdlopsRunPerThread * a_scale_thread_vec_size>{}));

    // TODO: make this field protected when b_scale_thread_copy_ is moved
    // here
    static constexpr auto b_scale_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<NRepeat / NXdlPack>{},
                   Number<KRepeat / KXdlPack>{},
                   Number<ScalesPerXdlopsRunPerThread * b_scale_thread_vec_size>{}));

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;

    static constexpr BTileDesc b_block_desc_n0_n1_n2_k0_k1;
};

} // namespace ck
