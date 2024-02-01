// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp"

namespace ck {

// Double LDS buffer
// Global Prefetech 3
// Local  prefill   2
// Local  prefetch  1

template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
          index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPacks>
struct BlockwiseGemmXdlops_pipeline_v4
{
};

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack
          // ,bool TransposeC //disable transposec right now...
          >
struct BlockwiseGemmXdlops_pipeline_v4<BlockGemmPipelineScheduler::Intrawave,
                                       BlockSize,
                                       FloatAB,
                                       FloatAcc,
                                       ATileDesc,
                                       BTileDesc,
                                       AMmaTileDesc,
                                       BMmaTileDesc,
                                       MPerBlock,
                                       NPerBlock,
                                       KPerBlock,
                                       MPerXDL,
                                       NPerXDL,
                                       MRepeat,
                                       NRepeat,
                                       KPack> : BlockwiseGemmXdlops_pipeline_base<BlockSize,
                                                                                  FloatAB,
                                                                                  FloatAcc,
                                                                                  ATileDesc,
                                                                                  BTileDesc,
                                                                                  AMmaTileDesc,
                                                                                  BMmaTileDesc,
                                                                                  MPerBlock,
                                                                                  NPerBlock,
                                                                                  KPerBlock,
                                                                                  MPerXDL,
                                                                                  NPerXDL,
                                                                                  MRepeat,
                                                                                  NRepeat,
                                                                                  KPack>

{
    using Base = BlockwiseGemmXdlops_pipeline_base<BlockSize,
                                                   FloatAB,
                                                   FloatAcc,
                                                   ATileDesc,
                                                   BTileDesc,
                                                   AMmaTileDesc,
                                                   BMmaTileDesc,
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
    using Base::xdlops_gemm;
    using typename Base::HotLoopInstList;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::CalculateCThreadOriginDataIndex8D;
    using Base::GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetCThreadBuffer;
    using Base::GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;

    using Base::a_block_desc_m0_m1_m2_k;
    using Base::b_block_desc_n0_n1_n2_k;

    using Base::AMmaKStride;
    using Base::BMmaKStride;

    static constexpr index_t MinimumLoop = 5;

    __host__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > MinimumLoop;
    }

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        if((num_loop - MinimumLoop) % 2 == 1)
        {
            return TailNumber::Odd;
        }
        else
        {
            return TailNumber::Even;
        }
    }

    __device__ static constexpr auto HotLoopScheduler()
    {
        // schedule
        constexpr auto num_ds_read_inst =
            HotLoopInstList::A_LDS_Read_Inst_Num + HotLoopInstList::B_LDS_Read_Inst_Num;
        constexpr auto num_ds_write_inst =
            HotLoopInstList::A_LDS_Write_Inst_Num + HotLoopInstList::B_LDS_Write_Inst_Num;

        constexpr auto num_buffer_load_inst =
            HotLoopInstList::A_Buffer_Load_Inst_Num + HotLoopInstList::B_Buffer_Load_Inst_Num;

        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto num_issue = num_buffer_load_inst;

        static_for<0, num_issue, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(
                0x100, num_ds_read_inst / num_buffer_load_inst, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);      // MFMA
            __builtin_amdgcn_sched_group_barrier(
                0x200, num_ds_write_inst / num_buffer_load_inst, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);       // MFMA
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);       // VMEM read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_mfma_inst / num_buffer_load_inst - 3, 0); // MFMA
        });
    }

    template <index_t stage>
    __device__ static constexpr auto TailScheduler()
    {
    }

    template <>
    __device__ static constexpr auto TailScheduler<1>()
    {
        // schedule
        constexpr auto num_ds_read_inst =
            HotLoopInstList::A_LDS_Read_Inst_Num + HotLoopInstList::B_LDS_Read_Inst_Num;
        constexpr auto num_ds_write_inst =
            HotLoopInstList::A_LDS_Write_Inst_Num + HotLoopInstList::B_LDS_Write_Inst_Num;
        ;
        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto num_issue = num_ds_write_inst;

        static_for<0, num_issue, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(
                0x100, num_ds_read_inst / num_ds_write_inst - 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_mfma_inst / num_ds_write_inst - 3, 0); // MFMA
        });
    }

    template <>
    __device__ static constexpr auto TailScheduler<2>()
    {
        // schedule
        constexpr auto num_ds_read_inst =
            HotLoopInstList::A_LDS_Read_Inst_Num + HotLoopInstList::B_LDS_Read_Inst_Num;
        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto num_issue = num_ds_read_inst;

        static_for<0, num_issue, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_mfma_inst / num_ds_read_inst, 0); // MFMA
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
              typename CThreadBuffer>
    __device__ void Run(const AGridDesc& a_grid_desc,
                        const ABlockDesc& a_block_desc,
                        ABlockTransfer& a_blockwise_copy,
                        const AGridBuffer& a_grid_buf,
                        ABlockBuffer& a_block_buf,
                        const ABlockTransferStep& a_block_copy_step,
                        const BGridDesc& b_grid_desc,
                        const BBlockDesc& b_block_desc,
                        BBlockTransfer& b_blockwise_copy,
                        const BGridBuffer& b_grid_buf,
                        BBlockBuffer& b_block_buf,
                        const BBlockTransferStep& b_block_copy_step,
                        CThreadBuffer& c_thread_buf,
                        index_t num_loop) const
    {
        __builtin_amdgcn_sched_barrier(0);
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(a_thread_buf), Number<2>{}> a_thread_bufs;
        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I0));
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(I0));

        // Local prefetch 1
        block_sync_lds();
        static_for<0, KRepeat, 1>{}([&](auto k) {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                   make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                   a_block_buf.At(I0),
                                   a_thread_desc_,
                                   make_tuple(m0, I0, k, I0),
                                   a_thread_bufs(I0));
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf.At(I0),
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, I0),
                                       b_thread_bufs(I0));
                });
            });
        });

        // Global prefetch 2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Local prefill 2
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I1));
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(I1));

        // Global prefetch 3
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            // This hot loop has two legacy loopover, to implement the double local buffer strategy
            do
            {
                // -------------------------------------------------------------------------------------------
                using PingP1 = Number<0>;
                using PongP1 = Number<1>;
                // MFMA: Ping Reg
                // DS_WRITE: To Ping LDS
                // DS_READ: Pong LDS to Pong Reg
                block_sync_lds();

                static_for<0, KRepeat, 1>{}([&](auto k) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                           a_block_buf.At(PongP1{}),
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k, I0),
                                           a_thread_bufs(PongP1{}));
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                               b_block_buf.At(PongP1{}),
                                               b_thread_desc_,
                                               make_tuple(n0, I0, k, I0),
                                               b_thread_bufs(PongP1{}));
                        });
                    });
                });

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(PingP1{}));
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(PingP1{}));

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<FloatAB, KPack> a_thread_vec;
                            vector_type<FloatAB, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<FloatAB>()(ik) =
                                    a_thread_bufs[PingP1{}][Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, I0, k0, ik))>{}];
                                b_thread_vec.template AsType<FloatAB>()(ik) =
                                    b_thread_bufs[PingP1{}][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0, I0, k0, ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                            xdlops_gemm.template Run(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        });
                    });
                });

                HotLoopScheduler();
                __builtin_amdgcn_sched_barrier(0);

                // -------------------------------------------------------------------------------------------
                using PingP2 = Number<1>;
                using PongP2 = Number<0>;
                // MFMA: Pong Reg
                // DS_WRITE: To Pong LDS
                // DS_READ: Ping LDS to Ping Reg
                block_sync_lds();

                static_for<0, KRepeat, 1>{}([&](auto k) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                           a_block_buf.At(PongP2{}),
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k, I0),
                                           a_thread_bufs(PongP2{}));
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                               b_block_buf.At(PongP2{}),
                                               b_thread_desc_,
                                               make_tuple(n0, I0, k, I0),
                                               b_thread_bufs(PongP2{}));
                        });
                    });
                });

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(PingP2{}));
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(PingP2{}));

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<FloatAB, KPack> a_thread_vec;
                            vector_type<FloatAB, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<FloatAB>()(ik) =
                                    a_thread_bufs[PingP2{}][Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, I0, k0, ik))>{}];
                                b_thread_vec.template AsType<FloatAB>()(ik) =
                                    b_thread_bufs[PingP2{}][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0, I0, k0, ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                            xdlops_gemm.template Run(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        });
                    });
                });

                HotLoopScheduler();
                __builtin_amdgcn_sched_barrier(0);

                i += 2;
            } while(i < (num_loop - 3));
        }

        // tail
        if constexpr(TailNum == TailNumber::Odd)
        {
            using PingP1 = Number<0>;
            using PongP1 = Number<1>;
            // MFMA: Ping Reg
            // DS_WRITE: To Ping LDS
            // DS_READ: Pong LDS to Pong Reg
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(PongP1{}),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(PongP1{}));
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf.At(PongP1{}),
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_bufs(PongP1{}));
                    });
                });
            });

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(PingP1{}));
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(PingP1{}));

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PingP1{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PingP1{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            TailScheduler<1>();
            __builtin_amdgcn_sched_barrier(0);

            // -------------------------------------------------------------------------------------------
            using PingP2 = Number<1>;
            using PongP2 = Number<0>;
            // MFMA: Pong Reg
            // DS_WRITE: To Pong LDS
            // DS_READ: Ping LDS to Ping Reg
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(PongP2{}),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(PongP2{}));
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf.At(PongP2{}),
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_bufs(PongP2{}));
                    });
                });
            });

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PingP2{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PingP2{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            TailScheduler<2>();
            __builtin_amdgcn_sched_barrier(0);

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PongP2{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PongP2{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            // 64 v_mfma
            __builtin_amdgcn_sched_group_barrier(0x008, 64, 0); // MFMA
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(TailNum == TailNumber::Even)
        {
            using PingP1 = Number<0>;
            using PongP1 = Number<1>;
            // MFMA: Ping Reg
            // DS_WRITE: To Ping LDS
            // DS_READ: Pong LDS to Pong Reg
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(PongP1{}),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(PongP1{}));
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf.At(PongP1{}),
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_bufs(PongP1{}));
                    });
                });
            });

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PingP1{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PingP1{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            TailScheduler<2>();
            __builtin_amdgcn_sched_barrier(0);

            // -------------------------------------------------------------------------------------------
            using PingP2 = Number<1>;
            // MFMA: Pong Reg
            // DS_WRITE: To Pong LDS
            // DS_READ: Ping LDS to Ping Reg

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PingP2{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PingP2{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.template Run(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            // 64 v_mfma
            __builtin_amdgcn_sched_group_barrier(0x008, 64, 0); // MFMA
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
};

} // namespace ck
