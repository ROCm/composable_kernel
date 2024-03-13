// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp"

namespace ck {

// Compute optimized pipeline
// GlobalPrefetchStages: 3
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 2

template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
          index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
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
          index_t MRepeat,
          index_t NRepeat,
          index_t KPacks>
struct BlockwiseGemmXdlops_pipeline_v5
{
};

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
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
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack
          // ,bool TransposeC //disable transposec right now...
          >
struct BlockwiseGemmXdlops_pipeline_v5<BlockGemmPipelineScheduler::Intrawave,
                                       BlockSize,
                                       FloatAB,
                                       FloatAcc,
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
    : BlockwiseGemmXdlops_pipeline_base<BlockSize,
                                        FloatAB,
                                        FloatAcc,
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
    using Base = BlockwiseGemmXdlops_pipeline_base<BlockSize,
                                                   FloatAB,
                                                   FloatAcc,
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
    using Base::A_K1;
    using Base::B_K1;
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

    static constexpr index_t PrefetchStages  = 3;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 2;
    static constexpr index_t HotloopUnroll   = 2;

    __host__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % HotloopUnroll == 1)
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

        constexpr auto num_dsread_stage1  = num_ds_read_inst / KRepeat * (KRepeat - 1);
        constexpr auto num_dsread_stage3  = num_ds_read_inst / KRepeat;
        constexpr auto num_mfma_stage2    = num_mfma_inst - num_ds_read_inst;
        constexpr auto num_mfma_per_issue = num_mfma_stage2 / num_buffer_load_inst;
        constexpr auto num_dswrite_per_issue =
            (num_ds_write_inst + num_buffer_load_inst - 1) / num_buffer_load_inst;
        constexpr auto num_issue_more = num_mfma_stage2 - num_mfma_per_issue * num_buffer_load_inst;
        constexpr auto num_issue_less = num_buffer_load_inst - num_issue_more;

        // stage 1
        static_for<0, num_dsread_stage1, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });

        // stage 2
        static_for<0, num_issue_more, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x200, num_dswrite_per_issue, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                     // MFMA
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);                     // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008, num_mfma_per_issue, 0);    // MFMA
        });
        static_for<0, num_issue_less, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x200, num_dswrite_per_issue, 0);  // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                      // MFMA
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);                      // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008, num_mfma_per_issue - 1, 0); // MFMA
        });

        // stage 3
        static_for<0, num_dsread_stage3, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });

        // IGLP COMPILER BUG:
        // If comment out following scheduler barrier would cause sanity fail.
        __builtin_amdgcn_sched_barrier(0);
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
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

        // Global prefetch 2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Global prefetch 3
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // Local prefetch 1
        block_sync_lds();
        static_for<0, MRepeat, 1>{}([&](auto m0) {
            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                               make_tuple(m0, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(m0, I0, I0, I0),
                               a_thread_buf);
        });
        static_for<0, NRepeat, 1>{}([&](auto n0) {
            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                               make_tuple(n0, I0, I0, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(n0, I0, I0, I0),
                               b_thread_buf);
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                auto LoopFunc = [&](auto vmem_buf) {
                    vector_type<FloatAB, KPack> a_thread_vec;
                    vector_type<FloatAB, KPack> b_thread_vec;

                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        if constexpr(k0 == (KRepeat - 1))
                        {
                            block_sync_lds();

                            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, vmem_buf);
                            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, vmem_buf);

                            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, vmem_buf);
                            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, vmem_buf);

                            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                            block_sync_lds();
                        }
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<FloatAB>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(m0, I0, I0, ik))>{}];
                                });
                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    b_thread_vec.template AsType<FloatAB>()(ik) =
                                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                            make_tuple(n0, I0, I0, ik))>{}];
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

                            a_thread_copy_.Run(
                                a_block_desc_m0_m1_m2_k,
                                make_tuple(m0, I0, I0, Number<(k0 + 1) % KRepeat * AMmaKStride>{}),
                                a_block_buf,
                                a_thread_desc_,
                                make_tuple(m0, I0, I0, I0),
                                a_thread_buf);
                        });

                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(
                                b_block_desc_n0_n1_n2_k,
                                make_tuple(n0, I0, I0, Number<(k0 + 1) % KRepeat * BMmaKStride>{}),
                                b_block_buf,
                                b_thread_desc_,
                                make_tuple(n0, I0, I0, I0),
                                b_thread_buf);
                        });
                    });

                    HotLoopScheduler();
                };

                LoopFunc(I0);
                LoopFunc(I1);

                i += HotloopUnroll;
            } while(i < (num_loop - PrefetchStages));
        }
        // tail
        auto ReadWriteCompFunc = [&](auto vmem_buf) {
            vector_type<FloatAB, KPack> a_thread_vec;
            vector_type<FloatAB, KPack> b_thread_vec;

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                if constexpr(k0 == (KRepeat - 1))
                {
                    block_sync_lds();

                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, vmem_buf);
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, vmem_buf);

                    block_sync_lds();
                }
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, I0, ik))>{}];
                        });
                        static_for<0, KPack, 1>{}([&](auto ik) {
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, I0, ik))>{}];
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
                    a_thread_copy_.Run(
                        a_block_desc_m0_m1_m2_k,
                        make_tuple(m0, I0, I0, Number<(k0 + 1) % KRepeat * AMmaKStride>{}),
                        a_block_buf,
                        a_thread_desc_,
                        make_tuple(m0, I0, I0, I0),
                        a_thread_buf);
                });

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(
                        b_block_desc_n0_n1_n2_k,
                        make_tuple(n0, I0, I0, Number<(k0 + 1) % KRepeat * BMmaKStride>{}),
                        b_block_buf,
                        b_thread_desc_,
                        make_tuple(n0, I0, I0, I0),
                        b_thread_buf);
                });
            });

            HotLoopScheduler();
        };
        auto ReadCompFunc = [&]() {
            vector_type<FloatAB, KPack> a_thread_vec;
            vector_type<FloatAB, KPack> b_thread_vec;

            static_for<0, KRepeat - 1, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, I0, ik))>{}];
                        });
                        static_for<0, KPack, 1>{}([&](auto ik) {
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, I0, ik))>{}];
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

                    a_thread_copy_.Run(
                        a_block_desc_m0_m1_m2_k,
                        make_tuple(m0, I0, I0, Number<(k0 + 1) % KRepeat * AMmaKStride>{}),
                        a_block_buf,
                        a_thread_desc_,
                        make_tuple(m0, I0, I0, I0),
                        a_thread_buf);
                });

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(
                        b_block_desc_n0_n1_n2_k,
                        make_tuple(n0, I0, I0, Number<(k0 + 1) % KRepeat * BMmaKStride>{}),
                        b_block_buf,
                        b_thread_desc_,
                        make_tuple(n0, I0, I0, I0),
                        b_thread_buf);
                });
            });

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, KPack, 1>{}([&](auto ik) {
                        a_thread_vec.template AsType<FloatAB>()(ik) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(m0, I0, I0, ik))>{}];
                    });
                    static_for<0, KPack, 1>{}([&](auto ik) {
                        b_thread_vec.template AsType<FloatAB>()(ik) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(n0, I0, I0, ik))>{}];
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

            HotLoopScheduler();
        };

        if constexpr(TailNum == TailNumber::Odd)
        {
            ReadWriteCompFunc(I0);
            ReadWriteCompFunc(I1);
            ReadCompFunc();
        }
        else if constexpr(TailNum == TailNumber::Even)
        {
            ReadWriteCompFunc(I0);
            ReadCompFunc();
        }
    }

    protected:
    // A[MRepeat, I1, I1, KPack]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{}, I1, I1, Number<KPack>{}));

    // B[NRepeat, N1, N2, KPack]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<NRepeat>{}, I1, I1, Number<KPack>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_{Base::CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{Base::CalculateBThreadOriginDataIndex()};
    using Base::c_thread_desc_;
};

} // namespace ck
