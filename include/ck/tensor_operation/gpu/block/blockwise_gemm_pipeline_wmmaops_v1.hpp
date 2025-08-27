// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp"

namespace ck {

// Naive pipeline with lowest resource request per WGP
// GlobalPrefetchStages: 1
// LocalPreFillStages: 1
// LocalPreFetchStages: 0
// LocalSharedMemoryBuffer: 1

template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
          index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename AccDataType,
          typename AWmmaTileDesc,
          typename BWmmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
struct BlockwiseGemmWmmaops_pipeline_v1
{
};

template <index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename AccDataType,
          typename AWmmaTileDesc,
          typename BWmmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
struct BlockwiseGemmWmmaops_pipeline_v1<BlockGemmPipelineScheduler::Intrawave,
                                        BlockSize,
                                        ADataType,
                                        BDataType,
                                        ComputeTypeA,
                                        ComputeTypeB,
                                        AccDataType,
                                        AWmmaTileDesc,
                                        BWmmaTileDesc,
                                        ABlockTransferSrcScalarPerVector,
                                        BBlockTransferSrcScalarPerVector,
                                        MPerBlock,
                                        NPerBlock,
                                        KPerBlock,
                                        MPerWmma,
                                        NPerWmma,
                                        MRepeat,
                                        NRepeat,
                                        KPack>
    : BlockwiseGemmWmmaops_pipeline_base<BlockSize,
                                         ADataType,
                                         BDataType,
                                         ComputeTypeA,
                                         ComputeTypeB,
                                         AccDataType,
                                         AWmmaTileDesc,
                                         BWmmaTileDesc,
                                         ABlockTransferSrcScalarPerVector,
                                         BBlockTransferSrcScalarPerVector,
                                         MPerBlock,
                                         NPerBlock,
                                         KPerBlock,
                                         MPerWmma,
                                         NPerWmma,
                                         MRepeat,
                                         NRepeat,
                                         KPack>

{
    using Base = BlockwiseGemmWmmaops_pipeline_base<BlockSize,
                                                    ADataType,
                                                    BDataType,
                                                    ComputeTypeA,
                                                    ComputeTypeB,
                                                    AccDataType,
                                                    AWmmaTileDesc,
                                                    BWmmaTileDesc,
                                                    ABlockTransferSrcScalarPerVector,
                                                    BBlockTransferSrcScalarPerVector,
                                                    MPerBlock,
                                                    NPerBlock,
                                                    KPerBlock,
                                                    MPerWmma,
                                                    NPerWmma,
                                                    MRepeat,
                                                    NRepeat,
                                                    KPack>;
    using Base::I0;

    using Base::A_K1;
    using Base::A_KRow;
    using Base::B_K1;
    using Base::B_KRow;
    using Base::KRepeat;
    using Base::WmmaK;

    using Base::wmma_gemm;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::
        GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;
    using Base::GetCThreadBuffer;
    using Base::
        GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;

    using Base::a_block_desc_k0_m0_m1_m2_k1;
    using Base::b_block_desc_k0_n0_n1_n2_k1;

    using typename Base::Empty;

    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    static bool BlockHasHotloop(index_t num_loop) { return num_loop > PrefetchStages; }

    static TailNumber BlockLoopTailNum(index_t num_loop)
    {
        ignore = num_loop;
        return TailNumber::Full;
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
              typename BScaleStruct>
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
                        // BScaleThreadCopy
                        BScaleStruct& b_scale_struct,
                        index_t num_loop,
                        index_t num_loop_per_scale) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_thread_desc_.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        b_scale_struct.template GlobalLoad<0>(num_loop_per_scale == 1);

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // Initialize C
        c_thread_buf.Clear();

        auto blockwise_gemm_func = [&]() {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(
                        a_block_desc_k0_m0_m1_m2_k1,
                        make_tuple(Number<k0 * KPack / A_K1 / A_KRow>{}, m0, I0, I0, I0, I0),
                        a_block_buf,
                        a_thread_desc_,
                        make_tuple(I0, m0, k0, I0, I0, I0),
                        a_thread_buf);
                });
                if constexpr(ck::is_same<BScaleStruct, Empty>::value == true)
                {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(
                            b_block_desc_k0_n0_n1_n2_k1,
                            make_tuple(Number<k0 * KPack / B_K1 / B_KRow>{}, n0, I0, I0, I0, I0),
                            b_block_buf,
                            b_thread_desc_,
                            make_tuple(I0, n0, k0, I0, I0, I0),
                            b_thread_buf);
                    });
                }
                else
                {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(
                            b_block_desc_k0_n0_n1_n2_k1,
                            make_tuple(Number<k0 * KPack / B_K1 / B_KRow>{}, n0, I0, I0, I0, I0),
                            b_block_buf,
                            b_scale_struct.b_scale_thread_bufs(
                                I0)[Number<n0 * BScaleStruct::num_scale_k_block +
                                           k0 / BScaleStruct::num_scale_krepeat>{}],
                            b_thread_desc_,
                            make_tuple(I0, n0, k0, I0, I0, I0),
                            b_thread_buf);
                    });
                }

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeTypeA, KPack / A_KRow> a_thread_vec;
                        vector_type<ComputeTypeB, KPack / B_KRow> b_thread_vec;

                        static_for<0, KPack / A_KRow, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(make_tuple(
                                    Number<ik / A_K1>{}, m0, k0, I0, I0, Number<ik % A_K1>{}))>{}];
                        });
                        static_for<0, KPack / B_KRow, 1>{}([&](auto ik) {
                            b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(make_tuple(
                                    Number<ik / B_K1>{}, n0, k0, I0, I0, Number<ik % B_K1>{}))>{}];
                        });

                        using wmma_input_type_a =
                            typename vector_type<ComputeTypeA, WmmaK / A_KRow>::type;
                        using wmma_input_type_b =
                            typename vector_type<ComputeTypeB, WmmaK / B_KRow>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, I0));

                        wmma_gemm.Run(a_thread_vec.template AsType<wmma_input_type_a>(),
                                      b_thread_vec.template AsType<wmma_input_type_b>(),
                                      c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });
        };

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                block_sync_lds();
                blockwise_gemm_func();

                block_sync_lds();
                b_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                i += 1;
            } while(i < (num_loop - 1));
        }

        // tail
        if constexpr(TailNum == TailNumber::Full)
        {
            block_sync_lds();
            blockwise_gemm_func();
        }
    }

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
};

template <index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename AccDataType,
          typename AWmmaTileDesc,
          typename BWmmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
struct BlockwiseGemmWmmaops_pipeline_v1<BlockGemmPipelineScheduler::Interwave,
                                        BlockSize,
                                        ADataType,
                                        BDataType,
                                        ComputeTypeA,
                                        ComputeTypeB,
                                        AccDataType,
                                        AWmmaTileDesc,
                                        BWmmaTileDesc,
                                        ABlockTransferSrcScalarPerVector,
                                        BBlockTransferSrcScalarPerVector,
                                        MPerBlock,
                                        NPerBlock,
                                        KPerBlock,
                                        MPerWmma,
                                        NPerWmma,
                                        MRepeat,
                                        NRepeat,
                                        KPack>
    : BlockwiseGemmWmmaops_pipeline_base<BlockSize,
                                         ADataType,
                                         BDataType,
                                         ComputeTypeA,
                                         ComputeTypeB,
                                         AccDataType,
                                         AWmmaTileDesc,
                                         BWmmaTileDesc,
                                         ABlockTransferSrcScalarPerVector,
                                         BBlockTransferSrcScalarPerVector,
                                         MPerBlock,
                                         NPerBlock,
                                         KPerBlock,
                                         MPerWmma,
                                         NPerWmma,
                                         MRepeat,
                                         NRepeat,
                                         KPack>

{
    using Base = BlockwiseGemmWmmaops_pipeline_base<BlockSize,
                                                    ADataType,
                                                    BDataType,
                                                    ComputeTypeA,
                                                    ComputeTypeB,
                                                    AccDataType,
                                                    AWmmaTileDesc,
                                                    BWmmaTileDesc,
                                                    ABlockTransferSrcScalarPerVector,
                                                    BBlockTransferSrcScalarPerVector,
                                                    MPerBlock,
                                                    NPerBlock,
                                                    KPerBlock,
                                                    MPerWmma,
                                                    NPerWmma,
                                                    MRepeat,
                                                    NRepeat,
                                                    KPack>;
    using Base::I0;
    using Base::I1;

    using Base::A_K1;
    using Base::A_KRow;
    using Base::B_K1;
    using Base::B_KRow;
    using Base::KRepeat;
    using Base::WmmaK;

    using Base::wmma_gemm;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::
        GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;
    using Base::GetCThreadBuffer;
    using Base::
        GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;

    using Base::a_block_desc_k0_m0_m1_m2_k1;
    using Base::b_block_desc_k0_n0_n1_n2_k1;

    using typename Base::Empty;

    static constexpr index_t NumKClusters      = CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS;
    static constexpr index_t KRepeatPerCluster = math::max(KRepeat / NumKClusters, 1);

    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    static bool BlockHasHotloop(index_t num_loop) { return num_loop > PrefetchStages; }

    static TailNumber BlockLoopTailNum(index_t num_loop)
    {
        ignore = num_loop;
        return TailNumber::Full;
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
              typename BScaleStruct>
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
                        // BScaleThreadCopy
                        BScaleStruct& b_scale_struct,
                        index_t num_loop,
                        index_t num_loop_per_scale) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_thread_desc_.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        b_scale_struct.template GlobalLoad<0>(num_loop_per_scale == 1);

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // Initialize C
        c_thread_buf.Clear();

        auto blockwise_gemm_func = [&]() {
            static_for<0, KRepeat, KRepeatPerCluster>{}([&](auto k0_offset) {
                static_for<0, KRepeatPerCluster, 1>{}([&](auto k0_inner) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(
                            a_block_desc_k0_m0_m1_m2_k1,
                            make_tuple(Number<(k0_offset + k0_inner) * KPack / A_K1 / A_KRow>{},
                                       m0,
                                       I0,
                                       I0,
                                       I0,
                                       I0),
                            a_block_buf,
                            a_thread_desc_,
                            make_tuple(I0, m0, k0_inner, I0, I0, I0),
                            a_thread_buf);
                    });
                    if constexpr(ck::is_same<BScaleStruct, Empty>::value == true)
                    {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(
                                b_block_desc_k0_n0_n1_n2_k1,
                                make_tuple(Number<(k0_offset + k0_inner) * KPack / B_K1 / B_KRow>{},
                                           n0,
                                           I0,
                                           I0,
                                           I0,
                                           I0),
                                b_block_buf,
                                b_thread_desc_,
                                make_tuple(I0, n0, k0_inner, I0, I0, I0),
                                b_thread_buf);
                        });
                    }
                    else
                    {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(
                                b_block_desc_k0_n0_n1_n2_k1,
                                make_tuple(Number<(k0_offset + k0_inner) * KPack / B_K1 / B_KRow>{},
                                           n0,
                                           I0,
                                           I0,
                                           I0,
                                           I0),
                                b_block_buf,
                                b_scale_struct.b_scale_thread_bufs(I0)[Number<
                                    n0 * BScaleStruct::num_scale_k_block +
                                    (k0_offset + k0_inner) / BScaleStruct::num_scale_krepeat>{}],
                                b_thread_desc_,
                                make_tuple(I0, n0, k0_inner, I0, I0, I0),
                                b_thread_buf);
                        });
                    }
                });

                __builtin_amdgcn_sched_barrier(0);
                // NOTE: Synchronize threads in a workgroup at the start of each MAC cluster,
                // but except the first, as we can shorten non-MAC cluster a bit and there's no
                // observable negative impact. The desired effect is waves in a workgroup
                // executing MAC in sync. This avoids some out-of-sync waves hijacking MAC
                // resource from other workgroups and reducing the chance of latency hiding by
                // waiting for the rest of the workgroup at the eventual sync point.
                if constexpr(k0_offset != 0 || KRepeat == 1)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                }
                static_for<0, KRepeatPerCluster, 1>{}([&](auto k0_inner) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<ComputeTypeA, KPack / A_KRow> a_thread_vec;
                            vector_type<ComputeTypeB, KPack / B_KRow> b_thread_vec;

                            static_for<0, KPack / A_KRow, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(Number<ik / A_K1>{},
                                                   m0,
                                                   k0_inner,
                                                   I0,
                                                   I0,
                                                   Number<ik % A_K1>{}))>{}];
                            });
                            static_for<0, KPack / B_KRow, 1>{}([&](auto ik) {
                                b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(Number<ik / B_K1>{},
                                                   n0,
                                                   k0_inner,
                                                   I0,
                                                   I0,
                                                   Number<ik % B_K1>{}))>{}];
                            });

                            using wmma_input_type_a =
                                typename vector_type<ComputeTypeA, WmmaK / A_KRow>::type;
                            using wmma_input_type_b =
                                typename vector_type<ComputeTypeB, WmmaK / B_KRow>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, I0));

                            // The block_sync_lds() here performs double duty:
                            // A) safeguard against data hazard.
                            // B) reduce VMEM FIFO congestion by applying small delays to
                            // different wavefronts.
                            // It is performed near the end of MAC cluster to minimize lgkmcnt
                            // penalty
                            if constexpr(k0_offset + k0_inner == KRepeat - 1 && m0 == MRepeat - 1 &&
                                         n0 == NRepeat - 1)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                block_sync_lds();
                                __builtin_amdgcn_sched_barrier(0);
                            }
                            wmma_gemm.Run(a_thread_vec.template AsType<wmma_input_type_a>(),
                                          b_thread_vec.template AsType<wmma_input_type_b>(),
                                          c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                            if constexpr(k0_inner == 0 && m0 == 0 && n0 == 0)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
                                __builtin_amdgcn_sched_barrier(0);
                            }
                        });
                    });
                });
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            });
        };

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                block_sync_lds();
                blockwise_gemm_func();

                b_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                i += 1;
            } while(i < (num_loop - 1));
        }

        // tail
        if constexpr(TailNum == TailNumber::Full)
        {
            block_sync_lds();
            blockwise_gemm_func();
        }
    }

    protected:
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<KPack / A_K1 / A_KRow>{},
                                                Number<MRepeat>{},
                                                Number<KRepeatPerCluster>{},
                                                I1,
                                                I1,
                                                Number<A_K1>{}),
                                     make_tuple(Number<A_K1>{},
                                                Number<KPack / A_KRow>{},
                                                Number<KPack / A_KRow * MRepeat>{},
                                                I0,
                                                I0,
                                                I1));

    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<KPack / B_K1 / B_KRow>{},
                                                Number<NRepeat>{},
                                                Number<KRepeatPerCluster>{},
                                                I1,
                                                I1,
                                                Number<B_K1>{}),
                                     make_tuple(Number<B_K1>{},
                                                Number<KPack / B_KRow>{},
                                                Number<KPack / B_KRow * NRepeat>{},
                                                I0,
                                                I0,
                                                I1));

    using AThreadCopy =
        ThreadwiseTensorSliceTransfer_v4<ADataType,
                                         ComputeTypeA,
                                         decltype(a_block_desc_k0_m0_m1_m2_k1),
                                         decltype(a_thread_desc_),
                                         Sequence<KPack / A_K1 / A_KRow, 1, 1, 1, 1, A_K1>,
                                         Sequence<0, 1, 2, 3, 4, 5>,
                                         5,
                                         A_K1,
                                         A_K1>;

    using BThreadCopy =
        ThreadwiseTensorSliceTransfer_v4<BDataType,
                                         ComputeTypeB,
                                         decltype(b_block_desc_k0_n0_n1_n2_k1),
                                         decltype(b_thread_desc_),
                                         Sequence<KPack / B_K1 / B_KRow, 1, 1, 1, 1, B_K1>,
                                         Sequence<0, 1, 2, 3, 4, 5>,
                                         5,
                                         B_K1,
                                         B_K1>;

    AThreadCopy a_thread_copy_{Base::CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{Base::CalculateBThreadOriginDataIndex()};
    using Base::c_thread_desc_;
};

} // namespace ck
