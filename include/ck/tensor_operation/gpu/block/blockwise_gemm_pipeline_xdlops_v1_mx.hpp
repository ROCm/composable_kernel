// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_mx_pipeline_xdlops_base.hpp"

namespace ck {

// Naive pipeline with lowest resource request per WGP
// GlobalPrefetchStages: 1
// LocalPreFillStages: 1
// LocalPreFetchStages: 0
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
struct BlockwiseGemmXdlops_pipeline_v1_mx
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
struct BlockwiseGemmXdlops_pipeline_v1_mx<BlockGemmPipelineScheduler::Intrawave,
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
    using Base::APackedSize;
    using Base::BMmaKStride;
    using Base::BPackedSize;
    using Base::KThreadChunk;

    using Base::KXdlPack;
    using Base::MXdlPack;
    using Base::NXdlPack;

    using AccType      = typename Base::AccType;
    using Tuple5       = typename Base::Tuple5;
    using ComputeTypeA = typename Base::ComputeTypeA;
    using ComputeTypeB = typename Base::ComputeTypeB;

    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    static constexpr auto ScalesPerKBlockSize =
        KPerBlock / ScaleBlockSize; // How many mx-vectors per K block

    //> How many mx-vectors in each row/col is processed in one call to xdlops_gemm.Run()
    static constexpr auto AScalesPerXdlopsRun =
        (APackedSize * KPack * xdlops_gemm.K0PerXdlops) / ScaleBlockSize;
    static constexpr auto BScalesPerXdlopsRun =
        (BPackedSize * KPack * xdlops_gemm.K0PerXdlops) / ScaleBlockSize;

    //> How many scales a thread must read to accommodate one call to xdlops_gemm.Run()
    static constexpr auto ScalesPerXdlopsRunPerThreadA =
        AScalesPerXdlopsRun / xdlops_gemm.mfma_instr.num_input_blks;
    static constexpr auto ScalesPerXdlopsRunPerThreadB =
        BScalesPerXdlopsRun / xdlops_gemm.mfma_instr.num_input_blks;

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

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
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

        // Global prefetch 1
        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
        b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Prefetch a_scales
        static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
            static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                a_scale_thread_copy.Run(a_scale_grid_desc,
                                        a_scale_grid_buf,
                                        a_scale_thread_desc,
                                        make_tuple(m0, k0, I0),
                                        a_scale_thread_buf);

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

        // Prefetch b_scales
        static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
            static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                b_scale_thread_copy.Run(b_scale_grid_desc,
                                        b_scale_grid_buf,
                                        b_scale_thread_desc,
                                        make_tuple(n0, k0, I0),
                                        b_scale_thread_buf);

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

        // Local prefill 1
        __builtin_amdgcn_s_waitcnt(3952); // wait for EXP_CNT, LDS, GDS, Constant and Message
        block_sync_lds();

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            // loop over k with the step KPerBlock
            index_t i = 0;
            do
            {
                // -------------------------------------------------------------------------------------------

                // wait previous blockwise copy to finish

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
                static_for<0, KRepeat, 1>{}([&](auto k) {
                    constexpr auto k_step =
                        k * xdlops_gemm.KPerXdlops * KPack / xdlops_gemm.K1PerXdlops;

#if 1
                    if(blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 32))
                    {
                        if constexpr(k == 0)
                        {
                            if(threadIdx.x == 0)
                            {
                                printf("MRepeat = %d\n", MRepeat); // 4
                                printf("NRepeat = %d\n", NRepeat); // 4
                                printf("KRepeat = %d\n", KRepeat); // 2
                                printf("KPack = %d\n", KPack);     // 32

                                printf("KXdlPack = %d\n", KXdlPack); //
                                printf("MXdlPack = %d\n", MXdlPack); //
                                printf("NXdlPack = %d\n", NXdlPack); //

                                printf("APackedSize = %d\n", APackedSize); //
                                printf("BPackedSize = %d\n", BPackedSize); //
                                printf("AMmaKStride = %d\n", AMmaKStride); //
                                printf("BMmaKStride = %d\n", BMmaKStride); //

                                printf("a_scale_thread_vec_size = %lu\n",
                                       a_scale_thread_vec_size); //
                                printf("b_scale_thread_vec_size = %lu\n",
                                       b_scale_thread_vec_size); //

                                printf("xdlops_gemm.GetRegSizePerXdlops() = %d\n",
                                       xdlops_gemm.GetRegSizePerXdlops()); // 4
                                printf("mfma_instr.k_per_blk = %d\n",
                                       xdlops_gemm.mfma_instr.k_per_blk); // 32
                                printf("xdlops_gemm.KPerXdlops = %d\n",
                                       xdlops_gemm.KPerXdlops); // 128
                                printf("xdlops_gemm.K1PerXdlops = %d\n",
                                       xdlops_gemm.K1PerXdlops); // 32
                                printf("xdlops_gemm.K0PerXdlops = %d\n\n",
                                       xdlops_gemm.K0PerXdlops); // 2

                                printf("sizeof(ComputeTypeA) = %lu\n\n",
                                       sizeof(ComputeTypeA)); //
                            }
                        }
                    }
#endif
                    // NOTE: KThreadChunk is the number of packed storages
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, xdlops_gemm.K1PerXdlops / APackedSize / KThreadChunk, 1>{}(
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
                                                   a_block_buf,
                                                   a_thread_desc_,
                                                   make_tuple(Number<m0 / MXdlPack>{},
                                                              I0,
                                                              Number<m0 % MXdlPack>{},
                                                              k,
                                                              Number<chunk * KThreadChunk>{}),
                                                   a_thread_buf);
                            });
                    });
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        // read block data in chunks to assemble correct thread vectors
                        static_for<0, xdlops_gemm.K1PerXdlops / BPackedSize / KThreadChunk, 1>{}(
                            [&](auto chunk) {
                                constexpr auto b_k_step_chunk =
                                    k_step +
                                    chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                                b_thread_copy_.Run(b_block_desc_n0_n1_n2_n3_k,
                                                   make_tuple(Number<n0 / NXdlPack>{},
                                                              I0,
                                                              Number<n0 % NXdlPack>{},
                                                              I0,
                                                              Number<b_k_step_chunk>{}),
                                                   b_block_buf,
                                                   b_thread_desc_,
                                                   make_tuple(Number<n0 / NXdlPack>{},
                                                              I0,
                                                              Number<n0 % NXdlPack>{},
                                                              k,
                                                              Number<chunk * KThreadChunk>{}),
                                                   b_thread_buf);
                            });
                    });
                });

                // load for next k loop
                block_sync_lds();
                a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
                b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                    static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                        static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                            constexpr index_t a_scale_offset =
                                a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));
                            constexpr index_t b_scale_offset =
                                b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                            static_assert(0 < ScalesPerXdlopsRunPerThreadA &&
                                              0 < ScalesPerXdlopsRunPerThreadB,
                                          "Must have at least one scale per Xdlops per Thread.");

                            vector_type<AScaleDataType, a_scale_thread_vec_size> a_scale_thread_vec;
                            vector_type<BScaleDataType, b_scale_thread_vec_size> b_scale_thread_vec;

                            // Pack scale_thread_buf into scale_thread_vec
                            static_for<0, a_scale_thread_vec_size, 1>{}([&](auto s) {
                                a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                    a_scale_thread_buf[Number<a_scale_offset + s>{}];
                            });
                            static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                                b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                    b_scale_thread_buf[Number<b_scale_offset + s>{}];
                            });

                            static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                                static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                    static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                        constexpr auto kxdl = ikxdl + k0 * KXdlPack;

                                        vector_type<ComputeTypeA, KPack> a_thread_vec;
                                        vector_type<ComputeTypeB, KPack> b_thread_vec;

                                        bool is_B_zero = true;
                                        bool is_A_zero = true;
                                        ignore         = is_B_zero;
                                        ignore         = is_A_zero;

                                        static_for<0, KPack, 1>{}([&](auto ik) {
                                            a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                                    make_tuple(m0, I0, imxdl, kxdl, ik))>{}];
                                            b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                                    make_tuple(n0, I0, inxdl, kxdl, ik))>{}];

#if 1 // check for zero A and B
                                            if(b_thread_vec.template AsType<ComputeTypeB>()(ik) ==
                                               ComputeTypeB{0})
                                            {
                                            }
                                            else
                                            {
                                                is_B_zero = false;
                                            }
                                            if(a_thread_vec.template AsType<ComputeTypeA>()(ik) ==
                                               ComputeTypeA{0})
                                            {
                                            }
                                            else
                                            {
                                                is_A_zero = false;
                                            }
#endif
                                        });

                                        using mfma_input_type_a = typename vector_type< //
                                            ComputeTypeA,
                                            xdlops_gemm.K1PerXdlops / APackedSize>::type;
                                        using mfma_input_type_b = typename vector_type< //
                                            ComputeTypeB,
                                            xdlops_gemm.K1PerXdlops / BPackedSize>::type;

                                        using mfma_scale_input_type_a = typename vector_type< //
                                            AScaleDataType,
                                            a_scale_thread_vec_size>::type;
                                        using mfma_scale_input_type_b = typename vector_type< //
                                            BScaleDataType,
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
                                            c_thread_buf.GetVectorTypeReference(
                                                Number<c_offset>{}));

                                        bool is_C_zero = true;
                                        ignore         = is_C_zero;
#if 1 // check for zero C
                                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}(
                                            [&](auto m) {
                                                if(c_thread_buf[Number<c_offset + m>{}] == 0.0f) {}
                                                else
                                                {
                                                    is_C_zero = false;
                                                }
                                            });
#endif

#if 1 // disable all output
                                        if(!is_B_zero && !is_A_zero)
                                        {
                                            // First MWaves * MPerXDL rows and NWaves * NPerXDL
                                            // columns
                                            if constexpr(m0 == 0 && n0 == 0 &&
                                                         (k0 == 0 || k0 == 0) &&
                                                         (inxdl == 0 || inxdl == 0) &&
                                                         (imxdl == 0 || imxdl == 0))
                                            {
// print out a_thread_vec
#if 0
                                                printf("blockId = %u; threadId = %u; i = %d; m0 = "
                                                       "%d; n0 = %d; k0 = %d; imxdl = %d; inxdl = "
                                                       "%d; ikxdl = %d :\n\ta_thread_vec = [%f, "
                                                       "%f, %f, %f, "
                                                       "%f, %f, %f, %f, "
                                                       "%f, %f, %f, %f, %f, %f, %f, %f,\n\t\t\t"
                                                       " %f, %f, %f, "
                                                       "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                                       "%f, %f, %f]\n",
                                                       blockIdx.x,
                                                       threadIdx.x,
                                                       i,
                                                       static_cast<int>(m0),
                                                       static_cast<int>(n0),
                                                       static_cast<int>(k0),
                                                       static_cast<int>(imxdl),
                                                       static_cast<int>(inxdl),
                                                       static_cast<int>(ikxdl),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<0>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<1>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<2>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<3>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<4>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<5>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<6>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<7>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<8>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<9>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<10>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<11>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<12>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<13>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<14>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<15>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<16>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<17>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<18>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<19>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<20>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<21>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<22>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<23>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<24>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<25>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<26>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<27>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<28>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<29>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<30>{})),
                                                       type_convert<float>(
                                                           a_thread_vec
                                                               .template AsType<ComputeTypeA>()(
                                                                   Number<31>{})));
#endif
// print out b_thread_vec
#if 0
                                                printf("blockId = %u; threadId = %u; i = %d; m0 = "
                                                       "%d; n0 = %d; k0 = %d; imxdl = %d; inxdl = "
                                                       "%d; ikxdl = %d :\n\tb_thread_vec = [%f, "
                                                       "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                                       "%f, %f, %f, %f, %f,\n\t\t\t  %f, %f, "
                                                       "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                                       "%f, %f, %f, %f]\n",
                                                       blockIdx.x,
                                                       threadIdx.x,
                                                       i,
                                                       static_cast<int>(m0),
                                                       static_cast<int>(n0),
                                                       static_cast<int>(k0),
                                                       static_cast<int>(imxdl),
                                                       static_cast<int>(inxdl),
                                                       static_cast<int>(ikxdl),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<0>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<1>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<2>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<3>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<4>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<5>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<6>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<7>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<8>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<9>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<10>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<11>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<12>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<13>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<14>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<15>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<16>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<17>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<18>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<19>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<20>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<21>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<22>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<23>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<24>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<25>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<26>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<27>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<28>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<29>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<30>{})),
                                                       type_convert<float>(
                                                           b_thread_vec
                                                               .template AsType<ComputeTypeB>()(
                                                                   Number<31>{})));
#endif
#if 0 // print out Scales
                                                if constexpr(a_scale_thread_vec_size == 4)
                                                {
#if 0
                                                    printf("blockId = %u; threadId = %u; i = %d; "
                                                           "m0 = %d : "
                                                           "a_scale_thread_vec[%d,%d] = {%f, %f, "
                                                           "%f, %f}\n",
                                                           blockIdx.x,
                                                           threadIdx.x,
                                                           i,
                                                           static_cast<int>(m0),
                                                           static_cast<int>(n0),
                                                           static_cast<int>(k0),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<0>{}]),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<1>{}]),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<2>{}]),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<3>{}]));
#endif
                                                    printf(
                                                        "blockId = %u; threadId = %u; i = %d; m0 = "
                                                        "%d; n0 = %d; k0 = %d; imxdl = %d; inxdl = "
                                                        "%d; ikxdl = %d; OpselB = %d: "
                                                        "b_scale_thread_vec = {%f, "
                                                        "%f, %f, %f}\n",
                                                        blockIdx.x,
                                                        threadIdx.x,
                                                        i,
                                                        static_cast<int>(m0),
                                                        static_cast<int>(n0),
                                                        static_cast<int>(k0),
                                                        static_cast<int>(imxdl),
                                                        static_cast<int>(inxdl),
                                                        static_cast<int>(ikxdl),
                                                        ikxdl * NXdlPack + inxdl,
                                                        type_convert<float>(
                                                            b_scale_thread_vec.template AsType<
                                                                BScaleDataType>()[Number<0>{}]),
                                                        type_convert<float>(
                                                            b_scale_thread_vec.template AsType<
                                                                BScaleDataType>()[Number<1>{}]),
                                                        type_convert<float>(
                                                            b_scale_thread_vec.template AsType<
                                                                BScaleDataType>()[Number<2>{}]),
                                                        type_convert<float>(
                                                            b_scale_thread_vec.template AsType<
                                                                BScaleDataType>()[Number<3>{}]));
                                                }
                                                else if constexpr(a_scale_thread_vec_size == 1)
                                                {
                                                    printf("blockId = %u; threadId = %u; i = %d; "
                                                           "m0 = %d : "
                                                           "a_scale_thread_vec[%d,%d] = {%f}\n",
                                                           blockIdx.x,
                                                           threadIdx.x,
                                                           i,
                                                           static_cast<int>(m0),
                                                           static_cast<int>(n0),
                                                           static_cast<int>(k0),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<0>{}]));

                                                    printf("blockId = %u; threadId = %u; i = %d; "
                                                           "m0 = %d : "
                                                           "b_scale_thread_vec[%d,%d] = {%f}\n",
                                                           blockIdx.x,
                                                           threadIdx.x,
                                                           i,
                                                           static_cast<int>(m0),
                                                           static_cast<int>(n0),
                                                           static_cast<int>(k0),
                                                           type_convert<float>(
                                                               b_scale_thread_vec.template AsType<
                                                                   BScaleDataType>()[Number<0>{}]));
                                                }
#endif
                                            }
                                        }
                                        if(!is_C_zero)
                                        {
                                            // First MWaves * MPerXDL rows and NWaves * NPerXDL
                                            // columns
                                            if constexpr(m0 == 0 && n0 == 0)
                                            {
                                                // print out c_thread_buf_per_scale
#if 0 // print out C

                                                printf("blockId = %u; threadId = %u; i = %d; m0 = "
                                                       "%d; n0 = %d; k0 = %d; imxdl = %d; inxdl = "
                                                       "%d; ikxdl = %d :\n\tc_thread_buf = [%f, "
                                                       "%f, %f, %f, "
                                                       "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                                       "%f, %f]\n",
                                                       blockIdx.x,
                                                       threadIdx.x,
                                                       i,
                                                       static_cast<int>(m0),
                                                       static_cast<int>(n0),
                                                       static_cast<int>(k0),
                                                       static_cast<int>(imxdl),
                                                       static_cast<int>(inxdl),
                                                       static_cast<int>(ikxdl),
                                                       c_thread_buf[Number<c_offset + 0>{}],
                                                       c_thread_buf[Number<c_offset + 1>{}],
                                                       c_thread_buf[Number<c_offset + 2>{}],
                                                       c_thread_buf[Number<c_offset + 3>{}],
                                                       c_thread_buf[Number<c_offset + 4>{}],
                                                       c_thread_buf[Number<c_offset + 5>{}],
                                                       c_thread_buf[Number<c_offset + 6>{}],
                                                       c_thread_buf[Number<c_offset + 7>{}],
                                                       c_thread_buf[Number<c_offset + 8>{}],
                                                       c_thread_buf[Number<c_offset + 9>{}],
                                                       c_thread_buf[Number<c_offset + 10>{}],
                                                       c_thread_buf[Number<c_offset + 11>{}],
                                                       c_thread_buf[Number<c_offset + 12>{}],
                                                       c_thread_buf[Number<c_offset + 13>{}],
                                                       c_thread_buf[Number<c_offset + 14>{}],
                                                       c_thread_buf[Number<c_offset + 15>{}]);
#endif
                                            }
                                        }

#endif
                                    });
                                });
                            });
                        });
                    });
                });

                // Prefetch a_scales
                static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                    static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                        a_scale_thread_copy.Run(a_scale_grid_desc,
                                                a_scale_grid_buf,
                                                a_scale_thread_desc,
                                                make_tuple(m0, k0, I0),
                                                a_scale_thread_buf);

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

                // Prefetch b_scales
                static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                    static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                        b_scale_thread_copy.Run(b_scale_grid_desc,
                                                b_scale_grid_buf,
                                                b_scale_thread_desc,
                                                make_tuple(n0, k0, I0),
                                                b_scale_thread_buf);

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

                __builtin_amdgcn_s_waitcnt(3952); // wait for EXP_CNT and LGKM_CNT
                block_sync_lds();

                i += 1;
            } while(i < (num_loop - 1));
        }

        // tail
        if constexpr(TailNum == TailNumber::Full)
        {
            static_for<0, KRepeat, 1>{}([&](auto k) {
                constexpr auto k_step =
                    k * xdlops_gemm.KPerXdlops * KPack / xdlops_gemm.K1PerXdlops;

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, xdlops_gemm.K1PerXdlops / APackedSize / KThreadChunk, 1>{}(
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
                                               a_block_buf,
                                               a_thread_desc_,
                                               make_tuple(Number<m0 / MXdlPack>{},
                                                          I0,
                                                          Number<m0 % MXdlPack>{},
                                                          k,
                                                          Number<chunk * KThreadChunk>{}),
                                               a_thread_buf);
                        });
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    // read block data in chunks to assemble correct thread vectors
                    static_for<0, xdlops_gemm.K1PerXdlops / BPackedSize / KThreadChunk, 1>{}(
                        [&](auto chunk) {
                            constexpr auto b_k_step_chunk =
                                k_step +
                                chunk * KThreadChunk * xdlops_gemm.mfma_instr.num_input_blks;
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_n3_k,
                                               make_tuple(Number<n0 / NXdlPack>{},
                                                          I0,
                                                          Number<n0 % NXdlPack>{},
                                                          I0,
                                                          Number<b_k_step_chunk>{}),
                                               b_block_buf,
                                               b_thread_desc_,
                                               make_tuple(Number<n0 / NXdlPack>{},
                                                          I0,
                                                          Number<n0 % NXdlPack>{},
                                                          k,
                                                          Number<chunk * KThreadChunk>{}),
                                               b_thread_buf);
                        });
                });
            });
            static_for<0, MRepeat / MXdlPack, 1>{}([&](auto m0) {
                static_for<0, NRepeat / NXdlPack, 1>{}([&](auto n0) {
                    static_for<0, KRepeat / KXdlPack, 1>{}([&](auto k0) {
                        constexpr index_t a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0));
                        constexpr index_t b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0, I0));

                        static_assert(0 < ScalesPerXdlopsRunPerThreadA &&
                                          0 < ScalesPerXdlopsRunPerThreadB,
                                      "Must have at least one scale per Xdlops per Thread.");

                        vector_type<AScaleDataType, a_scale_thread_vec_size> a_scale_thread_vec;
                        vector_type<BScaleDataType, b_scale_thread_vec_size> b_scale_thread_vec;

                        // Pack scale_thread_buf into scale_thread_vec
                        static_for<0, a_scale_thread_vec_size, 1>{}([&](auto s) {
                            a_scale_thread_vec.template AsType<AScaleDataType>()(s) =
                                a_scale_thread_buf[Number<a_scale_offset + s>{}];
                        });
                        static_for<0, b_scale_thread_vec_size, 1>{}([&](auto s) {
                            b_scale_thread_vec.template AsType<BScaleDataType>()(s) =
                                b_scale_thread_buf[Number<b_scale_offset + s>{}];
                        });

                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    constexpr auto kxdl = ikxdl + k0 * KXdlPack;

                                    vector_type<ComputeTypeA, KPack> a_thread_vec;
                                    vector_type<ComputeTypeB, KPack> b_thread_vec;

                                    bool is_B_zero = true;
                                    bool is_A_zero = true;
                                    ignore         = is_B_zero;
                                    ignore         = is_A_zero;

                                    static_for<0, KPack, 1>{}([&](auto ik) {
                                        a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                                make_tuple(m0, I0, imxdl, kxdl, ik))>{}];
                                        b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                            b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                                make_tuple(n0, I0, inxdl, kxdl, ik))>{}];

#if 1 // check for zero A and B
                                        if(b_thread_vec.template AsType<ComputeTypeB>()(ik) ==
                                           ComputeTypeB{0})
                                        {
                                        }
                                        else
                                        {
                                            is_B_zero = false;
                                        }
                                        if(a_thread_vec.template AsType<ComputeTypeA>()(ik) ==
                                           ComputeTypeA{0})
                                        {
                                        }
                                        else
                                        {
                                            is_A_zero = false;
                                        }
#endif
                                    });

                                    using mfma_input_type_a = typename vector_type< //
                                        ComputeTypeA,
                                        xdlops_gemm.K1PerXdlops / APackedSize>::type;
                                    using mfma_input_type_b = typename vector_type< //
                                        ComputeTypeB,
                                        xdlops_gemm.K1PerXdlops / BPackedSize>::type;

                                    using mfma_scale_input_type_a = typename vector_type< //
                                        AScaleDataType,
                                        a_scale_thread_vec_size>::type;
                                    using mfma_scale_input_type_b = typename vector_type< //
                                        BScaleDataType,
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

                                    bool is_C_zero = true;
                                    ignore         = is_C_zero;
#if 1 // check for zero C
                                    static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}(
                                        [&](auto m) {
                                            if(c_thread_buf[Number<c_offset + m>{}] == 0.0f) {}
                                            else
                                            {
                                                is_C_zero = false;
                                            }
                                        });
#endif

#if 1 // disable all output
                                    if(!is_B_zero && !is_A_zero)
                                    {
                                        // First MWaves * MPerXDL rows and NWaves * NPerXDL
                                        // columns
                                        if constexpr(m0 == 0 && n0 == 0 && (k0 == 0 || k0 == 0) &&
                                                     (inxdl == 0 || inxdl == 0) &&
                                                     (imxdl == 0 || imxdl == 0))
                                        {
// print out a_thread_vec
#if 0
                                            printf("blockId = %u; threadId = %u; i = %d; m0 = "
                                                   "%d; n0 = %d; k0 = %d; imxdl = %d; inxdl = "
                                                   "%d; ikxdl = %d :\n\ta_thread_vec = [%f, "
                                                   "%f, %f, %f, "
                                                   "%f, %f, %f, %f, "
                                                   "%f, %f, %f, %f, %f, %f, %f, %f,\n\t\t\t"
                                                   " %f, %f, %f, "
                                                   "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                                   "%f, %f, %f]\n",
                                                   blockIdx.x,
                                                   threadIdx.x,
                                                   -1,
                                                   static_cast<int>(m0),
                                                   static_cast<int>(n0),
                                                   static_cast<int>(k0),
                                                   static_cast<int>(imxdl),
                                                   static_cast<int>(inxdl),
                                                   static_cast<int>(ikxdl),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<0>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<1>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<2>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<3>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<4>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<5>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<6>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<7>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<8>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<9>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<10>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<11>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<12>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<13>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<14>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<15>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<16>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<17>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<18>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<19>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<20>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<21>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<22>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<23>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<24>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<25>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<26>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<27>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<28>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<29>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<30>{})),
                                                   type_convert<float>(
                                                       a_thread_vec.template AsType<ComputeTypeA>()(
                                                           Number<31>{})));
#endif
// print out b_thread_vec
#if 0
                                            printf("blockId = %u; threadId = %u; i = %d; m0 = "
                                                   "%d; n0 = %d; k0 = %d; imxdl = %d; inxdl = "
                                                   "%d; ikxdl = %d :\n\tb_thread_vec = [%f, "
                                                   "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                                   "%f, %f, %f, %f, %f,\n\t\t\t  %f, %f, "
                                                   "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                                   "%f, %f, %f, %f]\n",
                                                   blockIdx.x,
                                                   threadIdx.x,
                                                   -1,
                                                   static_cast<int>(m0),
                                                   static_cast<int>(n0),
                                                   static_cast<int>(k0),
                                                   static_cast<int>(imxdl),
                                                   static_cast<int>(inxdl),
                                                   static_cast<int>(ikxdl),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<0>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<1>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<2>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<3>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<4>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<5>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<6>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<7>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<8>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<9>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<10>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<11>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<12>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<13>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<14>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<15>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<16>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<17>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<18>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<19>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<20>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<21>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<22>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<23>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<24>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<25>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<26>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<27>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<28>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<29>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<30>{})),
                                                   type_convert<float>(
                                                       b_thread_vec.template AsType<ComputeTypeB>()(
                                                           Number<31>{})));
#endif
#if 0 // print out Scales
                                            if constexpr(a_scale_thread_vec_size == 4)
                                            {
#if 0
                                                    printf("blockId = %u; threadId = %u; i = %d; "
                                                           "m0 = %d : "
                                                           "a_scale_thread_vec[%d,%d] = {%f, %f, "
                                                           "%f, %f}\n",
                                                           blockIdx.x,
                                                           threadIdx.x,
                                                           -1,
                                                           static_cast<int>(m0),
                                                           static_cast<int>(n0),
                                                           static_cast<int>(k0),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<0>{}]),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<1>{}]),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<2>{}]),
                                                           type_convert<float>(
                                                               a_scale_thread_vec.template AsType<
                                                                   AScaleDataType>()[Number<3>{}]));
#endif
                                                printf("blockId = %u; threadId = %u; i = %d; m0 = "
                                                       "%d; n0 = %d; k0 = %d; imxdl = %d; inxdl = "
                                                       "%d; ikxdl = %d; OpselB = %d: "
                                                       "b_scale_thread_vec = {%f, "
                                                       "%f, %f, %f}\n",
                                                       blockIdx.x,
                                                       threadIdx.x,
                                                       -1,
                                                       static_cast<int>(m0),
                                                       static_cast<int>(n0),
                                                       static_cast<int>(k0),
                                                       static_cast<int>(imxdl),
                                                       static_cast<int>(inxdl),
                                                       static_cast<int>(ikxdl),
                                                       ikxdl * NXdlPack + inxdl,
                                                       type_convert<float>(
                                                           b_scale_thread_vec.template AsType<
                                                               BScaleDataType>()[Number<0>{}]),
                                                       type_convert<float>(
                                                           b_scale_thread_vec.template AsType<
                                                               BScaleDataType>()[Number<1>{}]),
                                                       type_convert<float>(
                                                           b_scale_thread_vec.template AsType<
                                                               BScaleDataType>()[Number<2>{}]),
                                                       type_convert<float>(
                                                           b_scale_thread_vec.template AsType<
                                                               BScaleDataType>()[Number<3>{}]));
                                            }
                                            else if constexpr(a_scale_thread_vec_size == 1)
                                            {
                                                printf("blockId = %u; threadId = %u; i = %d; "
                                                       "m0 = %d : "
                                                       "a_scale_thread_vec[%d,%d] = {%f}\n",
                                                       blockIdx.x,
                                                       threadIdx.x,
                                                       -1,
                                                       static_cast<int>(m0),
                                                       static_cast<int>(n0),
                                                       static_cast<int>(k0),
                                                       type_convert<float>(
                                                           a_scale_thread_vec.template AsType<
                                                               AScaleDataType>()[Number<0>{}]));

                                                printf("blockId = %u; threadId = %u; i = %d; "
                                                       "m0 = %d : "
                                                       "b_scale_thread_vec[%d,%d] = {%f}\n",
                                                       blockIdx.x,
                                                       threadIdx.x,
                                                       -1,
                                                       static_cast<int>(m0),
                                                       static_cast<int>(n0),
                                                       static_cast<int>(k0),
                                                       type_convert<float>(
                                                           b_scale_thread_vec.template AsType<
                                                               BScaleDataType>()[Number<0>{}]));
                                            }
#endif
                                        }
                                    }
                                    if(!is_C_zero)
                                    {
                                        // First MWaves * MPerXDL rows and NWaves * NPerXDL
                                        // columns
                                        if constexpr(m0 == 0 && n0 == 0)
                                        {
                                            // print out c_thread_buf_per_scale
#if 0 // print out C

                                            printf("blockId = %u; threadId = %u; i = %d; m0 = "
                                                   "%d; n0 = %d; k0 = %d; imxdl = %d; inxdl = "
                                                   "%d; ikxdl = %d :\n\tc_thread_buf = [%f, "
                                                   "%f, %f, %f, "
                                                   "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                                   "%f, %f]\n",
                                                   blockIdx.x,
                                                   threadIdx.x,
                                                   -1,
                                                   static_cast<int>(m0),
                                                   static_cast<int>(n0),
                                                   static_cast<int>(k0),
                                                   static_cast<int>(imxdl),
                                                   static_cast<int>(inxdl),
                                                   static_cast<int>(ikxdl),
                                                   c_thread_buf[Number<c_offset + 0>{}],
                                                   c_thread_buf[Number<c_offset + 1>{}],
                                                   c_thread_buf[Number<c_offset + 2>{}],
                                                   c_thread_buf[Number<c_offset + 3>{}],
                                                   c_thread_buf[Number<c_offset + 4>{}],
                                                   c_thread_buf[Number<c_offset + 5>{}],
                                                   c_thread_buf[Number<c_offset + 6>{}],
                                                   c_thread_buf[Number<c_offset + 7>{}],
                                                   c_thread_buf[Number<c_offset + 8>{}],
                                                   c_thread_buf[Number<c_offset + 9>{}],
                                                   c_thread_buf[Number<c_offset + 10>{}],
                                                   c_thread_buf[Number<c_offset + 11>{}],
                                                   c_thread_buf[Number<c_offset + 12>{}],
                                                   c_thread_buf[Number<c_offset + 13>{}],
                                                   c_thread_buf[Number<c_offset + 14>{}],
                                                   c_thread_buf[Number<c_offset + 15>{}]);
#endif
                                        }
                                    }

#endif
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
                   Number<ScalesPerXdlopsRunPerThreadA * a_scale_thread_vec_size>{}));

    // TODO: make this field protected when b_scale_thread_copy_ is moved
    // here
    static constexpr auto b_scale_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<NRepeat / NXdlPack>{},
                   Number<KRepeat / KXdlPack>{},
                   Number<ScalesPerXdlopsRunPerThreadB * b_scale_thread_vec_size>{}));

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
};

} // namespace ck
