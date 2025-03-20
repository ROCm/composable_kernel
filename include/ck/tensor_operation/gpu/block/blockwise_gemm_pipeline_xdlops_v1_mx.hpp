// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp"

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
          typename ComputeDataType,
          typename AccDataType,
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
          typename ComputeDataType,
          typename AccDataType,
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
          index_t KPack
          // ,bool TransposeC //disable transposec right now...
          >
struct BlockwiseGemmXdlops_pipeline_v1_mx<BlockGemmPipelineScheduler::Intrawave,
                                          ThreadBlockSize,
                                          ScaleBlockSize,
                                          ADataType,
                                          AScaleDataType,
                                          BDataType,
                                          BScaleDataType,
                                          ComputeDataType,
                                          AccDataType,
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
    : BlockwiseGemmXdlops_pipeline_base<ThreadBlockSize,
                                        ADataType,
                                        BDataType,
                                        ComputeDataType,
                                        AccDataType,
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
    using Base = BlockwiseGemmXdlops_pipeline_base<ThreadBlockSize,
                                                   ADataType,
                                                   BDataType,
                                                   ComputeDataType,
                                                   AccDataType,
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
    using Base::CalculateCThreadOriginDataIndex8D;
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

    using Tuple4 = typename Base::Tuple4;

    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    static constexpr auto ScalesPerKBlockSize =
        KPerBlock / ScaleBlockSize; // How many mx-vectors per K block size

    __host__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        ignore = num_loop;
        return TailNumber::Full;
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];

        const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

#if 0
        uint32_t i = 0;
        if((threadIdx.x == 0 + i || threadIdx.x == 32 + i) && blockIdx.x == 0)
        {
            printf("threadIdx = %u; xdlops_a_idx[I0] = %d; xdlops_a_idx[I1] = %d\n",
                   threadIdx.x,
                   xdlops_a_idx[I0],
                   xdlops_a_idx[I1]);
        }
#endif

        return make_tuple(0, waveId_m, xdlops_a_idx[I1], xdlops_gemm.KPerXdlops * xdlops_a_idx[I0]);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();
#if 0
        uint32_t i = 0;
        if((threadIdx.x == 0 + i || threadIdx.x == 32 + i) && blockIdx.x == 0)
        {

            printf("threadIdx = %u; xdlops_b_idx[I0] = %d; xdlops_b_idx[I1] = %d\n",
                   threadIdx.x,
                   xdlops_b_idx[I0],
                   xdlops_b_idx[I1]);
        }
#endif
        return make_tuple(0, waveId_n, xdlops_b_idx[I1], xdlops_gemm.KPerXdlops * xdlops_b_idx[I0]);
    }

    /**
     * @brief Constructor for BlockwiseGemmXdlops_pipeline_v1_mx.
     *
     * The primary purpose of this constructor is to modify default initialization of the base class
     * with the origin data index suitable for microscaling.
     *
     * @param a_origin The origin data index for matrix A.
     * @param b_origin The origin data index for matrix B.
     *
     */
    __host__ __device__
    BlockwiseGemmXdlops_pipeline_v1_mx(Tuple4 a_origin = CalculateAThreadOriginDataIndex(),
                                       Tuple4 b_origin = CalculateBThreadOriginDataIndex())
        : Base(a_origin, b_origin)
    {
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
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            b_thread_desc_.GetElementSpaceSize());

        auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
            a_scale_thread_desc.GetElementSpaceSize());

        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BScaleDataType>(
            b_scale_thread_desc.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        static_assert(xdlops_gemm.mfma_instr.num_groups_per_blk *
                              xdlops_gemm.mfma_instr.group_size ==
                          xdlops_gemm.GetRegSizePerXdlops(),
                      "Assume num_regs_per_blk == num_groups_per_blk * group_size");

        // Prefetch a_scales
        static_for<0, MRepeat, 1>{}([&](auto m0) {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, xdlops_gemm.mfma_instr.num_groups_per_blk, 1>{}([&](auto g) {
                    auto a_scale_thread_buf_group =
                        make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
                            a_scale_thread_desc_group.GetElementSpaceSize());

                    a_scale_thread_copy.Run(a_scale_grid_desc,
                                            a_scale_grid_buf,
                                            a_scale_thread_desc_group,
                                            make_tuple(I0, I0),
                                            a_scale_thread_buf_group);
#if 0
                    if constexpr(m0 == 0)
                    {
                        if(blockIdx.x == 0 && (threadIdx.x == 128 || threadIdx.x == 138))
                        {
                            printf("blockId = %u; threadId = %u; {m0,k0,g} = {%d,%d,%d} : "
                                   "a_scale_thread_buf_group = "
                                   "[%f, %f, "
                                   "%f, %f]\n",
                                   blockIdx.x,
                                   threadIdx.x,
                                   static_cast<int>(m0),
                                   static_cast<int>(k0),
                                   static_cast<int>(g),
                                   type_convert<float>(a_scale_thread_buf_group(Number<0>{})),
                                   type_convert<float>(a_scale_thread_buf_group(Number<1>{})),
                                   type_convert<float>(a_scale_thread_buf_group(Number<2>{})),
                                   type_convert<float>(a_scale_thread_buf_group(Number<3>{})));
                        }
                    }
#endif
                    static_for<0, xdlops_gemm.mfma_instr.group_size, 1>{}([&](auto i) {
                        constexpr index_t a_scale_offset =
                            a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, g, i));
                        a_scale_thread_buf(Number<a_scale_offset>{}) =
                            a_scale_thread_buf_group[Number<i>{}];
                    });
                    // go to the next group
                    a_scale_thread_copy.MoveSrcSliceWindow(
                        a_scale_grid_desc,
                        make_multi_index(2 * xdlops_gemm.mfma_instr.group_size, 0));
                }); // g

                // restore row id and advance to the next scale
                a_scale_thread_copy.MoveSrcSliceWindow(
                    a_scale_grid_desc,
                    make_multi_index(-2 * xdlops_gemm.mfma_instr.group_size *
                                         xdlops_gemm.mfma_instr.num_groups_per_blk,
                                     1));
            }); // k0

            // restore column id and advance to the next set of rows
            a_scale_thread_copy.MoveSrcSliceWindow(
                a_scale_grid_desc, make_multi_index(MWaves * MPerXDL, -ScalesPerKBlockSize));
        }); // m0

        // restore row id and advance to the next set of scales
        a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                               make_multi_index(-MPerBlock, ScalesPerKBlockSize));

        // Prefetch b_scales
        static_for<0, NRepeat, 1>{}([&](auto n0) {
            b_scale_thread_copy.Run(b_scale_grid_desc,
                                    b_scale_grid_buf,
                                    b_scale_thread_desc,
                                    make_tuple(n0, I0),
                                    b_scale_thread_buf);
            b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                   make_multi_index(NWaves * NPerXDL, 0));
        });

        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                               make_multi_index(-NPerBlock, ScalesPerKBlockSize));

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // Initialize C
        c_thread_buf.Clear();

        auto c_thread_buf_per_scale = remove_cvref_t<decltype(c_thread_buf)>();

#if 0
        [[maybe_unused]] auto print_type_name = [](const char* msg, auto param [[maybe_unused]]) {
            printf("%s = %s\n\n", msg, __PRETTY_FUNCTION__);
        };

        if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
        {

            print_type_name("xdlops_gemm", xdlops_gemm); // ck::XdlopsGemm<_BitInt(8), 32, 32, 16>
            using mfma_input_type =
                typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;
            print_type_name("mfma_input_type",
                            mfma_input_type{}); //_BitInt(8) __attribute__((ext_vector_type(8)))

            print_type_name("mfma = ", xdlops_gemm.mfma); // ck::MfmaSelector<_BitInt(8), 32, 32>
            print_type_name(
                "mfma_instr = ",
                xdlops_gemm.mfma_instr); // ck::mfma_type<MfmaInstr::mfma_f32_32x32x16f8f8>

            printf("MRepeat = %d\n", MRepeat); // 2
            printf("NRepeat = %d\n", NRepeat); // 2
            printf("KRepeat = %d\n", KRepeat); // 2
            printf("KPack = %d\n", KPack);     // 16
            printf("xdlops_gemm.GetRegSizePerXdlops() = %d\n",
                   xdlops_gemm.GetRegSizePerXdlops());                               // 16
            printf("mfma_instr.k_per_blk = %d\n", xdlops_gemm.mfma_instr.k_per_blk); // 8
        }
#endif
#if 0
        if((threadIdx.x == 0 || threadIdx.x == 32) && blockIdx.x == 0)
        {
            const void* ap = reinterpret_cast<const int*>(&a_block_buf);
            const void* bp = reinterpret_cast<const int*>(&b_block_buf);
            printf("threadIdx = %u; &a_block_buf = %p; &b_block_buf = %p\n\n", threadIdx.x, ap, bp);
        }
#endif
        // main body
        if constexpr(HasMainLoop)
        {
            // loop over k with the step KPerBlock
            index_t i = 0;
            do
            {
                // -------------------------------------------------------------------------------------------
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                block_sync_lds();
// print out a_scale_thread_buf
#if 0
                if(blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 10 ||
                                       threadIdx.x == 42))
                {
                    printf("blockId = %u; threadId = %u; i = %d : a_scale_thread_buf = [%f, %f, "
                           "%f, %f]\n",
                           blockIdx.x,
                           threadIdx.x,
                           i,
                           type_convert<float>(a_scale_thread_buf(Number<0>{})),
                           type_convert<float>(a_scale_thread_buf(Number<1>{})),
                           type_convert<float>(a_scale_thread_buf(Number<2>{})),
                           type_convert<float>(a_scale_thread_buf(Number<3>{})));
                }
#endif

#if 0
                if(blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 10 ||
                                       threadIdx.x == 42))
                {
                    printf("blockId = %u; threadId = %u; i = %d : b_scale_thread_buf = [%f, %f, "
                           "%f, %f]\n",
                           blockIdx.x,
                           threadIdx.x,
                           i,
                           type_convert<float>(b_scale_thread_buf(Number<0>{})),
                           type_convert<float>(b_scale_thread_buf(Number<1>{})),
                           type_convert<float>(b_scale_thread_buf(Number<2>{})),
                           type_convert<float>(b_scale_thread_buf(Number<3>{})));
                }
#endif

                static_for<0, KRepeat, 1>{}([&](auto k) {
                    constexpr auto a_k_step = k * AMmaKStride * KPack / xdlops_gemm.K1PerXdlops;
                    constexpr auto b_k_step = k * BMmaKStride * KPack / xdlops_gemm.K1PerXdlops;

                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<a_k_step>{}),
                                           a_block_buf,
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k, I0),
                                           a_thread_buf);
                    });
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<b_k_step>{}),
                                           b_block_buf,
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_buf);
                    });
                });

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
                            c_thread_buf_per_scale.Clear();
                            vector_type<ComputeDataType, KPack> a_thread_vec;
                            vector_type<ComputeDataType, KPack> b_thread_vec;

                            bool is_B_zero = true;
                            bool is_A_zero = true;
                            ignore         = is_B_zero;
                            ignore         = is_A_zero;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, I0, k0, ik))>{}];
                                b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0, I0, k0, ik))>{}];
#if 1
                                if(b_thread_vec.template AsType<ComputeDataType>()(ik) ==
                                   ComputeDataType{0})
                                {
                                }
                                else
                                {
                                    is_B_zero = false;
                                }
                                if(a_thread_vec.template AsType<ComputeDataType>()(ik) ==
                                   ComputeDataType{0})
                                {
                                }
                                else
                                {
                                    is_A_zero = false;
                                }
#endif
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataType,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            // MFMA accumulation
                            // m = 1:MPerXDL
                            //   n = 1:NPerXDL
                            //     k = 1:KPack
                            //       c(m,n) += a(m,k)*b(k,n)
                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(I0));

                            bool is_C_zero = true;
                            ignore         = is_C_zero;

                            static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto m) {
                                if(c_thread_buf_per_scale[Number<m>{}] == 0.0f) {}
                                else
                                {
                                    is_C_zero = false;
                                }
                            });

                            // one scale per k0
                            constexpr index_t b_scale_offset =
                                b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0));
#if 0
                            if constexpr(m0 == 0 && n0 == 0 && k0 == 0)
                            {
                                if(blockIdx.x == 0 && i == 0 &&
                                   (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 64 ||
                                    threadIdx.x == 96 || threadIdx.x == 128 || threadIdx.x == 160 ||
                                    threadIdx.x == 192 || threadIdx.x == 224))
                                {
                                    const auto wave_idx = GetWaveIdx();
                                    const auto waveId_m = wave_idx[I0];
                                    const auto waveId_n = wave_idx[I1];

                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : waveId = [%d, %d]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        waveId_m,
                                        waveId_n);

                                    // printf(
                                    //     "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d;
                                    //     k0 "
                                    //     "= %d : a_thread_vec = [%f]; b_thread_vec = [%f]\n",
                                    //     blockIdx.x,
                                    //     threadIdx.x,
                                    //     i,
                                    //     static_cast<int>(m0),
                                    //     static_cast<int>(n0),
                                    //     static_cast<int>(k0),
                                    //     type_convert<float>(
                                    //         a_thread_vec.template AsType<ComputeDataType>()(
                                    //             Number<0>{})),
                                    //     type_convert<float>(
                                    //         b_thread_vec.template AsType<ComputeDataType>()(
                                    //             Number<0>{})));
                                }
                            }
#endif
// print out a_thread_vec
#if 0
                            if(!is_A_zero && blockIdx.x == 0 && threadIdx.x == 138)
                            {
                                // First MWaves * MPerXDL rows and NWaves * NPerXDL columns
                                if constexpr(m0 == 0 && n0 == 0)
                                {
                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : a_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                        "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<0>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<1>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<2>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<3>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<4>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<5>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<6>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<7>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<8>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<9>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<10>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<11>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<12>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<13>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<14>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<15>{})));

// print out b_thread_vec
#if 0
                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : b_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                        "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<0>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<1>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<2>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<3>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<4>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<5>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<6>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<7>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<8>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<9>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<10>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<11>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<12>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<13>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<14>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<15>{})));
                                    printf("blockId = %u; threadId = %u; i = %d; m0 = %d : "
                                           "b_scale_thread_buf[%d,%d] = %f\n",
                                           blockIdx.x,
                                           threadIdx.x,
                                           i,
                                           static_cast<int>(m0),
                                           static_cast<int>(n0),
                                           static_cast<int>(k0),
                                           type_convert<float>(
                                               b_scale_thread_buf[Number<b_scale_offset>{}]));
#endif
                                }
                            }
                            if(!is_C_zero && threadIdx.x == 128)
                            {
                                // First MWaves * MPerXDL rows and NWaves * NPerXDL columns
                                if constexpr(m0 == 0 && n0 == 0)
                                {
// print out c_thread_buf_per_scale
#if 1
                                    constexpr index_t a_scale_offset =
                                        a_scale_thread_desc.CalculateOffset(
                                            make_tuple(m0, k0, I0, I0));

                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d :\n\tc_thread_buf_per_scale = [%f, %f, %f, %f, %f, "
                                        "%f, "
                                        "%f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                        "%f]\n\t    a_scale_thread_buf "
                                        "= [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                        "%f, "
                                        "%f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        c_thread_buf_per_scale(Number<0>{}),
                                        c_thread_buf_per_scale(Number<1>{}),
                                        c_thread_buf_per_scale(Number<2>{}),
                                        c_thread_buf_per_scale(Number<3>{}),
                                        c_thread_buf_per_scale(Number<4>{}),
                                        c_thread_buf_per_scale(Number<5>{}),
                                        c_thread_buf_per_scale(Number<6>{}),
                                        c_thread_buf_per_scale(Number<7>{}),
                                        c_thread_buf_per_scale(Number<8>{}),
                                        c_thread_buf_per_scale(Number<9>{}),
                                        c_thread_buf_per_scale(Number<10>{}),
                                        c_thread_buf_per_scale(Number<11>{}),
                                        c_thread_buf_per_scale(Number<12>{}),
                                        c_thread_buf_per_scale(Number<13>{}),
                                        c_thread_buf_per_scale(Number<14>{}),
                                        c_thread_buf_per_scale(Number<15>{}),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 0>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 1>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 2>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 3>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 4>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 5>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 6>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 7>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 8>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 9>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 10>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 11>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 12>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 13>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 14>{}]),
                                        type_convert<float>(
                                            a_scale_thread_buf[Number<a_scale_offset + 15>{}]));
#endif
                                }
                            }
#endif

#if 0
                            // for b_col == 74
                            if constexpr(m0 == 0 && n0 == 1)
                            {
                                if(blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 32))
                                {
                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : a_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                        "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<0>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<1>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<2>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<3>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<4>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<5>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<6>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<7>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<8>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<9>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<10>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<11>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<12>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<13>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<14>{})),
                                        type_convert<float>(
                                            a_thread_vec.template AsType<ComputeDataType>()(
                                                Number<15>{})));
                                }

                                if(blockIdx.x == 0 && (threadIdx.x == 10 || threadIdx.x == 42))
                                {
                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : b_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                        "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<0>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<1>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<2>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<3>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<4>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<5>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<6>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<7>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<8>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<9>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<10>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<11>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<12>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<13>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<14>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<15>{})));

                                    printf("blockId = %u; threadId = %u; i = %d; m0 = %d : "
                                           "b_scale_thread_buf[%d,%d] = %f\n",
                                           blockIdx.x,
                                           threadIdx.x,
                                           i,
                                           static_cast<int>(m0),
                                           static_cast<int>(n0),
                                           static_cast<int>(k0),
                                           type_convert<float>(
                                               b_scale_thread_buf[Number<b_scale_offset>{}]));
                                }
#if 1
                                if(!is_C_zero)
                                {
                                    printf("blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = "
                                           "%d; k0 "
                                           "= %d : c_thread_buf_per_scale = [%f, %f, %f, %f, "
                                           "%f, %f, "
                                           "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
                                           blockIdx.x,
                                           threadIdx.x,
                                           i,
                                           static_cast<int>(m0),
                                           static_cast<int>(n0),
                                           static_cast<int>(k0),
                                           c_thread_buf_per_scale(Number<0>{}),
                                           c_thread_buf_per_scale(Number<1>{}),
                                           c_thread_buf_per_scale(Number<2>{}),
                                           c_thread_buf_per_scale(Number<3>{}),
                                           c_thread_buf_per_scale(Number<4>{}),
                                           c_thread_buf_per_scale(Number<5>{}),
                                           c_thread_buf_per_scale(Number<6>{}),
                                           c_thread_buf_per_scale(Number<7>{}),
                                           c_thread_buf_per_scale(Number<8>{}),
                                           c_thread_buf_per_scale(Number<9>{}),
                                           c_thread_buf_per_scale(Number<10>{}),
                                           c_thread_buf_per_scale(Number<11>{}),
                                           c_thread_buf_per_scale(Number<12>{}),
                                           c_thread_buf_per_scale(Number<13>{}),
                                           c_thread_buf_per_scale(Number<14>{}),
                                           c_thread_buf_per_scale(Number<15>{}));
                                }
#endif
                            }
#endif

#if 0
                            // for b_col < 64
                            if(!is_B_zero && !is_A_zero)
                            {
                                if constexpr(m0 == 0 && n0 == 0)
                                {
                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : b_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                        "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<0>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<1>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<2>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<3>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<4>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<5>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<6>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<7>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<8>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<9>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<10>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<11>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<12>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<13>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<14>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<15>{})));
#if 0
                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : c_thread_buf_per_scale = [%f, %f, %f, %f, %f, %f, "
                                        "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        c_thread_buf_per_scale(Number<0>{}),
                                        c_thread_buf_per_scale(Number<1>{}),
                                        c_thread_buf_per_scale(Number<2>{}),
                                        c_thread_buf_per_scale(Number<3>{}),
                                        c_thread_buf_per_scale(Number<4>{}),
                                        c_thread_buf_per_scale(Number<5>{}),
                                        c_thread_buf_per_scale(Number<6>{}),
                                        c_thread_buf_per_scale(Number<7>{}),
                                        c_thread_buf_per_scale(Number<8>{}),
                                        c_thread_buf_per_scale(Number<9>{}),
                                        c_thread_buf_per_scale(Number<10>{}),
                                        c_thread_buf_per_scale(Number<11>{}),
                                        c_thread_buf_per_scale(Number<12>{}),
                                        c_thread_buf_per_scale(Number<13>{}),
                                        c_thread_buf_per_scale(Number<14>{}),
                                        c_thread_buf_per_scale(Number<15>{}));

                                    printf("blockId = %u; threadId = %u; i = %d; m0 = %d : "
                                           "b_scale_thread_buf[%d,%d] = %f\n",
                                           blockIdx.x,
                                           threadIdx.x,
                                           i,
                                           static_cast<int>(m0),
                                           static_cast<int>(n0),
                                           static_cast<int>(k0),
                                           type_convert<float>(
                                               b_scale_thread_buf[Number<b_scale_offset>{}]));
#endif
                                }
                            }
#endif

                            static_for<0, xdlops_gemm.mfma_instr.num_groups_per_blk, 1>{}(
                                [&](auto g) {
                                    static_for<0, xdlops_gemm.mfma_instr.group_size, 1>{}(
                                        [&](auto r) {
                                            constexpr index_t a_scale_offset =
                                                a_scale_thread_desc.CalculateOffset(
                                                    make_tuple(m0, k0, g, r));

                                            constexpr auto reg_offset =
                                                g * xdlops_gemm.mfma_instr.group_size + r;

                                            constexpr index_t c_offset =
                                                c_thread_desc_.CalculateOffset(
                                                    make_tuple(m0, n0, reg_offset));

                                            c_thread_buf(Number<c_offset>{}) +=
                                                c_thread_buf_per_scale[Number<reg_offset>{}] *
                                                type_convert<AccDataType>(
                                                    b_scale_thread_buf[Number<b_scale_offset>{}]) *
                                                type_convert<AccDataType>(
                                                    a_scale_thread_buf[Number<a_scale_offset>{}]);
                                        });
                                });
                        });
                    });
                });

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, xdlops_gemm.mfma_instr.num_groups_per_blk, 1>{}([&](auto g) {
                            auto a_scale_thread_buf_group =
                                make_static_buffer<AddressSpaceEnum::Vgpr, AScaleDataType>(
                                    a_scale_thread_desc_group.GetElementSpaceSize());

                            a_scale_thread_copy.Run(a_scale_grid_desc,
                                                    a_scale_grid_buf,
                                                    a_scale_thread_desc_group,
                                                    make_tuple(I0, I0),
                                                    a_scale_thread_buf_group);
#if 0
                            if(blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 32 ||
                                                   threadIdx.x == 64 || threadIdx.x == 96))
                            {
                                printf("blockId = %u; threadId = %u; {m0,k0,g} = {%d,%d,%d} : "
                                       "a_scale_thread_buf_group = "
                                       "[%f, %f, "
                                       "%f, %f]\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       static_cast<int>(m0),
                                       static_cast<int>(k0),
                                       static_cast<int>(g),
                                       type_convert<float>(a_scale_thread_buf_group(Number<0>{})),
                                       type_convert<float>(a_scale_thread_buf_group(Number<1>{})),
                                       type_convert<float>(a_scale_thread_buf_group(Number<2>{})),
                                       type_convert<float>(a_scale_thread_buf_group(Number<3>{})));
                            }
#endif
                            static_for<0, xdlops_gemm.mfma_instr.group_size, 1>{}([&](auto r) {
                                constexpr index_t a_scale_offset =
                                    a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, g, r));
                                a_scale_thread_buf(Number<a_scale_offset>{}) =
                                    a_scale_thread_buf_group[Number<r>{}];
                            });
                            // go to the next group
                            a_scale_thread_copy.MoveSrcSliceWindow(
                                a_scale_grid_desc,
                                make_multi_index(2 * xdlops_gemm.mfma_instr.group_size, 0));
                        }); // g

                        // restore row id and advance to the next scale
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc,
                            make_multi_index(-2 * xdlops_gemm.mfma_instr.group_size *
                                                 xdlops_gemm.mfma_instr.num_groups_per_blk,
                                             1));
                    }); // k0

                    // restore column id and advance to the next set of rows
                    a_scale_thread_copy.MoveSrcSliceWindow(
                        a_scale_grid_desc,
                        make_multi_index(MWaves * MPerXDL, -ScalesPerKBlockSize));
                }); // m0

                // restore row id and advance to the next set of scales
                a_scale_thread_copy.MoveSrcSliceWindow(
                    a_scale_grid_desc, make_multi_index(-MPerBlock, ScalesPerKBlockSize));

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_scale_thread_copy.Run(b_scale_grid_desc,
                                            b_scale_grid_buf,
                                            b_scale_thread_desc,
                                            make_tuple(n0, I0),
                                            b_scale_thread_buf);
                    b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                           make_multi_index(NWaves * NPerXDL, 0));
                });
                // NWaves * NPerXDL * NRepeat == NPerBlock
                b_scale_thread_copy.MoveSrcSliceWindow(
                    b_scale_grid_desc, make_multi_index(-NPerBlock, ScalesPerKBlockSize));

                block_sync_lds();
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                i += 1;

            } while(i < (num_loop - 1));
        }

        // tail
        if constexpr(TailNum == TailNumber::Full)
        {
            block_sync_lds();

#if 0
            if(blockIdx.x == 0 &&
               (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 10 || threadIdx.x == 42))
            {
                printf("blockId = %u; threadId = %u; i = %d : b_scale_thread_buf = [%f, %f, "
                       "%f, %f]\n",
                       blockIdx.x,
                       threadIdx.x,
                       -1,
                       type_convert<float>(b_scale_thread_buf(Number<0>{})),
                       type_convert<float>(b_scale_thread_buf(Number<1>{})),
                       type_convert<float>(b_scale_thread_buf(Number<2>{})),
                       type_convert<float>(b_scale_thread_buf(Number<3>{})));
            }
#endif

            static_for<0, KRepeat, 1>{}([&](auto k) {
                constexpr auto a_k_step = k * AMmaKStride * KPack / xdlops_gemm.K1PerXdlops;
                constexpr auto b_k_step = k * BMmaKStride * KPack / xdlops_gemm.K1PerXdlops;

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<a_k_step>{}),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_buf);
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<b_k_step>{}),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, I0),
                                       b_thread_buf);
                });
            });

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        c_thread_buf_per_scale.Clear();
                        vector_type<ComputeDataType, KPack> a_thread_vec;
                        vector_type<ComputeDataType, KPack> b_thread_vec;

                        bool is_B_zero = true;
                        bool is_A_zero = true;
                        ignore         = is_B_zero;
                        ignore         = is_A_zero;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
#if 1
                            if(b_thread_vec.template AsType<ComputeDataType>()(ik) ==
                               ComputeDataType{0})
                            {
                            }
                            else
                            {
                                is_B_zero = false;
                            }
                            if(a_thread_vec.template AsType<ComputeDataType>()(ik) ==
                               ComputeDataType{0})
                            {
                            }
                            else
                            {
                                is_A_zero = false;
                            }
#endif
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;

                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf_per_scale.GetVectorTypeReference(I0));

                        bool is_C_zero = true;
                        ignore         = is_C_zero;

                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto m) {
                            if(c_thread_buf_per_scale[Number<m>{}] == 0.0f) {}
                            else
                            {
                                is_C_zero = false;
                            }
                        });

                        // one scale per k0
                        constexpr index_t b_scale_offset =
                            b_scale_thread_desc.CalculateOffset(make_tuple(n0, k0));

// print out a_thread_vec
#if 0
                        if(!is_A_zero && blockIdx.x == 0 && threadIdx.x == 138)
                        {
                            // First MWaves * MPerXDL rows and NWaves * NPerXDL columns
                            if constexpr(m0 == 0 && n0 == 0)
                            {
                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                       "= %d : a_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                       "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<0>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<1>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<2>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<3>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<4>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<5>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<6>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<7>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<8>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<9>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<10>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<11>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<12>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<13>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<14>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<15>{})));

// print out b_thread_vec
#if 0
                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : b_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                        "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        -1,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<0>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<1>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<2>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<3>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<4>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<5>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<6>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<7>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<8>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<9>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<10>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<11>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<12>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<13>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<14>{})),
                                        type_convert<float>(
                                            b_thread_vec.template AsType<ComputeDataType>()(
                                                Number<15>{})));
                                    printf("blockId = %u; threadId = %u; i = %d; m0 = %d : "
                                           "b_scale_thread_buf[%d,%d] = %f\n",
                                           blockIdx.x,
                                           threadIdx.x,
                                           i,
                                           static_cast<int>(m0),
                                           static_cast<int>(n0),
                                           static_cast<int>(k0),
                                           type_convert<float>(
                                               b_scale_thread_buf[Number<b_scale_offset>{}]));
#endif
                            }
                        }
                        if(!is_C_zero && threadIdx.x == 128)
                        {
                            // First MWaves * MPerXDL rows and NWaves * NPerXDL columns
                            if constexpr(m0 == 0 && n0 == 0)
                            {
// print out c_thread_buf_per_scale
#if 1
                                constexpr index_t a_scale_offset =
                                    a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, I0, I0));

                                printf(
                                    "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                    "= %d :\n\tc_thread_buf_per_scale = [%f, %f, %f, %f, %f, %f, "
                                    "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n\t    "
                                    "a_scale_thread_buf "
                                    "= [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                                    "%f, %f]\n",
                                    blockIdx.x,
                                    threadIdx.x,
                                    -1,
                                    static_cast<int>(m0),
                                    static_cast<int>(n0),
                                    static_cast<int>(k0),
                                    c_thread_buf_per_scale(Number<0>{}),
                                    c_thread_buf_per_scale(Number<1>{}),
                                    c_thread_buf_per_scale(Number<2>{}),
                                    c_thread_buf_per_scale(Number<3>{}),
                                    c_thread_buf_per_scale(Number<4>{}),
                                    c_thread_buf_per_scale(Number<5>{}),
                                    c_thread_buf_per_scale(Number<6>{}),
                                    c_thread_buf_per_scale(Number<7>{}),
                                    c_thread_buf_per_scale(Number<8>{}),
                                    c_thread_buf_per_scale(Number<9>{}),
                                    c_thread_buf_per_scale(Number<10>{}),
                                    c_thread_buf_per_scale(Number<11>{}),
                                    c_thread_buf_per_scale(Number<12>{}),
                                    c_thread_buf_per_scale(Number<13>{}),
                                    c_thread_buf_per_scale(Number<14>{}),
                                    c_thread_buf_per_scale(Number<15>{}),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 0>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 1>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 2>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 3>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 4>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 5>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 6>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 7>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 8>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 9>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 10>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 11>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 12>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 13>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 14>{}]),
                                    type_convert<float>(
                                        a_scale_thread_buf[Number<a_scale_offset + 15>{}]));
#endif
                            }
                        }
#endif

#if 0
                        // for b_col == 74
                        if constexpr(m0 == 0 && n0 == 1)
                        {
                            if(blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 32))
                            {
                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                       "= %d : a_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                       "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<0>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<1>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<2>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<3>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<4>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<5>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<6>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<7>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<8>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<9>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<10>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<11>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<12>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<13>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<14>{})),
                                       type_convert<float>(
                                           a_thread_vec.template AsType<ComputeDataType>()(
                                               Number<15>{})));
                            }

                            if(blockIdx.x == 0 && (threadIdx.x == 10 || threadIdx.x == 42))
                            {
                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                       "= %d : b_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                       "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<0>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<1>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<2>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<3>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<4>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<5>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<6>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<7>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<8>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<9>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<10>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<11>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<12>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<13>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<14>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<15>{})));

                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d : "
                                       "b_scale_thread_buf[%d,%d] = %f\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       type_convert<float>(
                                           b_scale_thread_buf[Number<b_scale_offset>{}]));
                            }
#if 1
                            if(!is_C_zero)
                            {
                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = "
                                       "%d; k0 "
                                       "= %d : c_thread_buf_per_scale = [%f, %f, %f, %f, "
                                       "%f, %f, "
                                       "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       c_thread_buf_per_scale(Number<0>{}),
                                       c_thread_buf_per_scale(Number<1>{}),
                                       c_thread_buf_per_scale(Number<2>{}),
                                       c_thread_buf_per_scale(Number<3>{}),
                                       c_thread_buf_per_scale(Number<4>{}),
                                       c_thread_buf_per_scale(Number<5>{}),
                                       c_thread_buf_per_scale(Number<6>{}),
                                       c_thread_buf_per_scale(Number<7>{}),
                                       c_thread_buf_per_scale(Number<8>{}),
                                       c_thread_buf_per_scale(Number<9>{}),
                                       c_thread_buf_per_scale(Number<10>{}),
                                       c_thread_buf_per_scale(Number<11>{}),
                                       c_thread_buf_per_scale(Number<12>{}),
                                       c_thread_buf_per_scale(Number<13>{}),
                                       c_thread_buf_per_scale(Number<14>{}),
                                       c_thread_buf_per_scale(Number<15>{}));
                            }
#endif
                        }
#endif

#if 0
                        if(!is_B_zero && !is_A_zero)
                        {
                            if constexpr(m0 == 0)
                            {
                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                       "= %d : b_thread_vec = [%f, %f, %f, %f, %f, %f, %f, %f, "
                                       "%f, %f, %f, %f, %f, %f, %f, %f]\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<0>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<1>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<2>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<3>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<4>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<5>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<6>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<7>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<8>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<9>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<10>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<11>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<12>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<13>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<14>{})),
                                       type_convert<float>(
                                           b_thread_vec.template AsType<ComputeDataType>()(
                                               Number<15>{})));
#if 1
                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                       "= %d : c_thread_buf_per_scale = [%f, %f, %f, %f, %f, %f, "
                                       "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       c_thread_buf_per_scale(Number<0>{}),
                                       c_thread_buf_per_scale(Number<1>{}),
                                       c_thread_buf_per_scale(Number<2>{}),
                                       c_thread_buf_per_scale(Number<3>{}),
                                       c_thread_buf_per_scale(Number<4>{}),
                                       c_thread_buf_per_scale(Number<5>{}),
                                       c_thread_buf_per_scale(Number<6>{}),
                                       c_thread_buf_per_scale(Number<7>{}),
                                       c_thread_buf_per_scale(Number<8>{}),
                                       c_thread_buf_per_scale(Number<9>{}),
                                       c_thread_buf_per_scale(Number<10>{}),
                                       c_thread_buf_per_scale(Number<11>{}),
                                       c_thread_buf_per_scale(Number<12>{}),
                                       c_thread_buf_per_scale(Number<13>{}),
                                       c_thread_buf_per_scale(Number<14>{}),
                                       c_thread_buf_per_scale(Number<15>{}));

                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d : "
                                       "b_scale_thread_buf[%d,%d] = %f\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       type_convert<float>(
                                           b_scale_thread_buf[Number<b_scale_offset>{}]));
#endif
                            }
                        }
#endif

                        static_for<0, xdlops_gemm.mfma_instr.num_groups_per_blk, 1>{}([&](auto g) {
                            static_for<0, xdlops_gemm.mfma_instr.group_size, 1>{}([&](auto r) {
                                constexpr index_t a_scale_offset =
                                    a_scale_thread_desc.CalculateOffset(make_tuple(m0, k0, g, r));

                                constexpr auto reg_offset =
                                    g * xdlops_gemm.mfma_instr.group_size + r;

                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, reg_offset));

                                c_thread_buf(Number<c_offset>{}) +=
                                    c_thread_buf_per_scale[Number<reg_offset>{}] *
                                    type_convert<AccDataType>(
                                        b_scale_thread_buf[Number<b_scale_offset>{}]) *
                                    type_convert<AccDataType>(
                                        a_scale_thread_buf[Number<a_scale_offset>{}]);
                            });
                        });
                    });
                });
            });
        }
    }

    // TODO: make this field protected when a_scale_thread_copy_ is moved here
    static constexpr auto a_scale_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{},
                   Number<KRepeat>{},
                   Number<xdlops_gemm.mfma_instr.num_groups_per_blk>{},
                   Number<xdlops_gemm.mfma_instr.group_size>{}));

    // Is used to copy data from a_scale_grid to a_scale_thread
    static constexpr auto a_scale_thread_desc_group = make_naive_tensor_descriptor_packed(
        make_tuple(Number<xdlops_gemm.mfma_instr.group_size>{}, Number<1>{}));

    // TODO: make this field protected when b_scale_thread_copy_ is moved here
    static constexpr auto b_scale_thread_desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<NRepeat>{}, Number<KRepeat>{}));

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
};

} // namespace ck
