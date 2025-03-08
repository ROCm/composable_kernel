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
          index_t BlockSize,
          typename ADataType,
          typename BDataType,
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
          index_t MRepeat,
          index_t NRepeat,
          index_t KPacks>
struct BlockwiseGemmXdlops_pipeline_v1_mx
{
};

template <index_t BlockSize,
          typename ADataType,
          typename BDataType,
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
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack
          // ,bool TransposeC //disable transposec right now...
          >
struct BlockwiseGemmXdlops_pipeline_v1_mx<BlockGemmPipelineScheduler::Intrawave,
                                          BlockSize,
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
    : BlockwiseGemmXdlops_pipeline_base<BlockSize,
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
    using Base = BlockwiseGemmXdlops_pipeline_base<BlockSize,
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
              // BScale Thread Copy
              typename BScaleGridBuffer,
              typename BScaleGridDesc,
              typename BScaleThreadDesc,
              typename BScaleThreadTransfer,
              typename BScaleThreadTransferStep>
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
        // BScaleThreadCopy
        const BScaleGridDesc& b_scale_grid_desc,
        const BScaleThreadDesc& b_scale_thread_desc,
        BScaleThreadTransfer& b_scale_thread_copy,
        const BScaleGridBuffer& b_scale_grid_buf,
        const BScaleThreadTransferStep& b_scale_thread_copy_step,
        // num_loop
        index_t num_loop,
        index_t num_loop_per_scale) const
    {
        // assume kperblock = scaleblockk
        ignore            = num_loop_per_scale;
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            b_thread_desc_.GetElementSpaceSize());

        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, _Float16>(
            b_scale_thread_desc.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        static_for<0, NRepeat, 1>{}([&](auto n0) {
            b_scale_thread_copy.Run(b_scale_grid_desc,
                                    b_scale_grid_buf,
                                    b_scale_thread_desc,
                                    make_tuple(n0, I0),
                                    b_scale_thread_buf);

            b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                   b_scale_thread_copy_step.At(Number<0>{}));
        });
        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                               b_scale_thread_copy_step.At(Number<1>{}));

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

            // clang-format off
            // print_type_name(
            //     "c_thread_buf_per_scale",
            //     c_thread_buf_per_scale); // ck::StaticBufferTupleOfVector<ck::AddressSpaceEnum::Vgpr,
                                            // float, 4, 16, true>
            // clang-format on

            print_type_name("xdlops_gemm", xdlops_gemm); // ck::XdlopsGemm<_BitInt(8), 32, 32, 16>
            using mfma_input_type =
                typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;
            print_type_name("mfma_input_type",
                            mfma_input_type{}); //_BitInt(8) __attribute__((ext_vector_type(8)))

            print_type_name("mfma = ", xdlops_gemm.mfma); // ck::MfmaSelector<_BitInt(8), 32, 32>
            print_type_name(
                "mfma_instr = ",
                xdlops_gemm.mfma_instr); // ck::mfma_type<MfmaInstr::mfma_f32_32x32x16f8f8>

            // print_type_name(
            //     "a_thread_buf",
            //     a_thread_buf); // ck::StaticBuffer<ck::AddressSpaceEnum::Vgpr, _BitInt(8), 64,
            //     true>
            // print_type_name("a_block_buf",
            //                 a_block_buf); // ck::DynamicBuffer<ck::AddressSpaceEnum::Lds,
            // _BitInt(8), ck::integral_constant<long, 8192>, true,
            // ck::AmdBufferCoherenceEnum::DefaultCoherence>
            // print_type_name(
            //     "b_scale_thread_copy",
            //     b_scale_thread_copy); // ck::ThreadwiseTensorSliceTransfer_v2<_Float16, _Float16,
            // const
            // ck::TensorDescriptor<ck::Tuple<ck::Embed<ck::Tuple<int,
            // int>, ck::Tuple<int, int>>>, ck::Tuple<ck::Sequence<0>>,
            // ck::Tuple<ck::Sequence<1, 2>>, ck::Sequence<1, 2>, long> &,
            // const
            // ck::TensorDescriptor<ck::Tuple<ck::UnMerge<ck::Tuple<ck::integral_constant<int,
            // 2>, ck::integral_constant<int, 1>>, false>>,
            // ck::Tuple<ck::Sequence<0>>, ck::Tuple<ck::Sequence<1, 2>>,
            // ck::Sequence<1, 2>, ck::integral_constant<long, 2>>,
            // ck::Sequence<1, 1>, ck::Sequence<0, 1>, 1, 1, 1, false>

            // print_type_name("b_scale_thread_copy_step",
            //                 b_scale_thread_copy_step); // ck::Tuple<ck::Tuple<int, int>,
            // ck::Tuple<int, int>, ck::Tuple<int, int>>

            // print_type_name("b_scale_thread_buf",
            //                 b_scale_thread_buf); // ck::StaticBuffer<ck::AddressSpaceEnum::Vgpr,
            // _Float16, 2, true>

            print_type_name("b_thread_copy_", b_thread_copy_);
            print_type_name("b_block_desc_n0_n1_n2_k", b_block_desc_n0_n1_n2_k);
            print_type_name("b_thread_desc_", b_thread_desc_);

            print_type_name("b_block_buf",
                            b_block_buf); // ck::DynamicBuffer<ck::AddressSpaceEnum::Lds,
                                          // ck::f8_ocp_t, ck::integral_constant<long, 8192>, true,
                                          // ck::AmdBufferCoherenceEnum::DefaultCoherence>

            printf("MRepeat = %d\n", MRepeat); // 2
            printf("NRepeat = %d\n", NRepeat); // 2
            printf("KRepeat = %d\n", KRepeat); // 2
            printf("KPack = %d\n", KPack);     // 16
            printf("xdlops_gemm.GetRegSizePerXdlops() = %d\n",
                   xdlops_gemm.GetRegSizePerXdlops());                               // 16
            printf("mfma_instr.k_per_blk = %d\n", xdlops_gemm.mfma_instr.k_per_blk); // 8
        }
#endif
#if 1
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
            index_t i = 0;
            do
            {
                // -------------------------------------------------------------------------------------------
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                block_sync_lds();
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
                        c_thread_buf_per_scale.Clear();
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
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

                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(I0));

#if 0
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

                                    printf(
                                        "blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                        "= %d : b_scale_thread_buf = [%f, %f]\n",
                                        blockIdx.x,
                                        threadIdx.x,
                                        i,
                                        static_cast<int>(m0),
                                        static_cast<int>(n0),
                                        static_cast<int>(k0),
                                        type_convert<float>(b_scale_thread_buf(Number<0>{})),
                                        type_convert<float>(b_scale_thread_buf(Number<1>{})));
#endif
                                }
                            }
#endif
                        });
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                            c_thread_buf(Number<c_offset>{}) +=
                                c_thread_buf_per_scale[Number<t>{}] *
                                type_convert<AccDataType>(b_scale_thread_buf[n0]);
                        });
                    });
                });

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_scale_thread_copy.Run(b_scale_grid_desc,
                                            b_scale_grid_buf,
                                            b_scale_thread_desc,
                                            make_tuple(n0, I0),
                                            b_scale_thread_buf);

                    b_scale_thread_copy.MoveSrcSliceWindow(
                        b_scale_grid_desc, b_scale_thread_copy_step.At(Number<0>{}));
                });

                b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                       b_scale_thread_copy_step.At(Number<1>{}));

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
                    c_thread_buf_per_scale.Clear();
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
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

#if 0
                        if(!is_B_zero && !is_A_zero)
                        {
                            if constexpr(n0 == 0 && m0 == 0)
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
#if 0
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

                                printf("blockId = %u; threadId = %u; i = %d; m0 = %d; n0 = %d; k0 "
                                       "= %d : b_scale_thread_buf = [%f, %f]\n",
                                       blockIdx.x,
                                       threadIdx.x,
                                       -1,
                                       static_cast<int>(m0),
                                       static_cast<int>(n0),
                                       static_cast<int>(k0),
                                       type_convert<float>(b_scale_thread_buf(Number<0>{})),
                                       type_convert<float>(b_scale_thread_buf(Number<1>{})));
#endif
                            }
                        }
#endif
                    });
                    static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                        c_thread_buf(Number<c_offset>{}) +=
                            c_thread_buf_per_scale[Number<t>{}] *
                            type_convert<AccDataType>(b_scale_thread_buf[n0]);
                    });
                });
            });
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
