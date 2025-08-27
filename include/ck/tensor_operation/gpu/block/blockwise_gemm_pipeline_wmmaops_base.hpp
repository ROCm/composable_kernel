// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/wmma_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

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
          index_t KPack,
          bool TransposeC = false>
struct BlockwiseGemmWmmaops_pipeline_base
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I5 = Number<5>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t WaveSize = 32;

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

#if defined(__gfx12__)
    static constexpr index_t A_KRow = 2;
    static constexpr index_t B_KRow = 2;
#else
    static constexpr index_t A_KRow = 1;
    static constexpr index_t B_KRow = 1;
#endif

    static constexpr index_t A_K1 = AWmmaTileDesc{}.GetLength(I5);
    static constexpr index_t B_K1 = BWmmaTileDesc{}.GetLength(I5);

    static_assert(KPack % (A_K1 * A_KRow) == 0, "wrong!");
    static_assert(KPack % (B_K1 * B_KRow) == 0, "wrong!");

    static constexpr auto wmma_gemm =
        WmmaGemm<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma, KPack, TransposeC>{};

    static constexpr index_t KRepeat = KPerBlock / KPack;

    static constexpr auto WmmaK = Number<wmma_gemm.wmma_instr.k_per_wmma>{};

    using HotLoopInstList =
        ck::BlockwiseGemmWmmaops_pipeline_hotloop_inst<BlockSize,
                                                       MPerBlock,
                                                       NPerBlock,
                                                       KPerBlock,
                                                       ABlockTransferSrcScalarPerVector,
                                                       BBlockTransferSrcScalarPerVector,
                                                       A_K1,
                                                       B_K1,
                                                       A_K1,
                                                       B_K1,
                                                       MRepeat,
                                                       NRepeat,
                                                       MPerWmma,
                                                       NPerWmma,
                                                       wmma_gemm.wmma_instr.k_per_wmma>;

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              AccDataType,
                              MRepeat * NRepeat,
                              wmma_gemm.GetRegSizePerWmma(),
                              true>
        c_thread_buf_;

    struct Empty
    {
        __device__ Empty() {};
        template <index_t NBuffer>
        __device__ void GlobalLoad(bool cond)
        {
            ignore = NBuffer;
            ignore = cond;
        }
    };

    template <index_t ScaleSliceSizeN,
              index_t ScaleSliceSizeK,
              index_t NWaves,
              index_t ScaleBlockK,
              index_t NumberOfBuffers,
              typename GridDesc,
              typename ThreadCopy,
              typename GridBuffer,
              typename ThreadStaticBuffer,
              typename BScaleThreadDesc>
    struct BScale
    {
        __device__ BScale(GridDesc b_scale_grid_desc_,
                          ThreadCopy b_scale_thread_copy_,
                          GridBuffer b_scale_grid_buf_)
            : b_scale_thread_copy(b_scale_thread_copy_),
              b_scale_grid_desc(b_scale_grid_desc_),
              b_scale_grid_buf(b_scale_grid_buf_) {};

        static constexpr index_t num_scale_k_block = BScaleThreadDesc{}.GetLength(Number<1>{});
        static constexpr index_t num_scale_krepeat = KRepeat / num_scale_k_block;

        static constexpr auto b_scale_thread_desc = BScaleThreadDesc{};

        static constexpr auto b_scale_thread_copy_step =
            make_tuple(make_multi_index(NWaves * NPerWmma, 0),
                       make_multi_index(-NPerBlock, 0),
                       make_multi_index(-NPerBlock, (KPerBlock + ScaleBlockK - 1) / ScaleBlockK));

        template <index_t NBuffer>
        __device__ void GlobalLoad(bool cond)
        {
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                b_scale_thread_copy.Run(b_scale_grid_desc,
                                        b_scale_grid_buf,
                                        b_scale_thread_desc,
                                        make_tuple(n0, Number<0>{}),
                                        b_scale_thread_bufs(Number<NBuffer>{}));

                b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                       b_scale_thread_copy_step.At(Number<0>{}));
            });

            if(cond)
            {
                b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                       b_scale_thread_copy_step.At(Number<2>{}));
            }
            else
            {
                b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                       b_scale_thread_copy_step.At(Number<1>{}));
            }
        }

        ThreadCopy b_scale_thread_copy;
        GridDesc b_scale_grid_desc;
        GridBuffer b_scale_grid_buf;
        StaticallyIndexedArray<ThreadStaticBuffer, Number<NumberOfBuffers>{}> b_scale_thread_bufs;
    };

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];

        const auto wmma_a_idx = wmma_gemm.CalculateAThreadOriginDataIndex();

#if defined(__gfx12__)
        const auto wmma_krow = wmma_gemm.GetSubGroupId();
#else
        const auto wmma_krow = 0;
#endif

        //  |KRepeat   |MRepeat|MWave    |KRow  |MLane  |KPack
        return make_tuple(0, 0, waveId_m, wmma_krow, wmma_a_idx, 0);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto wmma_b_idx = wmma_gemm.CalculateBThreadOriginDataIndex();

#if defined(__gfx12__)
        const auto wmma_krow = wmma_gemm.GetSubGroupId();
#else
        const auto wmma_krow = 0;
#endif

        //  |KRepeat   |NRepeat|Nwave     |KRow  |NLane  |KPack
        return make_tuple(0, 0, waveId_n, wmma_krow, wmma_b_idx, 0);
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk();

        constexpr auto mrepeat_mwave_mperwmma_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerWmma))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperwmma_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerWmma))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperwmma_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperwmma_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    using Tuple6 = decltype(CalculateAThreadOriginDataIndex());

    /**
     * @brief Constructor for BlockwiseGemmWmmaops_pipeline_base.
     *
     * This constructor initializes the thread copy objects for matrices A and B.
     * It also performs several compile-time checks to ensure the correctness of the
     * matrix tile descriptors.
     *
     * @param a_origin The origin data index for matrix A.
     * @param b_origin The origin data index for matrix B.
     *
     * @note The constructor includes static assertions to ensure that:
     * - The matrix tile descriptors for A and B are known at compile-time.
     * - The number of threads in the thread block matches the product of MWaves, NWaves, and
     * WaveSize.
     * - The dimensions of the block are divisible by the product of the corresponding WMMA and
     * repeat dimensions.
     */
    __host__ __device__
    BlockwiseGemmWmmaops_pipeline_base(Tuple6 a_origin = CalculateAThreadOriginDataIndex(),
                                       Tuple6 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(AWmmaTileDesc::IsKnownAtCompileTime() &&
                          BWmmaTileDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize");

        static_assert(MPerBlock % (MPerWmma * MRepeat) == 0 &&
                          NPerBlock % (NPerWmma * NRepeat) == 0,
                      "wrong!");
    }

    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();

        constexpr auto MAccVgprs = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I2];
        constexpr auto AccStride = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I3];
        return make_naive_tensor_descriptor(
            //        |MRepeat           |MWave |MSubGroup |NRepeat           |NWave
            //        |NThreadPerSubGroup |MAccVgprs
            make_tuple(Number<MRepeat>{}, I1, I1, Number<NRepeat>{}, I1, I1, MAccVgprs),
            make_tuple(Number<NRepeat>{} * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       AccStride));
    }

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<MPerWmma>{},
                                                           Number<NRepeat>{},
                                                           Number<NWaves>{},
                                                           Number<NPerWmma>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
                c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma);
    }

    // Describe how data allocated in thread copy src buffer
    // M0_M1_M2 = MRepeat_MWave_MPerWmma, N0_N1_N2 = NRepeat_NWave_NPerWmma
    static constexpr AWmmaTileDesc a_block_desc_k0_m0_m1_m2_k1;
    static constexpr BWmmaTileDesc b_block_desc_k0_n0_n1_n2_k1;

    protected:
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<KPack / A_K1 / A_KRow>{},
                                                Number<MRepeat>{},
                                                Number<KRepeat>{},
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
                                                Number<KRepeat>{},
                                                I1,
                                                I1,
                                                Number<B_K1>{}),
                                     make_tuple(Number<B_K1>{},
                                                Number<KPack / B_KRow>{},
                                                Number<KPack / B_KRow * NRepeat>{},
                                                I0,
                                                I0,
                                                I1));

    // C[M, N, NumRegWmma]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, wmma_gemm.GetRegSizePerWmma()));

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

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
