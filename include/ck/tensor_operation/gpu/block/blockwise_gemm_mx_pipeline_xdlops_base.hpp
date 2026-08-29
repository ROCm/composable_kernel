// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/utility/thread_buf_to_vec_loader.hpp"

namespace ck {

template <index_t BlockSize,
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
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          bool TransposeC = false>
struct BlockwiseGemmXdlops_mx_pipeline_base
{
    using ComputeTypeA = ADataType;
    using ComputeTypeB = BDataType;
    using AccType      = float; // for now only support V_MFMA_SCALE_F32

    static constexpr index_t APackedSize = packed_size_v<ComputeTypeA>;
    static constexpr index_t BPackedSize = packed_size_v<ComputeTypeB>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t MWaves   = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t NWaves   = NPerBlock / (NRepeat * NPerXDL);
    static constexpr index_t WaveSize = BlockSize / MWaves / NWaves;

    static constexpr index_t A_K0 = ATileDesc{}.GetLength(I0);
    static constexpr index_t B_K0 = BTileDesc{}.GetLength(I0);
    static constexpr index_t A_K1 = ATileDesc{}.GetLength(I2);
    // static constexpr index_t B_K1 = BTileDesc{}.GetLength(I2);
    static constexpr index_t B_K1 =
        BTileDesc{}.GetLength(Number < BTileDesc{}.GetNumOfDimension() == 4 ? 3 : 2 > {});

    static constexpr auto xdlops_gemm = XdlopsGemm<ComputeTypeA,
                                                   MPerXDL,
                                                   NPerXDL,
                                                   KPack * APackedSize,
                                                   ComputeTypeB,
                                                   TransposeC,
                                                   true>{};

    static constexpr index_t AKPack      = KPack;
    static constexpr index_t BKPack      = KPack * APackedSize / BPackedSize;
    static constexpr index_t AMmaKStride = AKPack;
    static constexpr index_t BMmaKStride = BKPack;

    // store rows/cols into thread registers in chunks of 16 for FP8
    // e.g. [k0,...,k15,k64,...,k79] or [k0,...,k15,k32,...,k47]
    // or in chunks of 32 / APackedSize for FP6/FP4
    static constexpr index_t KThreadChunk = (APackedSize == 1) ? 16 : 32 / APackedSize;

    // FP8/FP4 mixed precision is valid
    static_assert(APackedSize == BPackedSize || KThreadChunk == 16,
                  "APackedSize must be equal to BPackedSize for now");

    static constexpr index_t KPerThread    = KPerBlock / xdlops_gemm.K0PerXdlops;
    static constexpr index_t KRepeat       = KPerThread / KPack;
    static constexpr index_t KPerInnerLoop = KPack;

    // Tuning parameters for better 8-bit access pattern
    // gfx125 scale32 wmma instructions can access two sets of scales per wave per matrix
    // gfx950 mfma instructions can access four sets of scales per wave per matrix
    // Ultimately, we aim to support the following configurations:
    // gfx950: MXdlPack=2, NXdlPack=2, KXdlPack=2
    // gfx1250, scale32: MXdlPack=2, NXdlPack=2, KXdlPack=1
    // gfx1250, scale16: MXdlPack=1, NXdlPack=1, KXdlPack=1
    static constexpr index_t MXdlPack = (ScaleBlockSize == 32) ? 2 : 1;
    static constexpr index_t NXdlPack = (ScaleBlockSize == 32) ? 2 : 1;
    static constexpr index_t KXdlPack = (xdlops_gemm.K1PerXdlops == 64) ? 1 : 2;

    using mx_scale_t                        = e8m0_bexp_t;
    static constexpr auto scale_pack_size_a = sizeof(AScaleDataType) / sizeof(mx_scale_t);
    static constexpr auto scale_pack_size_b = sizeof(BScaleDataType) / sizeof(mx_scale_t);

    static_assert(scale_pack_size_a == 1 || scale_pack_size_a == 2 || scale_pack_size_a == 4,
                  "A scale must be packed into 1, 2 or 4 bytes!");
    static_assert(scale_pack_size_b == 1 || scale_pack_size_b == 2 || scale_pack_size_b == 4,
                  "B scale must be packed into 1, 2 or 4 bytes!");

    // MX WMMA/MFMA builtins pack scales into int32_t registers
    static constexpr auto a_scale_thread_vec_size = sizeof(int32_t) / scale_pack_size_a;
    static constexpr auto b_scale_thread_vec_size = sizeof(int32_t) / scale_pack_size_b;

    // Detect FP4/FP6 separately for A and B based on packed_size_v:
    // FP4: packed_size_v = 2 (f4x2_pk_t)
    // FP6: packed_size_v = 16 or 32 (f6x16_pk_t, f6x32_pk_t, bf6x16_pk_t, bf6x32_pk_t)
    // FP8: packed_size_v = 1 or other small values
    // Note: 2x MFMA speedup requires BOTH operands to be the right type
    static constexpr bool IsF4_A = (packed_size_v<ComputeTypeA> == 2);
    static constexpr bool IsF4_B = (packed_size_v<ComputeTypeB> == 2);
    static constexpr bool IsF6_A =
        (packed_size_v<ComputeTypeA> == 16 || packed_size_v<ComputeTypeA> == 32);
    static constexpr bool IsF6_B =
        (packed_size_v<ComputeTypeB> == 16 || packed_size_v<ComputeTypeB> == 32);

    using HotLoopInstList = ck::BlockwiseGemmXdlops_pipeline_hotloop_inst< //
        BlockSize,
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
        MPerXDL,
        NPerXDL,
        xdlops_gemm.KPerXdlops,
        IsF4_A,
        IsF4_B,
        IsF6_A,
        IsF6_B>;
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(KPerThread % KPack == 0,
                  "Wrong KPack setting; try increasing KPerThread or decreasing KPack");
#endif

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              AccType,
                              MRepeat * NRepeat,
                              xdlops_gemm.GetRegSizePerXdlops(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() [[clang::lifetimebound]]
    {
        return c_thread_buf_;
    }

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

        const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

        return make_tuple(0, waveId_m, 0, xdlops_a_idx[I1], KThreadChunk * xdlops_a_idx[I0]);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(0, waveId_n, 0, xdlops_b_idx[I1], KThreadChunk * xdlops_b_idx[I0]);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
    CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        constexpr auto mrepeat_mwave_mperxdl_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_unmerge_transform(make_tuple(MRepeat / MXdlPack, MWaves, MXdlPack, MPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3>{}));

        constexpr auto nrepeat_nwave_nperxdl_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_unmerge_transform(make_tuple(NRepeat / NXdlPack, NWaves, NXdlPack, NPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3>{}));

        // We pack 2 mfma in M/N direction, so we need to divide by 2
        const index_t c_thread_m = mrepeat_mwave_mperxdl_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0 / MXdlPack, waveId_m, m0 % MXdlPack, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperxdl_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0 / NXdlPack, waveId_n, n0 % NXdlPack, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    using Tuple5 = decltype(CalculateAThreadOriginDataIndex());

    /**
     * @brief Constructor for BlockwiseGemmXdlops_mx_pipeline_base.
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
     * - The dimensions of the block are divisible by the product of the corresponding XDL and
     * repeat dimensions.
     */
    __host__ __device__
    BlockwiseGemmXdlops_mx_pipeline_base(Tuple5 a_origin = CalculateAThreadOriginDataIndex(),
                                         Tuple5 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(AMmaTileDesc::IsKnownAtCompileTime() && BMmaTileDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");
        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");
        static_assert(MRepeat % MXdlPack == 0);
#endif
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, N, M0, M1, M2));
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl, packed mfma
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat / MXdlPack>{},
                                                              Number<NRepeat / NXdlPack>{},
                                                              I1,
                                                              I1,
                                                              Number<MXdlPack>{},
                                                              Number<NXdlPack>{},
                                                              M0,
                                                              M1,
                                                              M2,
                                                              N));
    }
    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl' packed mfma
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat / MXdlPack>{},
                                                              Number<NRepeat / NXdlPack>{},
                                                              I1,
                                                              I1,
                                                              Number<MXdlPack>{},
                                                              Number<NXdlPack>{},
                                                              N,
                                                              M0,
                                                              M1,
                                                              M2));
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl_packed mfma
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat / MXdlPack>{},
                                                           Number<NRepeat / NXdlPack>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MXdlPack>{},
                                                           Number<NXdlPack>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3(
            c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'_packed mfma
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat / MXdlPack>{},
                                                           Number<NRepeat / NXdlPack>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MXdlPack>{},
                                                           Number<NXdlPack>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(
            c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_block_desc_g_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_G_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        const auto c_grid_desc_g_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_g_m_n,
            make_tuple(make_pass_through_transform(G),
                       make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3, 5>{}, Sequence<2, 4, 6>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_grid_desc_g_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto GetCThreadDesc() { return c_thread_desc_; }

    static constexpr AMmaTileDesc a_block_desc_m0_m1_m2_m3_k;
    static constexpr BMmaTileDesc b_block_desc_n0_n1_n2_n3_k;

    protected:
    // M1, N1 as double buffer index
    // Read buffer + Compute buffer
    // A[M0, M1, M2, KPack]
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<MRepeat / MXdlPack>{}, I1, Number<MXdlPack>{}, Number<KRepeat>{}, Number<AKPack>{}));

    // B[N0, N1, N2, KPack]
    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<NRepeat / NXdlPack>{}, I1, Number<NXdlPack>{}, Number<KRepeat>{}, Number<BKPack>{}));

    // C[M, N, NumRegXdlops]
    static constexpr auto c_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat / MXdlPack>{},
                                                       Number<NRepeat / NXdlPack>{},
                                                       Number<MXdlPack>{},
                                                       Number<NXdlPack>{},
                                                       xdlops_gemm.GetRegSizePerXdlops()));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<ADataType,
                                                         ComputeTypeA,
                                                         decltype(a_block_desc_m0_m1_m2_m3_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, 1, KThreadChunk>,
                                                         Sequence<0, 1, 2, 3, 4>,
                                                         4,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<BDataType,
                                                         ComputeTypeB,
                                                         decltype(b_block_desc_n0_n1_n2_n3_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, 1, KThreadChunk>,
                                                         Sequence<0, 1, 2, 3, 4>,
                                                         4,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
