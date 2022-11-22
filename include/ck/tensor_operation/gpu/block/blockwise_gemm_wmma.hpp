// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/wmma_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

enum struct LoopScheduler
{
    Default,
};

constexpr LoopScheduler make_default_loop_scheduler()
{
    return LoopScheduler::Default;
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerWMMA,
          index_t NPerWMMA,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
struct BlockwiseGemmWMMA_k0mk1_k0nk1_m0m1n0n1n2m2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t WaveSize = get_warp_size();

    static constexpr index_t MPerBlock = AK0MK1BlockDesc{}.GetLength(I1);
    static constexpr index_t NPerBlock = BK0NK1BlockDesc{}.GetLength(I1);
    static constexpr index_t KPerBlock = BK0NK1BlockDesc{}.GetLength(I0) * BK0NK1BlockDesc{}.GetLength(I2);

    static constexpr index_t A_K0 = AK0MK1BlockDesc{}.GetLength(I0);
    static constexpr index_t B_K0 = BK0NK1BlockDesc{}.GetLength(I0);
    static constexpr index_t A_K1 = AK0MK1BlockDesc{}.GetLength(I2);
    static constexpr index_t B_K1 = BK0NK1BlockDesc{}.GetLength(I2);

    static constexpr auto wmma_gemm = WMMAGemm<FloatAB, MPerWMMA, NPerWMMA, KPack>{};

    static constexpr index_t KPerThread = KPerBlock / wmma_gemm.K0PerWMMA;

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWMMA);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWMMA);

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              wmma_gemm.GetRegSizePerWMMA(),
                              true>
        c_thread_buf_;

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

        const auto WMMA_a_idx = wmma_gemm.CalculateAThreadOriginDataIndex();

        return make_tuple(0, waveId_m, WMMA_a_idx[I1], KPerThread * WMMA_a_idx[I0]);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto WMMA_b_idx = wmma_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(0, waveId_n, WMMA_b_idx[I1], KPerThread * WMMA_b_idx[I0]);
    }

    template <index_t m0, index_t n0, index_t WMMA_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<WMMA_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk(WMMA_i, blk_i);

        constexpr auto mrepeat_mwave_mperWMMA_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerWMMA))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperWMMA_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerWMMA))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperWMMA_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperWMMA_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    template <index_t m0, index_t n0, index_t WMMA_i, index_t blk_i>
    __device__ static auto
        CalculateCThreadOriginDataIndex8D(Number<m0>, Number<n0>, Number<WMMA_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk4D(WMMA_i, blk_i);

        return make_tuple(Number<m0>{},
                          Number<n0>{},
                          waveId_m,
                          waveId_n,
                          blk_idx[I0],
                          blk_idx[I1],
                          blk_idx[I2],
                          blk_idx[I3]);
    }

    __host__ __device__ BlockwiseGemmWMMA_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1()
    {
        static_assert(AK0MK1BlockDesc::IsKnownAtCompileTime() &&
                          BK0NK1BlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerWMMA * MRepeat) == 0 && NPerBlock % (NPerWMMA * NRepeat) == 0,
                      "wrong!");
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = wmma_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = wmma_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerWMMA>{},
                                                           Number<NPerWMMA>{}));

        return wmma_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerWMMA>{},
                                                           Number<NPerWMMA>{}));

        return wmma_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
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
            make_tuple(make_unmerge_transform(make_tuple(M / (MWaves * MPerWMMA), MWaves, MPerWMMA)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerWMMA), NWaves, NPerWMMA))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return wmma_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto MakeABlockDescriptor_K0_M0_M1_M2_K1()
    {
        return transform_tensor_descriptor(
            AK0MK1BlockDesc{},
            make_tuple(
                make_pass_through_transform(make_tuple(Number<A_K0>{}, Number<A_K1>{})),
                make_unmerge_transform(
                    make_tuple(Number<MRepeat>{}, Number<MWaves>{}, Number<MPerWMMA>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 4>{}, Sequence<1, 2, 3>{}));
    }

    __host__ __device__ static constexpr auto MakeBBlockDescriptor_K0_N0_N1_N2_K1()
    {
        return transform_tensor_descriptor(
            BK0NK1BlockDesc{},
            make_tuple(
                make_pass_through_transform(make_tuple(Number<B_K0>{}, Number<B_K1>{})),
                make_unmerge_transform(
                    make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerWMMA>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 4>{}, Sequence<1, 2, 3>{}));
    }

    static constexpr auto a_block_desc_k0_m0_m1_m2_k1 = MakeABlockDescriptor_K0_M0_M1_M2_K1();
    static constexpr auto b_block_desc_k0_n0_n1_n2_k1 = MakeBBlockDescriptor_K0_N0_N1_N2_K1();

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        constexpr auto RepeatDiff = MRepeat - NRepeat;
        constexpr auto WmmaK = wmma_gemm.k_per_wmma;

        static_for<0, KPerBlock / WmmaK, 1>{}([&](auto iWmmaK){
            // Cut to Repeat Retangle to Square, assume MRepeat > NRepeat
            static_for<0, RepeatDiff, 1>{}([&](auto iCut){
                static_for<0, NRepeat, 1>{}([&](auto iN){
                    vector_type<FloatAB, WmmaK> a_thread_vec;
                    vector_type<FloatAB, WmmaK> b_thread_vec;

                    static_for<0, WmmaK, 1>{}([&](auto iK) {
                        a_thread_vec.template AsType<FloatAB>()(iK) =
                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                make_tuple(iCut, 0, 0, iK))>{}];
                        b_thread_vec.template AsType<FloatAB>()(iK) =
                            b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                make_tuple(iN, 0, 0, iK))>{}];
                    });
                    using wmma_input_type = typename vector_type<FloatAB, WmmaK>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(iCut, iN, 0));
                    wmma_gemm.template Run(
                            a_thread_vec.template AsType<wmma_input_type>(),
                            b_thread_vec.template AsType<wmma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                });
                a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                   make_tuple(Number<iWmmaK>{}, iCut, I0, I0, I0),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(I0, I0, I0, I0),
                                   a_thread_buf);
            });
            // Run FIFO fashion loopover in Square
            static_for<0, NRepeat, 1>{}([&](auto WmmaInnerloop){
                static_for<WmmaInnerloop, NRepeat, 1>{}([&](auto iN){
                    vector_type<FloatAB, WmmaK> a_thread_vec;
                    vector_type<FloatAB, WmmaK> b_thread_vec;

                    static_for<0, WmmaK, 1>{}([&](auto iK) {
                        a_thread_vec.template AsType<FloatAB>()(iK) =
                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                make_tuple(WmmaInnerloop+RepeatDiff, 0, 0, iK))>{}];
                        b_thread_vec.template AsType<FloatAB>()(iK) =
                            b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                make_tuple(iN, 0, 0, iK))>{}];
                    });
                    using wmma_input_type = typename vector_type<FloatAB, WmmaK>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(WmmaInnerloop+RepeatDiff, iN, 0));
                    wmma_gemm.template Run(
                            a_thread_vec.template AsType<wmma_input_type>(),
                            b_thread_vec.template AsType<wmma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                });
                a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                   make_tuple(Number<iWmmaK>{}, WmmaInnerloop+RepeatDiff, I0, I0, I0),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(I0, I0, I0, I0),
                                   a_thread_buf);
                static_for<WmmaInnerloop+RepeatDiff, MRepeat, 1>{}([&](auto iM){
                    vector_type<FloatAB, WmmaK> a_thread_vec;
                    vector_type<FloatAB, WmmaK> b_thread_vec;

                    static_for<0, WmmaK, 1>{}([&](auto iK) {
                        a_thread_vec.template AsType<FloatAB>()(iK) =
                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                make_tuple(iM, 0, 0, iK))>{}];
                        b_thread_vec.template AsType<FloatAB>()(iK) =
                            b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                make_tuple(WmmaInnerloop, 0, 0, iK))>{}];
                    });
                    using wmma_input_type = typename vector_type<FloatAB, WmmaK>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(iM, WmmaInnerloop, 0));
                    wmma_gemm.template Run(
                            a_thread_vec.template AsType<wmma_input_type>(),
                            b_thread_vec.template AsType<wmma_input_type>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                });
                b_thread_copy_.Run(b_block_desc_k0_n0_n1_n2_k1,
                                   make_tuple(Number<iWmmaK>{}, WmmaInnerloop, I0, I0, I0),
                                   b_block_buf,
                                   b_thread_desc_,
                                   make_tuple(I0, I0, I0, I0),
                                   b_thread_buf);
            });
        });
    }

    protected:
    // A[M0, M1, M2, K0 = WmmaK]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<WmmaK>{}));

    // B[N0, N1, N2, K0 = WmmaK]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<WmmaK>{}));

    // C[M, N, NumRegWMMA]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, wmma_gemm.GetRegSizePerWMMA()));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_block_desc_k0_m0_m1_m2_k1),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, WmmaK>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_block_desc_k0_n0_n1_n2_k1),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, WmmaK>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_{CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{CalculateBThreadOriginDataIndex()};
};

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerWMMA,
          index_t NPerWMMA,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          LoopScheduler LoopSched>
constexpr auto BlockwiseGemmWMMA_k0mk1_k0nk1_m0m1n0n1n2m2_Selector()
{
    if constexpr(LoopSched == LoopScheduler::Default)
    {
        return BlockwiseGemmWMMA_k0mk1_k0nk1_m0m1n0n1n2m2<BlockSize,
                                                             FloatAB,
                                                             FloatAcc,
                                                             AK0MK1BlockDesc,
                                                             BK0NK1BlockDesc,
                                                             MPerWMMA,
                                                             NPerWMMA,
                                                             MRepeat,
                                                             NRepeat,
                                                             KPack>{};
    }
};

} // namespace ck
