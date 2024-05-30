// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
 
#ifndef CK_BLOCKWISE_GEMM_DLOPS_V3_HPP
#define CK_BLOCKWISE_GEMM_DLOPS_V3_HPP

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tall_and_skinny_gemm.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ABlockDesc_K0_M_K1,
          typename BThreadDesc_K0_N_K1,
          index_t MPerThread,
          index_t NPerBlock,
          index_t K0PerLoop>
struct BlockwiseGemmDlops_km_kn_m0m1n0n1_v3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    using CIndex = MultiIndex<4>;

    static constexpr auto K0 = ABlockDesc_K0_M_K1{}.GetLength(I0);
    static constexpr auto M  = ABlockDesc_K0_M_K1{}.GetLength(I1);
    static constexpr auto K1 = ABlockDesc_K0_M_K1{}.GetLength(I2);

    static constexpr auto NPerThread = BThreadDesc_K0_N_K1{}.GetLength(I1);

    static constexpr auto M0 = M / MPerThread;
    static constexpr auto M1 = MPerThread;

    static constexpr auto N  = NPerBlock;
    static constexpr auto N0 = N / NPerThread;
    static constexpr auto N1 = NPerThread;

    static constexpr auto a_thread_mtx_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<K0PerLoop>{}, Number<MPerThread>{}, Number<K1>{}));

    static constexpr auto b_thread_mtx_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<K0PerLoop>{}, Number<NPerThread>{}, Number<K1>{}));

    static constexpr auto c_thread_mtx_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<I1>{}, Number<M1>{}, Number<I1>{}, Number<N1>{}));

    __device__ BlockwiseGemmDlops_km_kn_m0m1n0n1_v3()
        : c_thread_origin_data_idx_{CalculateCThreadOriginOnBlock_BM0_BM1_BN0_BN1(
              get_thread_local_1d_id())},
          a_thread_copy_{make_tuple(0, c_thread_origin_data_idx_[I0] * MPerThread, 0)}
    {
        static_assert(ABlockDesc_K0_M_K1::IsKnownAtCompileTime() &&
                          BThreadDesc_K0_N_K1::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ABlockDesc_K0_M_K1{}.GetLength(I0) == BThreadDesc_K0_N_K1{}.GetLength(I0) &&
                          ABlockDesc_K0_M_K1{}.GetLength(I2) == BThreadDesc_K0_N_K1{}.GetLength(I2),
                      "wrong! E dimension not consistent\n");

        static_assert(K0 % K0PerLoop == 0, "");

        static_assert(M % MPerThread == 0 && N % NPerThread == 0,
                      "wrong! Cannot evenly divide work among\n");

        static_assert(BlockSize == M0 * N0, "wrong! wrong blocksize\n");
    }

    __device__ static constexpr auto GetCThreadTensorLengths_BM0_BM1_BN0_BN1()
    {
        return Sequence<I1, M1, I1, N1>{};
    }

    __device__ static CIndex CalculateCThreadOriginOnBlock_BM0_BM1_BN0_BN1(index_t thread_id)
    {
        constexpr auto c_threadid_to_m0_m1_n0_n1_thread_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M0, I1, N0, I1))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

        const auto c_m0_m1_n0_n1_thread_cluster_idx =
            c_threadid_to_m0_m1_n0_n1_thread_cluster_adaptor.CalculateBottomIndex(
                make_multi_index(thread_id));

        return c_m0_m1_n0_n1_thread_cluster_idx;
    }

    template <typename ABlockBuffer, typename BThreadBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BThreadBuffer& b_thread_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        static_assert(
            is_same<remove_cvref_t<typename ABlockBuffer::type>, remove_cvref_t<FloatA>>::value &&
            is_same<remove_cvref_t<typename BThreadBuffer::type>, remove_cvref_t<FloatB>>::value &&
            is_same<remove_cvref_t<typename CThreadBuffer::type>, remove_cvref_t<FloatC>>::value &&
            "wrong! inconsistent type");

        constexpr auto a_block_mtx = ABlockDesc_K0_M_K1{};

        // thread A buffer for GEMM
        StaticBuffer<AddressSpaceEnum::Vgpr, FloatA, a_thread_mtx_.GetElementSpaceSize(), true>
            a_thread_buf;

        constexpr auto threadwise_gemm = ThreadwiseGemmDlops_km_kn_mn_v3<FloatA,
                                                                         FloatB,
                                                                         FloatC,
                                                                         decltype(a_thread_mtx_),
                                                                         decltype(b_thread_mtx_),
                                                                         decltype(c_thread_mtx_)>{};

        static_for<0, K0, K0PerLoop>{}([&](auto k0_begin) {
            a_thread_copy_.Run(a_block_mtx,
                               make_tuple(k0_begin, I0, I0),
                               a_block_buf,
                               a_thread_mtx_,
                               make_tuple(I0, I0, I0),
                               a_thread_buf);

            threadwise_gemm.Run(a_thread_buf,
                                make_tuple(I0, I0, I0),
                                b_thread_buf,
                                make_tuple(k0_begin, I0, I0),
                                c_thread_buf,
                                make_tuple(I0, I0, I0, I0));
        });
    }

    private:
    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatA,
                                                         FloatA,
                                                         ABlockDesc_K0_M_K1,
                                                         decltype(a_thread_mtx_),
                                                         Sequence<K0PerLoop, MPerThread, K1>,
                                                         Sequence<0, 1, 2>,
                                                         2,
                                                         K1,
                                                         K1>;

    CIndex c_thread_origin_data_idx_;

    AThreadCopy a_thread_copy_;
};

} // namespace ck
#endif
