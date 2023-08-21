// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/amd_gemm_dpp.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_contraction_dl_dpp8.hpp"

namespace ck {

/**
 * DPP8 version of blockwise GEMM algorithm. It uses DPP8 instruction modifier to limit
 * the data loaded from LDS to registers.
 *
 * The algorithm groups threads into groups of size `dpp8::lane_group_size` and splits the matrix C
 * between them in such a way that threads from the same group need the same chunk of either
 * matrix A (or B, respectively). Without the usage of DPP8, each thread would need to load the
 * whole chunk from LDS to its own register space.
 * Usage of DPP8 modifiers allow each thread to load less data, exactly `1 / dpp8::lane_group_size`
 * of the chunk, and then share that data with other threads from the same lane group.
 *
 * Assumptions coming from the usage of DPP8:
 *   1. `BM10BN10ThreadClusterBM10Xs[1] == dpp8::lane_group_size` or
 *      `BM10BN10ThreadClusterBN10Xs[1] == dpp8::lane_group_size` -
 *        - it makes consecutive `dpp8::lane_group_size` threads use the same chunk of either
 *          matrix A or B;
 *        - based on these values we determine which matrix to share.
 *   2. `BM1PerThreadBM11 % dpp8::lane_group_size == 0` (if sharing A) or
 *      `BN1PerThreadBN11 % dpp8::lane_group_size == 0` (if sharing B) -
 *        - we have to make sure that the data to split is divisible by the number of
 *          threads in the group.
 *
 * General algorithm:
 * C[BM0, BM1, BN0, BN1] += transpose(A[K, BM0, BM1]) * B[K, BN0, BN1]
 * A and B are visible to the whole block, C is distributed among each thread
 * Assume:
 *   1. A:
 *     1. ABlockDesc_BK0_BM_BK1 is known at compile-time
 *     2. ABlockBuffer is DynamicBuffer
 *   2. B:
 *     1. BBlockDesc_BK0_BN_BK1 is known at compile-time
 *     2. BBlockBuffer is DynamicBuffer
 *   3. C:
 *     1. CThreadDesc_BM0_BM11_BN0_BN11 is known at compile-time
 *     2. CThreadBuffer is StaticBuffer
 *   4. BM10BN10ThreadClusterBM10Xs::Size() = BM10BN10ThreadClusterBN10Xs::Size() == 2
 */
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ABlockDesc_BK0_BM_BK1,
          typename BBlockDesc_BK0_BN_BK1,
          index_t BM1PerThreadBM11,
          index_t BN1PerThreadBN11,
          index_t BK0PerThread,
          typename BM10BN10ThreadClusterBM10Xs, // Sequence<BM10BN10ThreadClusterBM100,
                                                //          BM10BN10ThreadClusterBM101, ...>
          typename BM10BN10ThreadClusterBN10Xs, // Sequence<BM10BN10ThreadClusterBN100,
                                                //          BM10BN10ThreadClusterBN101, ...>
          index_t AThreadCopyScalarPerVector_BM11,
          index_t BThreadCopyScalarPerVector_BN11,
          typename enable_if<ABlockDesc_BK0_BM_BK1::IsKnownAtCompileTime() &&
                                 BBlockDesc_BK0_BN_BK1::IsKnownAtCompileTime(),
                             bool>::type = false>
struct BlockwiseGemmDlDpp8_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_loop_BM0_BN0
{
    using AIndex = MultiIndex<4>;
    using BIndex = MultiIndex<4>;
    using CIndex = MultiIndex<4>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr index_t BK0 = ABlockDesc_BK0_BM_BK1{}.GetLength(I0);
    static constexpr index_t BK1 = ABlockDesc_BK0_BM_BK1{}.GetLength(I2);
    static constexpr index_t BM  = ABlockDesc_BK0_BM_BK1{}.GetLength(I1);
    static constexpr index_t BN  = BBlockDesc_BK0_BN_BK1{}.GetLength(I1);

    static constexpr index_t BM100 = BM10BN10ThreadClusterBM10Xs{}[I0];
    static constexpr index_t BN100 = BM10BN10ThreadClusterBN10Xs{}[I0];

    static constexpr index_t BM101 = BM10BN10ThreadClusterBM10Xs{}[I1];
    static constexpr index_t BN101 = BM10BN10ThreadClusterBN10Xs{}[I1];

    static constexpr index_t BM11 = BM1PerThreadBM11;
    static constexpr index_t BN11 = BN1PerThreadBN11;

    static constexpr index_t BM1 = BM100 * BM101 * BM11;
    static constexpr index_t BN1 = BN100 * BN101 * BN11;

    static constexpr index_t BM0 = BM / BM1;
    static constexpr index_t BN0 = BN / BN1;

    // We assume that either `BM101` or `BN101` is equal to `dpp8::lane_group_size`. It makes all
    // threads in a lane group need the same chunk of B or A matrices and we can share them using
    // DPP.
    static_assert(BM101 == dpp8::lane_group_size || BN101 == dpp8::lane_group_size);
    static constexpr bool ShareB = BM101 == dpp8::lane_group_size ? true : false;
    static constexpr bool ShareA = !ShareB;

    // If DPP shares A (B, respectively), lane group gets `BM1PerThreadBM11` (`BN1PerThreadBN11`,
    // respectively) elements, so we split them between threads in lane group so each thread loads
    // less data from LDS.
    static constexpr index_t BM1PerThread =
        ShareA ? BM1PerThreadBM11 / dpp8::lane_group_size : BM1PerThreadBM11;
    static constexpr index_t BN1PerThread =
        ShareB ? BN1PerThreadBN11 / dpp8::lane_group_size : BN1PerThreadBN11;

    __host__ __device__ static constexpr auto
    MakeABlockDescriptor_BK0_BM0_BM1_BK1(const ABlockDesc_BK0_BM_BK1& a_block_desc_bk0_bm_bk1)
    {
        const auto a_block_bk0_bm0_bm1_bk1 = transform_tensor_descriptor(
            a_block_desc_bk0_bm_bk1,
            make_tuple(make_pass_through_transform(Number<BK0>{}),
                       make_unmerge_transform(make_tuple(Number<BM0>{}, Number<BM1>{})),
                       make_pass_through_transform(Number<BK1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return a_block_bk0_bm0_bm1_bk1;
    }

    __host__ __device__ static constexpr auto
    MakeBBlockDescriptor_BK0_BN0_BN1_BK1(const BBlockDesc_BK0_BN_BK1& b_block_desc_bk0_bn_bk1)
    {
        const auto b_block_desc_bk0_bn0_bn1_bk1 = transform_tensor_descriptor(
            b_block_desc_bk0_bn_bk1,
            make_tuple(make_pass_through_transform(Number<BK0>{}),
                       make_unmerge_transform(make_tuple(Number<BN0>{}, Number<BN1>{})),
                       make_pass_through_transform(Number<BK1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return b_block_desc_bk0_bn0_bn1_bk1;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM_BN()
    {
        // upper: [BM0, BM100, BM101, BM11, BN0, BN100, BN101, BN11]
        // lower: [BM, BN]
        constexpr auto c_block_adaptor_m0_m100_m101_m11_n0_n100_n101_n11_to_m_n =
            make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(
                               Number<BM0>{}, Number<BM100>{}, Number<BM101>{}, Number<BM11>{})),
                           make_unmerge_transform(make_tuple(
                               Number<BN0>{}, Number<BN100>{}, Number<BN101>{}, Number<BN11>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4, 5, 6, 7>{}));

        return c_block_adaptor_m0_m100_m101_m11_n0_n100_n101_n11_to_m_n;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM0_BM1_BN0_BN1()
    {
        // upper: [BM0, BM100, BM101, BM11, BN0, BN100, BN101, BN11]
        // lower: [BM0, BM1, BN0, BN1]
        constexpr auto c_block_adaptor_m0_m100_m101_m11_n0_n100_n101_n11_to_m0_m1_n0_n1 =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(Number<BM0>{}),
                           make_unmerge_transform(
                               make_tuple(Number<BM100>{}, Number<BM101>{}, Number<BM11>{})),
                           make_pass_through_transform(Number<BN0>{}),
                           make_unmerge_transform(
                               make_tuple(Number<BN100>{}, Number<BN101>{}, Number<BN11>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}, Sequence<5, 6, 7>{}));

        return c_block_adaptor_m0_m100_m101_m11_n0_n100_n101_n11_to_m0_m1_n0_n1;
    }

    __host__ __device__ static constexpr auto GetCThreadTensorLengths_BM0_BM1_BN0_BN1()
    {
        return Sequence<BM0, BM11, BN0, BN11>{};
    }

    static constexpr auto a_block_desc_bk0_bm0_bm1_bk1_ =
        MakeABlockDescriptor_BK0_BM0_BM1_BK1(ABlockDesc_BK0_BM_BK1{});

    static constexpr auto b_block_desc_bk0_bn0_bn1_bk1_ =
        MakeBBlockDescriptor_BK0_BN0_BN1_BK1(BBlockDesc_BK0_BN_BK1{});

    public:
    __device__ BlockwiseGemmDlDpp8_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_loop_BM0_BN0()
        : c_thread_origin_data_idx_{CalculateCThreadOriginOnBlock_BM0_BM1_BN0_BN1(
              get_thread_local_1d_id())},
          a_thread_copy_{CalculateAThreadOriginOnBlock_BK0_BM0_BM1_BK1()},
          b_thread_copy_{CalculateBThreadOriginOnBlock_BK0_BN0_BN1_BK1()}
    {
        static_assert(ABlockDesc_BK0_BM_BK1::IsKnownAtCompileTime() &&
                          BBlockDesc_BK0_BN_BK1::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(BM % BM1 == 0 && BN % BN1 == 0, "wrong!");

        static_assert(ABlockDesc_BK0_BM_BK1{}.GetLength(I0) ==
                          BBlockDesc_BK0_BN_BK1{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        static_assert(BM10BN10ThreadClusterBM10Xs::Size() == 2 &&
                          BM10BN10ThreadClusterBN10Xs::Size() == 2,
                      "wrong!");
    }

    __device__ static CIndex CalculateCThreadOriginOnBlock_BM0_BM1_BN0_BN1(index_t thread_id)
    {
        // lower: [BM0, BM1, BN0, BN1]
        // upper: [BM0, BM100, BM101, BM11, BN0, BN100, BN101, BN11]
        constexpr auto adaptor0 =
            MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM0_BM1_BN0_BN1();

        // lower: [BM0, BM100, BM101, BM11, BN0, BN100, BN101, BN11]
        // upper: [Tid, BM0, BM11, BN0, BN11]
        constexpr auto adaptor1 = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(BM100, BN100, BM101, BN101)),
                       make_pass_through_transform(BM0),
                       make_pass_through_transform(BM11),
                       make_pass_through_transform(BN0),
                       make_pass_through_transform(BN11)),
            make_tuple(
                Sequence<1, 5, 2, 6>{}, Sequence<0>{}, Sequence<3>{}, Sequence<4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        constexpr auto adaptor = chain_tensor_adaptors(adaptor0, adaptor1);

        return adaptor.CalculateBottomIndex(make_multi_index(thread_id, 0, 0, 0, 0));
    }

    __device__ AIndex CalculateAThreadOriginOnBlock_BK0_BM0_BM1_BK1()
    {
        const auto offsetBM0 = c_thread_origin_data_idx_[I0];
        // If sharing matrix A, we need a separate BM1 offset for each thread in lane group.
        const auto offsetBM1 = ShareA ? c_thread_origin_data_idx_[I1] +
                                            dpp8::get_thread_idx_in_lane_group() * BM1PerThread
                                      : c_thread_origin_data_idx_[I1];
        return make_tuple(0, offsetBM0, offsetBM1, 0);
    }

    __device__ BIndex CalculateBThreadOriginOnBlock_BK0_BN0_BN1_BK1()
    {
        const auto offsetBN0 = c_thread_origin_data_idx_[I2];
        // If sharing matrix B, we need a separate BN1 offset for each thread in lane group.
        const auto offsetBN1 = ShareB ? c_thread_origin_data_idx_[I3] +
                                            dpp8::get_thread_idx_in_lane_group() * BN1PerThread
                                      : c_thread_origin_data_idx_[I3];
        return make_tuple(0, offsetBN0, offsetBN1, 0);
    }

    template <typename CThreadDesc_BM0_BM11_BN0_BN11,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename CThreadBuffer>
    __device__ void Run(const CThreadDesc_BM0_BM11_BN0_BN11&,
                        const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        static_assert(CThreadDesc_BM0_BM11_BN0_BN11::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(
            a_thread_desc_bk0_bm0_bm1_bk1_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatB>(
            b_thread_desc_bk0_bn0_bn1_bk1_.GetElementSpaceSize());

        constexpr auto threadwise_contraction =
            ThreadwiseContractionDlDpp8_A_TK0_TM0_TM1_TK1_B_TK0_TN0_TN1_TK1_C_TM0_TM1_TN0_TN1<
                FloatA,
                FloatB,
                FloatC,
                decltype(a_thread_desc_bk0_bm0_bm1_bk1_),
                decltype(b_thread_desc_bk0_bn0_bn1_bk1_),
                CThreadDesc_BM0_BM11_BN0_BN11,
                Sequence<BK0PerThread, BK1>,
                Sequence<1, BM1PerThreadBM11>,
                Sequence<1, BN1PerThreadBN11>,
                ShareA>{};

        static_for<0, BN0, 1>{}([&](auto bn0) {
            static_for<0, BM0, 1>{}([&](auto bm0) {
                a_thread_copy_.Run(a_block_desc_bk0_bm0_bm1_bk1_,
                                   make_tuple(I0, bm0, I0, I0),
                                   a_block_buf,
                                   a_thread_desc_bk0_bm0_bm1_bk1_,
                                   make_tuple(I0, I0, I0, I0),
                                   a_thread_buf);

                b_thread_copy_.Run(b_block_desc_bk0_bn0_bn1_bk1_,
                                   make_tuple(I0, bn0, I0, I0),
                                   b_block_buf,
                                   b_thread_desc_bk0_bn0_bn1_bk1_,
                                   make_tuple(I0, I0, I0, I0),
                                   b_thread_buf);

                threadwise_contraction.Run(a_thread_buf,
                                           make_tuple(I0, I0, I0, I0),
                                           b_thread_buf,
                                           make_tuple(I0, I0, I0, I0),
                                           c_thread_buf,
                                           make_tuple(bm0, I0, bn0, I0));

                static_for<BK0PerThread, BK0, BK0PerThread>{}([&](auto bk0) {
                    a_thread_copy_.Run(a_block_desc_bk0_bm0_bm1_bk1_,
                                       make_tuple(bk0, bm0, I0, I0),
                                       a_block_buf,
                                       a_thread_desc_bk0_bm0_bm1_bk1_,
                                       make_tuple(I0, I0, I0, I0),
                                       a_thread_buf);

                    b_thread_copy_.Run(b_block_desc_bk0_bn0_bn1_bk1_,
                                       make_tuple(bk0, bn0, I0, I0),
                                       b_block_buf,
                                       b_thread_desc_bk0_bn0_bn1_bk1_,
                                       make_tuple(I0, I0, I0, I0),
                                       b_thread_buf);

                    threadwise_contraction.Run(a_thread_buf,
                                               make_tuple(I0, I0, I0, I0),
                                               b_thread_buf,
                                               make_tuple(I0, I0, I0, I0),
                                               c_thread_buf,
                                               make_tuple(bm0, I0, bn0, I0));
                });
            });
        });
    }

    private:
    // A[BK0, BM0, BM1, BK1]
    static constexpr auto a_thread_desc_bk0_bm0_bm1_bk1_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BK0PerThread>{}, Number<BM0>{}, Number<BM1PerThread>{}, Number<BK1>{}));

    // B[BK0, BN0, BN1, BK1]
    static constexpr auto b_thread_desc_bk0_bn0_bn1_bk1_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BK0PerThread>{}, Number<BN0>{}, Number<BN1PerThread>{}, Number<BK1>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4r1<
        FloatA,
        FloatA,
        decltype(a_block_desc_bk0_bm0_bm1_bk1_),
        decltype(a_thread_desc_bk0_bm0_bm1_bk1_),
        Sequence<BK0PerThread, 1, BM1PerThread, BK1>, // SliceLengths
        Sequence<0, 1, 2, 3>,                         // DimAccessOrder
        Sequence<1, 1, BM1PerThread, BK1>,            // SrcVectorTensorLengths
        Sequence<0, 1, 2, 3>>;                        // SrcVectorTensorContiguousDimOrder

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4r1<
        FloatB,
        FloatB,
        decltype(b_block_desc_bk0_bn0_bn1_bk1_),
        decltype(b_thread_desc_bk0_bn0_bn1_bk1_),
        Sequence<BK0PerThread, 1, BN1PerThread, BK1>, // SliceLengths
        Sequence<0, 1, 2, 3>,                         // DimAccessOrder
        Sequence<1, 1, BN1PerThread, BK1>,            // SrcVectorTensorLengths
        Sequence<0, 1, 2, 3>>;                        // SrcVectorTensorContiguousDimOrder

    CIndex c_thread_origin_data_idx_;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
