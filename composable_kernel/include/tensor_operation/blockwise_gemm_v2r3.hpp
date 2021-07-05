#ifndef CK_BLOCKWISE_GEMM_V2R3_HPP
#define CK_BLOCKWISE_GEMM_V2R3_HPP

#include "common_header.hpp"
#include "tensor_adaptor.hpp"
#include "threadwise_dynamic_tensor_slice_transfer_v2.hpp"
#include "threadwise_gemm_v2.hpp"

namespace ck {

// C[M0, M1, N0, N1] += transpose(A[K, M0, M1]) * B[K, N0, N1]
// A and B are visable to the whole block, C is distributed among each thread
// Assume:
//   1. A:
//     1. AK0MK1BlockDesc is known at compile-time
//     2. ABlockBuffer is DynamicBuffer
//   2. B:
//     1. BK0NK1BlockDesc is known at compile-time
//     2. BBlockBuffer is DynamicBuffer
//   3. C:
//     1. CM0M1N0N1ThreadDesc is known at compile-time
//     2. CThreadBuffer is StaticBuffer
// Also assume:
//   M0 = N0 = 2. It will do 2x2 pipelined read and fma (ABBA optimization)
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t M1PerThreadM11,
          index_t N1PerThreadN11,
          index_t KPerThread,
          index_t M1N1ThreadClusterM100,
          index_t M1N1ThreadClusterN100,
          index_t M1N1ThreadClusterM101,
          index_t M1N1ThreadClusterN101,
          index_t AThreadCopyScalarPerVector_M11,
          index_t BThreadCopyScalarPerVector_N11,
          typename std::enable_if<AK0MK1BlockDesc::IsKnownAtCompileTime() &&
                                      BK0NK1BlockDesc::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct BlockwiseGemm_k0mk1_k0nk1_m0m1n0n1_v2r3_pipeline_2x2
{
    using AIndex = MultiIndex<3>;
    using BIndex = MultiIndex<3>;
    using CIndex = MultiIndex<4>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr index_t K0 = AK0MK1BlockDesc{}.GetLength(I0);
    static constexpr index_t K1 = AK0MK1BlockDesc{}.GetLength(I2);
    static constexpr index_t M  = AK0MK1BlockDesc{}.GetLength(I1);
    static constexpr index_t N  = BK0NK1BlockDesc{}.GetLength(I1);

    static constexpr index_t M100 = M1N1ThreadClusterM100;
    static constexpr index_t N100 = M1N1ThreadClusterN100;

    static constexpr index_t M101 = M1N1ThreadClusterM101;
    static constexpr index_t N101 = M1N1ThreadClusterN101;

    static constexpr index_t M11 = M1PerThreadM11;
    static constexpr index_t N11 = N1PerThreadN11;

    static constexpr index_t M1 = M1N1ThreadClusterM100 * M1N1ThreadClusterM101 * M1PerThreadM11;
    static constexpr index_t N1 = M1N1ThreadClusterN100 * M1N1ThreadClusterN101 * N1PerThreadN11;

    static constexpr index_t M0 = M / M1;
    static constexpr index_t N0 = N / N1;

    __host__ __device__ static constexpr auto
    MakeAK0M0M1K1BlockDescriptor(const AK0MK1BlockDesc& a_k0_m_k1_block_desc)
    {
        const auto a_k0_m0_m1_k1_block_desc = transform_dynamic_tensor_descriptor(
            a_k0_m_k1_block_desc,
            make_tuple(make_pass_through_transform(Number<K0>{}),
                       make_unmerge_transform(make_tuple(Number<M0>{}, Number<M1>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return a_k0_m0_m1_k1_block_desc;
    }

    __host__ __device__ static constexpr auto
    MakeBK0N0N1K1BlockDescriptor(const BK0NK1BlockDesc& b_k0_n_k1_block_desc)
    {
        const auto b_k0_n0_n1_k1_block_desc = transform_dynamic_tensor_descriptor(
            b_k0_n_k1_block_desc,
            make_tuple(make_pass_through_transform(Number<K0>{}),
                       make_unmerge_transform(make_tuple(Number<N0>{}, Number<N1>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return b_k0_n0_n1_k1_block_desc;
    }

    __host__ __device__ static constexpr auto MakeCM0M100M101M11N0N100N101N11ToMNBlockAdaptor()
    {
        // upper: [M0, M100, M101, M11, N0, N100, N101, N11]
        // lower: [M, N]
        constexpr auto c_m0_m100_m101_m11_n0_n100_n101_n11_to_m_n_block_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(
                               Number<M0>{}, Number<M100>{}, Number<M101>{}, Number<M11>{})),
                           make_unmerge_transform(make_tuple(
                               Number<N0>{}, Number<N100>{}, Number<N101>{}, Number<N11>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4, 5, 6, 7>{}));

        return c_m0_m100_m101_m11_n0_n100_n101_n11_to_m_n_block_adaptor;
    }

    __host__ __device__ static constexpr auto
    MakeCM0M100M101M11N0N100N101N11ToM0M1N0N1BlockAdaptor()
    {
        // upper: [M0, M100, M101, M11, N0, N100, N101, N11]
        // lower: [M0, M1, N0, N1]
        constexpr auto c_m0_m100_m101_m11_n0_n100_n101_n11_to_m0_m1_n0_n1_block_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(Number<M0>{}),
                           make_unmerge_transform(
                               make_tuple(Number<M100>{}, Number<M101>{}, Number<M11>{})),
                           make_pass_through_transform(Number<N0>{}),
                           make_unmerge_transform(
                               make_tuple(Number<N100>{}, Number<N101>{}, Number<N11>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}, Sequence<5, 6, 7>{}));

        return c_m0_m100_m101_m11_n0_n100_n101_n11_to_m0_m1_n0_n1_block_adaptor;
    }

    __host__ __device__ static constexpr auto GetCM0M1N0N1ThreadTensorLengths()
    {
        return Sequence<M0, M11, N0, N11>{};
    }

    static constexpr auto a_k0_m0_m1_k1_block_desc_ =
        MakeAK0M0M1K1BlockDescriptor(AK0MK1BlockDesc{});
    static constexpr auto b_k0_n0_n1_k1_block_desc_ =
        MakeBK0N0N1K1BlockDescriptor(BK0NK1BlockDesc{});

    public:
    __device__ BlockwiseGemm_k0mk1_k0nk1_m0m1n0n1_v2r3_pipeline_2x2()
        : c_thread_origin_data_idx_{CalculateCM0M1N0N1ThreadOriginOnBlock(
              get_thread_local_1d_id())},
          a_thread_copy_{
              make_tuple(0, c_thread_origin_data_idx_[I0], c_thread_origin_data_idx_[I1], 0)},
          b_thread_copy_{
              make_tuple(0, c_thread_origin_data_idx_[I2], c_thread_origin_data_idx_[I3], 0)}
    {
        static_assert(AK0MK1BlockDesc::IsKnownAtCompileTime() &&
                          BK0NK1BlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(BlockSize == M101 * M100 * N101 * N100,
                      "wrong! blocksize and cluster size not consistent");

        static_assert(M % M1 == 0 && N % N1 == 0, "wrong!");

        static_assert(AK0MK1BlockDesc{}.GetLength(I0) == BK0NK1BlockDesc{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        // TODO: remove this restriction
        static_assert(M0 == 2 && N0 == 2, "wrong");
    }

    __device__ static CIndex CalculateCM0M1N0N1ThreadOriginOnBlock(index_t thread_id)
    {
        // lower: [M0, M1, N0, N1]
        // upper: [M0, M100, M101, M11, N0, N100, N101, N11]
        constexpr auto adaptor0 = MakeCM0M100M101M11N0N100N101N11ToM0M1N0N1BlockAdaptor();

        // lower: [M0, M100, M101, M11, N0, N100, N101, N11]
        // upper: [Tid, M0, M11, N0, N11]
        constexpr auto adaptor1 = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(M100, N100, M101, N101)),
                       make_pass_through_transform(M0),
                       make_pass_through_transform(M11),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(N11)),
            make_tuple(
                Sequence<1, 5, 2, 6>{}, Sequence<0>{}, Sequence<3>{}, Sequence<4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        constexpr auto adaptor = chain_tensor_adaptors(adaptor0, adaptor1);

        return adaptor.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id(), 0, 0, 0, 0));
    }

    template <typename CM0M1N0N1ThreadDesc,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename CThreadBuffer>
    __device__ void Run(const CM0M1N0N1ThreadDesc& c_m0_m1_n0_n1_thread_desc,
                        const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        static_assert(CM0M1N0N1ThreadDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        // TODO: remove this restriction
        static_assert(M0 == 2 && N0 == 2 && CM0M1N0N1ThreadDesc{}.GetLength(I0) == M0 &&
                          CM0M1N0N1ThreadDesc{}.GetLength(I2) == N0,
                      "wrong");

        auto a_thread_buf = make_static_buffer<AddressSpace::Vgpr, FloatA>(
            a_k0_m0_m1_k1_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpace::Vgpr, FloatB>(
            b_k0_n0_n1_k1_thread_desc_.GetElementSpaceSize());

        constexpr auto threadwise_gemm =
            ThreadwiseGemm_k0m0m1k1_k0n0n1k1_m0m1n0n1<FloatA,
                                                      FloatB,
                                                      FloatC,
                                                      decltype(a_k0_m0_m1_k1_thread_desc_),
                                                      decltype(b_k0_n0_n1_k1_thread_desc_),
                                                      CM0M1N0N1ThreadDesc,
                                                      Sequence<KPerThread, K1>,
                                                      Sequence<1, M1PerThreadM11>,
                                                      Sequence<1, N1PerThreadN11>>{};

        // read A_sub_0
        a_thread_copy_.Run(a_k0_m0_m1_k1_block_desc_,
                           make_tuple(I0, I0, I0, I0),
                           a_block_buf,
                           a_k0_m0_m1_k1_thread_desc_,
                           make_tuple(I0, I0, I0, I0),
                           a_thread_buf);

        // read B_sub_0
        b_thread_copy_.Run(b_k0_n0_n1_k1_block_desc_,
                           make_tuple(I0, I0, I0, I0),
                           b_block_buf,
                           b_k0_n0_n1_k1_thread_desc_,
                           make_tuple(I0, I0, I0, I0),
                           b_thread_buf);

        // read B_sub_1
        b_thread_copy_.Run(b_k0_n0_n1_k1_block_desc_,
                           make_tuple(I0, I1, I0, I0),
                           b_block_buf,
                           b_k0_n0_n1_k1_thread_desc_,
                           make_tuple(I0, I1, I0, I0),
                           b_thread_buf);

        // read A_sub_1
        a_thread_copy_.Run(a_k0_m0_m1_k1_block_desc_,
                           make_tuple(I0, I1, I0, I0),
                           a_block_buf,
                           a_k0_m0_m1_k1_thread_desc_,
                           make_tuple(I0, I1, I0, I0),
                           a_thread_buf);

        // C_sub_00 += transpose(A_sub_0) * B_sub_0
        threadwise_gemm.Run(a_thread_buf,
                            make_tuple(I0, I0, I0, I0),
                            b_thread_buf,
                            make_tuple(I0, I0, I0, I0),
                            c_thread_buf,
                            make_tuple(I0, I0, I0, I0));

        // C_sub_01 += transpose(A_sub_0) * B_sub_1
        threadwise_gemm.Run(a_thread_buf,
                            make_tuple(I0, I0, I0, I0),
                            b_thread_buf,
                            make_tuple(I0, I1, I0, I0),
                            c_thread_buf,
                            make_tuple(I0, I0, I1, I0));

        // loop over rest of k
        static_for<KPerThread, K0, KPerThread>{}([&](auto k) {
            // read A_sub_0
            a_thread_copy_.Run(a_k0_m0_m1_k1_block_desc_,
                               make_tuple(k, I0, I0, I0),
                               a_block_buf,
                               a_k0_m0_m1_k1_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               a_thread_buf);

            // C_sub_10 += transpose(A_sub_1) * B_sub_0
            threadwise_gemm.Run(a_thread_buf,
                                make_tuple(I0, I1, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I0, I0, I0),
                                c_thread_buf,
                                make_tuple(I1, I0, I0, I0));

            // read B_sub_0
            b_thread_copy_.Run(b_k0_n0_n1_k1_block_desc_,
                               make_tuple(k, I0, I0, I0),
                               b_block_buf,
                               b_k0_n0_n1_k1_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               b_thread_buf);

            // C_sub_11 += transpose(A_sub_1) * B_sub_1
            threadwise_gemm.Run(a_thread_buf,
                                make_tuple(I0, I1, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I1, I0, I0),
                                c_thread_buf,
                                make_tuple(I1, I0, I1, I0));

            // read B_sub_1
            b_thread_copy_.Run(b_k0_n0_n1_k1_block_desc_,
                               make_tuple(k, I1, I0, I0),
                               b_block_buf,
                               b_k0_n0_n1_k1_thread_desc_,
                               make_tuple(I0, I1, I0, I0),
                               b_thread_buf);

            // read A_sub_1
            a_thread_copy_.Run(a_k0_m0_m1_k1_block_desc_,
                               make_tuple(k, I1, I0, I0),
                               a_block_buf,
                               a_k0_m0_m1_k1_thread_desc_,
                               make_tuple(I0, I1, I0, I0),
                               a_thread_buf);

            // C_sub_00 += transpose(A_sub_0) * B_sub_0
            threadwise_gemm.Run(a_thread_buf,
                                make_tuple(I0, I0, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I0, I0, I0),
                                c_thread_buf,
                                make_tuple(I0, I0, I0, I0));

            // C_sub_01 += transpose(A_sub_0) * B_sub_1
            threadwise_gemm.Run(a_thread_buf,
                                make_tuple(I0, I0, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I1, I0, I0),
                                c_thread_buf,
                                make_tuple(I0, I0, I1, I0));
        });

        // C_sub_10 += transpose(A_sub_1) * B_sub_0
        threadwise_gemm.Run(a_thread_buf,
                            make_tuple(I0, I1, I0, I0),
                            b_thread_buf,
                            make_tuple(I0, I0, I0, I0),
                            c_thread_buf,
                            make_tuple(I1, I0, I0, I0));

        // C_sub_11 += transpose(A_sub_1) * B_sub_1
        threadwise_gemm.Run(a_thread_buf,
                            make_tuple(I0, I1, I0, I0),
                            b_thread_buf,
                            make_tuple(I0, I1, I0, I0),
                            c_thread_buf,
                            make_tuple(I1, I0, I1, I0));
    }

    private:
    // A[K0, M0, M1, K1]
    static constexpr auto a_k0_m0_m1_k1_thread_desc_ =
        make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<KPerThread>{}, Number<M0>{}, Number<M1PerThreadM11>{}, Number<K1>{}));

    // B[K0, N0, N1, K1]
    static constexpr auto b_k0_n0_n1_k1_thread_desc_ =
        make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<KPerThread>{}, Number<N0>{}, Number<N1PerThreadN11>{}, Number<K1>{}));

    using AThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4r1<
        FloatA,
        FloatA,
        decltype(a_k0_m0_m1_k1_block_desc_),
        decltype(a_k0_m0_m1_k1_thread_desc_),
        Sequence<KPerThread, 1, M1PerThreadM11, K1>, // SliceLengths
        Sequence<0, 1, 2, 3>,                        // DimAccessOrder
        Sequence<1, 1, M1PerThreadM11, K1>,          // SrcVectorTensorLengths
        Sequence<0, 1, 2, 3>>;                       // SrcVectorTensorContiguousDimOrder

    using BThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4r1<
        FloatB,
        FloatB,
        decltype(b_k0_n0_n1_k1_block_desc_),
        decltype(b_k0_n0_n1_k1_thread_desc_),
        Sequence<KPerThread, 1, N1PerThreadN11, K1>, // SliceLengths
        Sequence<0, 1, 2, 3>,                        // DimAccessOrder
        Sequence<1, 1, N1PerThreadN11, K1>,          // SrcVectorTensorLengths
        Sequence<0, 1, 2, 3>>;                       // SrcVectorTensorContiguousDimOrder

    CIndex c_thread_origin_data_idx_;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
#endif
