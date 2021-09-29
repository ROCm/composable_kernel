#ifndef CK_BLOCKWISE_GEMM_DLOPS_V3_HPP
#define CK_BLOCKWISE_GEMM_DLOPS_V3_HPP

#include "common_header.hpp"
#include "threadwise_gemm_dlops_v3.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ABlockDesc_E1_K_E2,
          typename BBlockDesc_E1_N_Ho_Wo_E2,
          typename CThreadDesc_K_N_Ho_Wo,
          index_t EPerThreadLoop>
struct BlockwiseGemmDlops_km_kn_m0m1n0n1_v3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    struct MatrixIndex
    {
        index_t k;
        index_t n;
        index_t h;
        index_t w;
    };

    static constexpr auto E1 = ABlockDesc_E1_K_E2{}.GetLength(I0);
    static constexpr auto K  = ABlockDesc_E1_K_E2{}.GetLength(I1);
    static constexpr auto E2 = ABlockDesc_E1_K_E2{}.GetLength(I2);

    static constexpr auto H = BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I2);
    static constexpr auto W = BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I3);

    static constexpr auto KPerThread = CThreadDesc_K_N_Ho_Wo{}.GetLength(I0);
    static constexpr auto HPerThread = CThreadDesc_K_N_Ho_Wo{}.GetLength(I2);
    static constexpr auto WPerThread = CThreadDesc_K_N_Ho_Wo{}.GetLength(I3);

    static constexpr index_t KPerThreadLoop = 2;

    static constexpr auto a_thread_mtx_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<EPerThreadLoop>{}, Number<KPerThreadLoop>{}, Number<E2>{}));

    static constexpr auto b_thread_mtx_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<EPerThreadLoop>{},
                                                       Number<1>{},
                                                       Number<HPerThread>{},
                                                       Number<WPerThread>{},
                                                       Number<E2>{}));

    static constexpr auto c_thread_mtx_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<KPerThreadLoop>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

    __device__ BlockwiseGemmDlops_km_kn_m0m1n0n1_v3()
        : c_thread_begin_mtx_idx_{GetBeginOfCThreadDesc_K_N_Ho_Wo(get_thread_local_1d_id())},
          a_thread_copy_{make_tuple(0, c_thread_begin_mtx_idx_.k * KPerThread, 0)}
    {
        static_assert(ABlockDesc_E1_K_E2::IsKnownAtCompileTime() &&
                          BBlockDesc_E1_N_Ho_Wo_E2::IsKnownAtCompileTime() &&
                          CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(
            ABlockDesc_E1_K_E2{}.GetLength(I0) == BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I0) &&
                ABlockDesc_E1_K_E2{}.GetLength(I2) == BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I4),
            "wrong! E dimension not consistent\n");

        static_assert(E1 % EPerThreadLoop == 0, "");
        static_assert(KPerThread % KPerThreadLoop == 0, "");

        static_assert(K % KPerThread == 0 && H % HPerThread == 0 && W % WPerThread == 0,
                      "wrong! Cannot evenly divide work among\n");

        constexpr auto KThreadCluster = K / KPerThread;
        constexpr auto HThreadCluster = H / HPerThread;
        constexpr auto WThreadCluster = W / WPerThread;

        static_assert(BlockSize == KThreadCluster * HThreadCluster * WThreadCluster,
                      "wrong! wrong blocksize\n");
    }

    __device__ static constexpr auto GetCThreadDesc_K_N_Ho_WoLengths()
    {
        return Sequence<KPerThread, 1, HPerThread, WPerThread>{};
    }

    __device__ static MatrixIndex GetBeginOfCThreadDesc_K_N_Ho_Wo(index_t thread_id)
    {
        constexpr index_t HPerBlock = BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I2);
        constexpr index_t WPerBlock = BBlockDesc_E1_N_Ho_Wo_E2{}.GetLength(I3);

        constexpr auto num_w_threads  = WPerBlock / WPerThread;
        constexpr auto num_h_threads  = HPerBlock / HPerThread;
        constexpr auto num_hw_threads = num_w_threads * num_h_threads;

        index_t k_thread_id  = thread_id / num_hw_threads;
        index_t hw_thread_id = thread_id % num_hw_threads;

        index_t h_thread_id = hw_thread_id / num_w_threads;
        index_t w_thread_id = hw_thread_id % num_w_threads;

        return MatrixIndex{k_thread_id, 1, h_thread_id, w_thread_id};
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

        constexpr auto a_block_mtx = ABlockDesc_E1_K_E2{};

        // thread A buffer for GEMM
        StaticBuffer<AddressSpaceEnum_t::Vgpr, FloatA, a_thread_mtx_.GetElementSpaceSize(), true>
            a_thread_buf;

        constexpr auto threadwise_gemm = ThreadwiseGemmDlops_km_kn_mn_v3<FloatA,
                                                                         FloatB,
                                                                         FloatC,
                                                                         decltype(a_thread_mtx_),
                                                                         decltype(b_thread_mtx_),
                                                                         decltype(c_thread_mtx_)>{};

        static_for<0, E1, EPerThreadLoop>{}([&](auto e_begin) {
            static_for<0, KPerThread, KPerThreadLoop>{}([&](auto k_begin) {
                a_thread_copy_.Run(a_block_mtx,
                                   make_tuple(e_begin, k_begin, I0),
                                   a_block_buf,
                                   a_thread_mtx_,
                                   make_tuple(I0, I0, I0),
                                   a_thread_buf);

                threadwise_gemm.Run(a_thread_buf,
                                    make_tuple(I0, I0, I0),
                                    b_thread_buf,
                                    make_tuple(e_begin, I0, I0, I0, I0),
                                    c_thread_buf,
                                    make_tuple(k_begin, I0, I0, I0));
            });
        });
    }

    template <typename ABlockSliceMoveStepIdx>
    __device__ void MoveABlockSliceWindow(const ABlockSliceMoveStepIdx& a_block_slice_move_step_idx)
    {
        a_thread_copy_.MoveSrcSliceWindow(ABlockDesc_E1_K_E2{}, a_block_slice_move_step_idx);
    }

    private:
    MatrixIndex c_thread_begin_mtx_idx_;

    using AThreadCopy =
        ThreadwiseTensorSliceTransfer_v4<FloatA,
                                         FloatA,
                                         ABlockDesc_E1_K_E2,
                                         decltype(a_thread_mtx_),
                                         Sequence<EPerThreadLoop, KPerThreadLoop, E2>,
                                         Sequence<0, 1, 2>,
                                         2,
                                         E2,
                                         E2>;

    AThreadCopy a_thread_copy_;
};

} // namespace ck
#endif
