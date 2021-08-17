#ifndef CK_BLOCKWISE_GEMM_XDLOPS_HPP
#define CK_BLOCKWISE_GEMM_XDLOPS_HPP

#include "common_header.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "xdlops_gemm.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t K1>
struct BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using CIndex = MultiIndex<2>;

    static constexpr index_t WaveSize = 64;

    static constexpr index_t MPerBlock = AK0MK1BlockDesc{}.GetLength(I1);
    static constexpr index_t NPerBlock = BK0NK1BlockDesc{}.GetLength(I1);

    static constexpr index_t K0        = BK0NK1BlockDesc{}.GetLength(I0);
    static constexpr index_t KPerBlock = K0;

    static constexpr auto xdlops_gemm = XdlopsGemm<FloatAB, MPerXDL, NPerXDL, K1>{};

    static constexpr auto CXdlopsLayout = xdlops_gemm.GetCXdlopsLayout();

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerXDL);

    __device__ static constexpr auto GetCM0N0M1N1M2M3M4N2ThreadDesc()
    {
        constexpr auto M0 = Number<CXdlopsLayout.M1()>{};
        constexpr auto M2 = Number<CXdlopsLayout.M0()>{};

        return make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, I1, M0, I1, M2, I1));
    }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        const auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
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

        return make_tuple(xdlops_a_idx[I0], 0, waveId_m, xdlops_a_idx[I1], 0);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(xdlops_b_idx[I0], 0, waveId_n, xdlops_b_idx[I1], 0);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static CIndex
        CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        constexpr auto mrepeat_mwave_mperxdl_to_m = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<MWaves>{}, Number<MPerXDL>{}));

        constexpr auto nrepeat_nwave_nperxdl_to_n = make_naive_tensor_descriptor_packed(
            make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerXDL>{}));

        const index_t c_thread_m =
            mrepeat_mwave_mperxdl_to_m.CalculateOffset(make_tuple(m0, waveId_m, blk_idx[I0]));
        const index_t c_thread_n =
            nrepeat_nwave_nperxdl_to_n.CalculateOffset(make_tuple(n0, waveId_n, blk_idx[I1]));

        return CIndex{c_thread_m, c_thread_n};
    }

    __host__ __device__ BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1()
    {
        static_assert(AK0MK1BlockDesc::IsKnownAtCompileTime() &&
                          BK0NK1BlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(AK0MK1BlockDesc{}.GetLength(I0) == BK0NK1BlockDesc{}.GetLength(I0),
                      "wrong! K0 dimension not consistent");

        static_assert(AK0MK1BlockDesc{}.GetLength(I2) == BK0NK1BlockDesc{}.GetLength(I2),
                      "wrong! K1 dimension not consistent");

        static_assert(BlockSize == MWaves * NWaves * WaveSize,
                      "BlockSize != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");

        constexpr index_t NumBlks   = CXdlopsLayout.GetNumBlks();
        constexpr index_t NumXdlops = CXdlopsLayout.GetNumXdlops();

        static_assert(NumBlks == 1 && NumXdlops == 1, "K Reduction Mfma only");
    }

    __host__ __device__ static constexpr auto GetCM0N0M1N1M2M3M4N2BlockDescriptor()
    {
        constexpr auto M2 = Number<CXdlopsLayout.M1()>{};
        constexpr auto M3 = Number<CXdlopsLayout.N1()>{};
        constexpr auto M4 = Number<CXdlopsLayout.M0()>{};
        constexpr auto N2 = Number<CXdlopsLayout.N0()>{};

        return make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                              Number<NRepeat>{},
                                                              Number<MWaves>{},
                                                              Number<NWaves>{},
                                                              Number<M2>{},
                                                              Number<M3>{},
                                                              Number<M4>{},
                                                              Number<N2>{}));
    }

    template <typename CMNGridDesc>
    __host__ __device__ static constexpr auto
    MakeCM0N0M1N1M2M3M4N2GridDescriptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        ///\To-do: pass CGrid desc transform deep inside xdlops gemm
        constexpr auto M2 = Number<CXdlopsLayout.M1()>{};
        constexpr auto M3 = Number<CXdlopsLayout.N1()>{};
        constexpr auto M4 = Number<CXdlopsLayout.M0()>{};
        constexpr auto N2 = Number<CXdlopsLayout.N0()>{};

        return transform_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, M2, M3, M4)),
                       make_unmerge_transform(make_tuple(NRepeat, NWaves, N2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));
    }

    __host__ __device__ static constexpr auto MakeAK0M0M1M2K1BlockDescriptor()
    {
        return transform_tensor_descriptor(
            AK0MK1BlockDesc{},
            make_tuple(make_pass_through_transform(Number<KPerBlock>{}),
                       make_unmerge_transform(
                           make_tuple(Number<MRepeat>{}, Number<MWaves>{}, Number<MPerXDL>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));
    }

    __host__ __device__ static constexpr auto MakeBK0N0N1N2K1BlockDescriptor()
    {
        return transform_tensor_descriptor(
            BK0NK1BlockDesc{},
            make_tuple(make_pass_through_transform(Number<KPerBlock>{}),
                       make_unmerge_transform(
                           make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerXDL>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));
    }

    static constexpr auto a_k0_m0_m1_m2_k1_block_desc = MakeAK0M0M1M2K1BlockDescriptor();
    static constexpr auto b_k0_n0_n1_n2_k1_block_desc = MakeBK0N0N1N2K1BlockDescriptor();

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum_t::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum_t::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        vector_type<FloatAB, K1> a_thread_vec;

        vector_type<FloatAB, K1> b_thread_vec;

        static_for<0, KPerBlock, xdlops_gemm.KPerXdlops>{}([&](auto k0) {
            // read A
            a_thread_copy_.Run(a_k0_m0_m1_m2_k1_block_desc,
                               make_tuple(k0, I0, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0, I0, I0),
                               a_thread_buf);

            // read B
            b_thread_copy_.Run(b_k0_n0_n1_n2_k1_block_desc,
                               make_tuple(k0, I0, I0, I0, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I0, I0, I0, I0),
                               b_thread_buf);

            using mfma_input_type =
                typename vector_type<FloatAB, xdlops_gemm.mfma_type.k_per_blk>::type;

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, K1, 1>{}([&](auto i) {
                        a_thread_vec.template AsType<FloatAB>()(i) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(0, m0, 0, 0, i))>{}];
                    });

                    static_for<0, K1, 1>{}([&](auto i) {
                        b_thread_vec.template AsType<FloatAB>()(i) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(0, n0, 0, 0, i))>{}];
                    });

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    xdlops_gemm.template Run<c_offset>(
                        a_thread_vec.template AsType<mfma_input_type>(),
                        b_thread_vec.template AsType<mfma_input_type>(),
                        c_thread_buf);
                });
            });
        });
    }

    private:
    // A[K, M]
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(I1, Number<MRepeat>{}, I1, I1, Number<K1>{}));

    // B[K, N]
    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(I1, Number<NRepeat>{}, I1, I1, Number<K1>{}));

    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, Number<xdlops_gemm.GetNumXdlops()>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_k0_m0_m1_m2_k1_block_desc),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, MRepeat, 1, 1, K1>,
                                                         Sequence<0, 1, 2, 3, 4>,
                                                         4,
                                                         K1,
                                                         1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_k0_n0_n1_n2_k1_block_desc),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, NRepeat, 1, 1, K1>,
                                                         Sequence<0, 1, 2, 3, 4>,
                                                         4,
                                                         K1,
                                                         1>;

    AThreadCopy a_thread_copy_{CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{CalculateBThreadOriginDataIndex()};
};

} // namespace ck
#endif
