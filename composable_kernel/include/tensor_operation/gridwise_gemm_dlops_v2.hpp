#ifndef CK_GRIDWISE_GEMM_V2_HPP
#define CK_GRIDWISE_GEMM_V2_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "blockwise_gemm_dlops_v3.hpp"

namespace ck {

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v2(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AGridDesc_E0_E1_K0_K1_E2 a_e0_e1_k0_k1_e2_grid_desc,
            const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
            const CGridDesc_K_N_H0_H1_H2_W0_W1_W2 c_k_n_h0_h1_h2_w0_w1_w2_grid_desc,
            const CBlockIdToBlockClusterAdaptor_K_N_H_W c_blockid_to_k_n_h_w_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_e0_e1_k0_k1_e2_grid_desc,
                      b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                      c_k_n_h0_h1_h2_w0_w1_w2_grid_desc,
                      c_blockid_to_k_n_h_w_block_cluster_adaptor,
                      integral_constant<bool, HasMainE0BlockLoop>{});
}
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
// pass tensor descriptor by CONSTANT void pointer
// CONSTANT is needed to inform compiler void pointers in the kernel signature are pointing to
// non-modifiable parameter address space, so compiler can enable corresponding optimization
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v2(const FloatAB* __restrict__ p_a_grid,
                             const FloatAB* __restrict__ p_b_grid,
                             FloatC* __restrict__ p_c_grid,
                             const void CONSTANT* p_a_e0_e1_k0_k1_e2_grid_desc,
                             const void CONSTANT* p_b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                             const void CONSTANT* p_c_k_n_h0_h1_h2_w0_w1_w2_grid_desc,
                             const void CONSTANT* p_c_blockid_to_k_n_h_w_block_cluster_adaptor)
{
    // first cast void CONSTANT void* to void*
    // second cast void* to Desc*
    // the copy constructor of tensor descriptor doesn't take address_space(4)
    const auto a_e0_e1_k0_k1_e2_grid_desc = *reinterpret_cast<const AGridDesc_E0_E1_K0_K1_E2*>(
        cast_pointer_to_generic_address_space(p_a_e0_e1_k0_k1_e2_grid_desc));
    const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        *reinterpret_cast<const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2*>(
            cast_pointer_to_generic_address_space(p_b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc));
    const auto c_k_n_h0_h1_h2_w0_w1_w2_grid_desc =
        *reinterpret_cast<const CGridDesc_K_N_H0_H1_H2_W0_W1_W2*>(
            cast_pointer_to_generic_address_space(p_c_k_n_h0_h1_h2_w0_w1_w2_grid_desc));
    const auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        *reinterpret_cast<const CBlockIdToBlockClusterAdaptor_K_N_H_W*>(
            cast_pointer_to_generic_address_space(p_c_blockid_to_k_n_h_w_block_cluster_adaptor));

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_e0_e1_k0_k1_e2_grid_desc,
                      b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                      c_k_n_h0_h1_h2_w0_w1_w2_grid_desc,
                      c_blockid_to_k_n_h_w_block_cluster_adaptor,
                      integral_constant<bool, HasMainE0BlockLoop>{});
}
#endif

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename AGridDesc_E0_E1_K_E2,
          typename BGridDesc_E0_E1_N_Ho_Wo_E2,
          typename CGridDesc_K_N_Ho_Wo,
          index_t E1_,
          index_t E2_,
          index_t K2_,
          index_t KPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t E1PerBlock,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
          typename ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_E2,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGlobalStepHacks,
          typename BGlobalStepHacks,
          typename CGlobalStepHacks,
          typename AGlobalMoveSliceWindowStepHacks,
          typename BGlobalMoveSliceWindowStepHacks,
          index_t activ_type = 0>
struct GridwiseGemmDlops_km_kn_mn_v3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto E1 = Number<E1_>{};
    static constexpr auto E2 = Number<E2_>{};
    static constexpr auto K2 = Number<K2_>{};

    static constexpr auto NPerBlock = I1;

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = Number<ABlockTransferDstScalarPerVector_E2>{};

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_e0_e1_k1_e2_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(I1, Number<E1>{}, Number<KPerBlock>{}, Number<E2>{}), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size = math::integer_least_multiple(
            a_e0_e1_k1_e2_block_desc.GetElementSpaceSize(), max_lds_align);

        return a_block_space_size * sizeof(FloatAB);
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CGridDesc_K_N_Ho_Wo& c_k_n_ho_wo_grid_desc)
    {
        const auto K  = c_k_n_ho_wo_grid_desc.GetLength(I0);
        const auto N  = c_k_n_ho_wo_grid_desc.GetLength(I1);
        const auto Ho = c_k_n_ho_wo_grid_desc.GetLength(I2);
        const auto Wo = c_k_n_ho_wo_grid_desc.GetLength(I3);

        const auto K0 = K / KPerBlock;
        const auto N0 = N / NPerBlock;
        const auto H0 = Ho / HoPerBlock;
        const auto W0 = Wo / WoPerBlock;

        const index_t grid_size = K0 * N0 * H0 * W0;

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainE0BlockLoop(const index_t E0)
    {
        const bool has_main_e0_block_loop = E0 > 1;

        return has_main_e0_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasMainE1BlockLoop()
    {
        const bool has_main_e1_block_loop = (E1 + E1PerBlock) / (2 * E1PerBlock) > 1;

        return has_main_e1_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasDoubleTailE1BlockLoop()
    {
        const bool has_double_tail_e1_block_loop = (E1 / E1PerBlock) % 2 == 0;

        return has_double_tail_e1_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeAE0E1K0K1E2GridDescriptor(const AGridDesc_E0_E1_K_E2& a_e0_e1_k_e2_grid_desc)
    {
        const auto E0 = a_e0_e1_k_e2_grid_desc.GetLength(I0);
        const auto K  = a_e0_e1_k_e2_grid_desc.GetLength(I2);

        const auto K1 = Number<KPerBlock>{};
        const auto K0 = K / K1;

        const auto a_e0_e1_k0_k1_e2_grid_desc = transform_tensor_descriptor(
            a_e0_e1_k_e2_grid_desc,
            make_tuple(make_pass_through_transform(E0),
                       make_pass_through_transform(E1),
                       make_unmerge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}));

        return a_e0_e1_k0_k1_e2_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCK0K1NH0H1H2W0W1W2GridDescriptor(const CGridDesc_K_N_Ho_Wo& c_k_n_ho_wo_grid_desc)
    {
        const auto K  = c_k_n_ho_wo_grid_desc.GetLength(I0);
        const auto N  = c_k_n_ho_wo_grid_desc.GetLength(I1);
        const auto Ho = c_k_n_ho_wo_grid_desc.GetLength(I2);
        const auto Wo = c_k_n_ho_wo_grid_desc.GetLength(I3);

        const auto K1 = Number<KPerBlock>{};
        const auto K0 = K / K1;

        const auto H2 = Number<HoPerThread>{};
        const auto H1 = Number<HoPerBlock / HoPerThread>{};
        const auto H0 = Ho / (H1 * H2);

        const auto W2 = Number<WoPerThread>{};
        const auto W1 = Number<WoPerBlock / WoPerThread>{};
        const auto W0 = Wo / (W1 * W2);

        const auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = transform_tensor_descriptor(
            c_k_n_ho_wo_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_unmerge_transform(make_tuple(H0, H1, H2)),
                       make_unmerge_transform(make_tuple(W0, W1, W2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}, Sequence<6, 7, 8>{}));

        return c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc;
    }

    __host__ __device__ static constexpr auto MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(
        const BGridDesc_E0_E1_N_Ho_Wo_E2& b_e0_e1_n_ho_wo_e2_grid_desc)
    {
        const auto E0 = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I0);
        // const auto E1 = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I1);
        const auto N  = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I2);
        const auto Ho = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I3);
        const auto Wo = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I4);
        // const auto E2 = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I5);

        const auto H2 = Number<HoPerThread>{};
        const auto H1 = Number<HoPerBlock / HoPerThread>{};
        const auto H0 = Ho / (H1 * H2);

        const auto W2 = Number<WoPerThread>{};
        const auto W1 = Number<WoPerBlock / WoPerThread>{};
        const auto W0 = Wo / (W1 * W2);

        const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
            transform_tensor_descriptor(b_e0_e1_n_ho_wo_e2_grid_desc,
                                        make_tuple(make_pass_through_transform(E0),
                                                   make_pass_through_transform(E1),
                                                   make_pass_through_transform(N),
                                                   make_unmerge_transform(make_tuple(H0, H1, H2)),
                                                   make_unmerge_transform(make_tuple(W0, W1, W2)),
                                                   make_pass_through_transform(E2)),
                                        make_tuple(Sequence<0>{},
                                                   Sequence<1>{},
                                                   Sequence<2>{},
                                                   Sequence<3>{},
                                                   Sequence<4>{},
                                                   Sequence<5>{}),
                                        make_tuple(Sequence<0>{},
                                                   Sequence<1>{},
                                                   Sequence<2>{},
                                                   Sequence<3, 4, 5>{},
                                                   Sequence<6, 7, 8>{},
                                                   Sequence<9>{}));

        return b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockIdToKNHoWoBlockClusterAdaptor(const CGridDesc_K_N_Ho_Wo& c_k_n_ho_wo_grid_desc)
    {
        const auto K  = c_k_n_ho_wo_grid_desc.GetLength(I0);
        const auto N  = c_k_n_ho_wo_grid_desc.GetLength(I1);
        const auto Ho = c_k_n_ho_wo_grid_desc.GetLength(I2);
        const auto Wo = c_k_n_ho_wo_grid_desc.GetLength(I3);

        const auto K0 = K / KPerBlock;
        const auto N0 = N / NPerBlock;
        const auto H0 = Ho / HoPerBlock;
        const auto W0 = Wo / WoPerBlock;

        const auto c_blockid_to_k_n_ho_wo_block_cluster_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(K0, N0, H0, W0))),
            make_tuple(Sequence<0, 1, 2, 3>{}),
            make_tuple(Sequence<0>{}));

        return c_blockid_to_k_n_ho_wo_block_cluster_adaptor;
    }

    using AGridDesc_E0_E1_K0_K1_E2 =
        decltype(MakeAE0E1K0K1E2GridDescriptor(AGridDesc_E0_E1_K_E2{}));
    using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 =
        decltype(MakeCK0K1NH0H1H2W0W1W2GridDescriptor(CGridDesc_K_N_Ho_Wo{}));
    using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
        decltype(MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(BGridDesc_E0_E1_N_Ho_Wo_E2{}));
    using CBlockIdToBlockClusterAdaptor_K_N_H_W =
        decltype(MakeCBlockIdToKNHoWoBlockClusterAdaptor(CGridDesc_K_N_Ho_Wo{}));

    template <bool HasMainE0BlockLoop>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_global,
        const FloatAB* __restrict__ p_b_global,
        FloatC* __restrict__ p_c_global,
        FloatAB* __restrict__ p_shared_block,
        const AGridDesc_E0_E1_K0_K1_E2& a_e0_e1_k0_k1_e2_grid_desc,
        const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2& b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
        const CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2& c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
        const CBlockIdToBlockClusterAdaptor_K_N_H_W& c_blockid_to_k_n_h_w_block_cluster_adaptor,
        integral_constant<bool, HasMainE0BlockLoop>)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_a_global, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_b_global, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_c_global, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

        constexpr auto HasMainE1BlockLoop       = CalculateHasMainE1BlockLoop();
        constexpr auto HasDoubleTailE1BlockLoop = CalculateHasDoubleTailE1BlockLoop();

        // const auto Ho = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetLength(I3);
        // const auto Wo = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetLength(I4);

        const auto c_k_n_h_w_block_cluster_idx =
            c_blockid_to_k_n_h_w_block_cluster_adaptor.CalculateBottomIndex(
                make_multi_index(get_block_1d_id()));

        const index_t k_block_work_id =
            __builtin_amdgcn_readfirstlane(c_k_n_h_w_block_cluster_idx[I0]);
        const index_t n_block_work_id =
            __builtin_amdgcn_readfirstlane(c_k_n_h_w_block_cluster_idx[I1]);
        const index_t ho_block_work_id =
            __builtin_amdgcn_readfirstlane(c_k_n_h_w_block_cluster_idx[I2]);
        const index_t wo_block_work_id =
            __builtin_amdgcn_readfirstlane(c_k_n_h_w_block_cluster_idx[I3]);

        constexpr auto max_lds_align = Number<ABlockTransferDstScalarPerVector_E2>{};

        constexpr auto a_e1_k1_e2_block_gemm_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E1PerBlock>{}, Number<KPerBlock>{}, Number<E2>{}), max_lds_align);

        constexpr auto b_e1_n_h_w_e2_block_gemm_desc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<E1PerBlock>{},
                                                           I1,
                                                           Number<HoPerBlock>{},
                                                           Number<WoPerBlock>{},
                                                           Number<E2>{}));

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<KPerThread>{}, I1, Number<HoPerThread>{}, Number<WoPerThread>{}));

        auto blockwise_gemm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v3<BlockSize,
                                                 FloatAB,
                                                 FloatAB,
                                                 FloatAcc,
                                                 decltype(a_e1_k1_e2_block_gemm_desc),
                                                 decltype(b_e1_n_h_w_e2_block_gemm_desc),
                                                 decltype(c_k1_n_h2_w2_thread_gemm_desc),
                                                 EPerThread,
                                                 K2>{};

        auto c_thread_mtx_index =
            blockwise_gemm.GetBeginOfCThreadDesc_K_N_Ho_Wo(get_thread_local_1d_id());

        const auto k_thread_id  = c_thread_mtx_index[I0];
        const auto ho_thread_id = c_thread_mtx_index[I2];
        const auto wo_thread_id = c_thread_mtx_index[I3];

        constexpr auto a_e0_e1_k0_k1_e2_block_copy_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<I1>{}, Number<E1>{}, I1, Number<KPerBlock>{}, Number<E2>{}),
            max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum_t::Set,
                                            Sequence<I1, E1, I1, KPerBlock, E2>,
                                            ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
                                            ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
                                            ABlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(a_e0_e1_k0_k1_e2_grid_desc),
                                            decltype(a_e0_e1_k0_k1_e2_block_copy_desc),
                                            ABlockTransferSrcAccessOrder,
                                            Sequence<0, 1, 2, 3, 4>, // ABlockTransferDstAccessOrder
                                            ABlockTransferSrcVectorDim,
                                            4, // ABlockTransferDstVectorDim
                                            ABlockTransferSrcScalarPerVector,
                                            ABlockTransferDstScalarPerVector_E2,
                                            1,
                                            1,
                                            AThreadTransferSrcResetCoordinateAfterRun,
                                            false>(a_e0_e1_k0_k1_e2_grid_desc,
                                                   make_multi_index(0, 0, k_block_work_id, 0, 0),
                                                   a_e0_e1_k0_k1_e2_block_copy_desc,
                                                   make_multi_index(0, 0, 0, 0, 0));

        constexpr auto a_block_slice_copy_step = make_multi_index(I1, 0, 0, 0, 0);

        constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<E1PerBlock>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<HoPerThread>{},
                                                           I1,
                                                           I1,
                                                           Number<WoPerThread>{},
                                                           Number<E2>{}));

        auto b_threadwise_transfer = ThreadwiseTensorSliceTransfer_v2<
            FloatAB,
            FloatAB,
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc),
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc),
            Sequence<I1, E1PerBlock, I1, I1, I1, HoPerThread, I1, I1, WoPerThread, E2>,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorDim,
            BBlockTransferSrcScalarPerVector,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                  make_multi_index(0,
                                   0,
                                   n_block_work_id,
                                   ho_block_work_id,
                                   ho_thread_id,
                                   0,
                                   wo_block_work_id,
                                   wo_thread_id,
                                   0,
                                   0));

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_shared_block, a_e0_e1_k0_k1_e2_block_copy_desc.GetElementSpaceSize());

        // register allocation for output
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAcc,
                     c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                     true>
            c_thread_buf;

#if 0
        // initialize output thread tensor
        ThreadwiseTensorSliceSet_v1<FloatAcc,
                                    decltype(c_k1_n_h2_w2_thread_gemm_desc),
                                    Sequence<KPerThread, NPerBlock, HoPerThread, WoPerThread>>{}
            .Run(c_k1_n_h2_w2_thread_gemm_desc,
                 make_tuple(I0, I0, I0, I0),
                 c_thread_buf,
                 FloatAcc{0});
#endif

        constexpr auto b_thread_slice_copy_step =
            make_multi_index(0, E1PerBlock, 0, 0, 0, 0, 0, 0, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_e0_e1_k_e2_global_step_hacks                   = AGlobalStepHacks{};
        constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks = BGlobalStepHacks{};

        // double regsiter buffer for b
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAB,
                     b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc.GetElementSpaceSize(),
                     true>
            b_thread_even_buf, b_thread_odd_buf;

        if constexpr(HasMainE0BlockLoop)
        {
            const auto E0 = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetLength(I0);

            index_t e0_block_data_begin = 0;

            do
            {
                // LDS double buffer: preload data
                {
                    a_blockwise_copy.RunRead(
                        a_e0_e1_k0_k1_e2_grid_desc, a_global_buf, a_e0_e1_k_e2_global_step_hacks);

                    b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              b_global_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              b_thread_even_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks);

                    a_blockwise_copy.RunWrite(a_e0_e1_k0_k1_e2_block_copy_desc, a_block_buf);
                }

                __syncthreads();

                if constexpr(HasMainE1BlockLoop)
                {
                    index_t e1_block_data_begin = 0;

                    // LDS double buffer: main body
                    // use Do-While loop instead of For loop to simplify control flow
                    do
                    {
                        // even iteration
                        b_threadwise_transfer.MoveSrcSliceWindow(
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                            b_thread_slice_copy_step,
                            BGlobalMoveSliceWindowStepHacks{});

                        b_threadwise_transfer.Run(
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                            b_global_buf,
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                            make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                            b_thread_odd_buf,
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks);

                        // LDS double buffer: GEMM on current data
                        blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                        blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                        b_threadwise_transfer.MoveSrcSliceWindow(
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                            b_thread_slice_copy_step,
                            BGlobalMoveSliceWindowStepHacks{});

                        b_threadwise_transfer.Run(
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                            b_global_buf,
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                            make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                            b_thread_even_buf,
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks);

                        // LDS double buffer: GEMM on current data
                        blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);

                        blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                        e1_block_data_begin += 2 * E1PerBlock;

                    } while(e1_block_data_begin < E1 - 2 * E1PerBlock);
                }

                // LDS double buffer: tail
                if constexpr(HasDoubleTailE1BlockLoop) // if has 2 iteration left
                {
                    b_threadwise_transfer.MoveSrcSliceWindow(
                        b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                        b_thread_slice_copy_step,
                        BGlobalMoveSliceWindowStepHacks{});

                    b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              b_global_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              b_thread_odd_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks);

                    // LDS double buffer: GEMM on 2nd-last data
                    blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                    blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                    // LDS double buffer: GEMM on last data
                    blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);
                }
                else // if has 1 iteration left
                {
                    // LDS double buffer: GEMM on last data
                    blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);
                }

                a_blockwise_copy.MoveSrcSliceWindow(a_e0_e1_k0_k1_e2_grid_desc,
                                                    a_block_slice_copy_step,
                                                    AGlobalMoveSliceWindowStepHacks{});

                blockwise_gemm.MoveABlockSliceWindow(make_tuple(-(E1 - E1PerBlock), 0, 0));

                b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                                         b_thread_slice_copy_step,
                                                         BGlobalMoveSliceWindowStepHacks{});

                e0_block_data_begin += 1;

            } while(e0_block_data_begin < E0);
        }
        else
        {
            // LDS double buffer: preload data
            {
                a_blockwise_copy.RunRead(
                    a_e0_e1_k0_k1_e2_grid_desc, a_global_buf, a_e0_e1_k_e2_global_step_hacks);

                b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                          b_global_buf,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                          b_thread_even_buf,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks);

                a_blockwise_copy.RunWrite(a_e0_e1_k0_k1_e2_block_copy_desc, a_block_buf);
            }

            __syncthreads();

            if constexpr(HasMainE1BlockLoop)
            {
                index_t e1_block_data_begin = 0;

                // LDS double buffer: main body
                // use Do-While loop instead of For loop to simplify control flow
                do
                {
                    // even iteration
                    b_threadwise_transfer.MoveSrcSliceWindow(
                        b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                        b_thread_slice_copy_step,
                        BGlobalMoveSliceWindowStepHacks{});

                    b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              b_global_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              b_thread_odd_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks);

                    // LDS double buffer: GEMM on current data
                    blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                    blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                    b_threadwise_transfer.MoveSrcSliceWindow(
                        b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                        b_thread_slice_copy_step,
                        BGlobalMoveSliceWindowStepHacks{});

                    b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              b_global_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              b_thread_even_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks);

                    // LDS double buffer: GEMM on current data
                    blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);

                    blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                    e1_block_data_begin += 2 * E1PerBlock;

                } while(e1_block_data_begin < E1 - 2 * E1PerBlock);
            }

            // LDS double buffer: tail
            if constexpr(HasDoubleTailE1BlockLoop) // if has 2 iteration left
            {
                b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                                         b_thread_slice_copy_step,
                                                         BGlobalMoveSliceWindowStepHacks{});

                b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                          b_global_buf,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                          b_thread_odd_buf,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks);

                // LDS double buffer: GEMM on 2nd-last data
                blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);
            }
            else // if has 1 iteration left
            {
                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);
            }
        }

        // activ
        {
            static_for<0, c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(), 1>{}([&](auto i) {
                if constexpr(activ_type == 1)
                {
                    c_thread_buf(i) = c_thread_buf[i] >= 0 ? c_thread_buf[i] : 0.0;
                }
                else if constexpr(activ_type == 2)
                {
                    FloatAcc x = 1.0 + exp(-c_thread_buf[i]);

                    asm volatile("\n \
                        v_rcp_f32 %0, %1 \n"
                                 : "=v"(x)
                                 : "0"(x));

                    c_thread_buf(i) = x;
                }
            });
        }

        // output: register to global memory
        {
            // hack to control index calculation when iterating over c_k_n_h0_h1_h2_w0_w1_w2_global
            // tensor
            constexpr auto c_k_n_h0_h1_h2_w0_w1_w2_global_tensor_step_hacks = CGlobalStepHacks{};

            constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc =
                make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                               Number<KPerThread>{},
                                                               I1,
                                                               I1,
                                                               I1,
                                                               Number<HoPerThread>{},
                                                               I1,
                                                               I1,
                                                               Number<WoPerThread>{}));

            const index_t k_thread_data_on_global = k_thread_id * KPerThread;

            ThreadwiseTensorSliceTransfer_v1r3<
                FloatAcc,
                FloatC,
                decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc),
                decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc),
                Sequence<I1, KPerThread, I1, I1, I1, HoPerThread, I1, I1, WoPerThread>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                true>(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                      make_multi_index(k_block_work_id,
                                       k_thread_data_on_global,
                                       n_block_work_id,
                                       ho_block_work_id,
                                       ho_thread_id,
                                       0,
                                       wo_block_work_id,
                                       wo_thread_id,
                                       0))
                .Run(c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc,
                     make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                     c_thread_buf,
                     c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                     c_global_buf,
                     c_k_n_h0_h1_h2_w0_w1_w2_global_tensor_step_hacks);
        }
    }
};

} // namespace ck
#endif
