#ifndef CK_GRIDWISE_GEMM_XDLOPS_V2R3_SPLITM_HPP
#define CK_GRIDWISE_GEMM_XDLOPS_V2R3_SPLITM_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_gemm_xdlops.hpp"
#include "blockwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_set.hpp"

namespace ck {

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_B_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_xdlops_v2r3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const index_t a_batch_stride,
            const index_t c_batch_stride,
            const AGridDesc_B_K0_M_K1 a_grid_desc_b_k0_m_k1,
            const BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1,
            const CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2,
            const Block2CTileMap block_2_ctile_map)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_c_grid,
                                                  a_batch_stride,
                                                  c_batch_stride,
                                                  p_shared_block,
                                                  a_grid_desc_b_k0_m_k1,
                                                  b_grid_desc_k0_n_k1,
                                                  c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2,
                                                  block_2_ctile_map);
}
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_B_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2,
          typename Block2CTileMap>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_xdlops_v2r3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const index_t a_batch_stride,
            const index_t c_batch_stride,
            const void CONSTANT* p_a_grid_desc_b_k0_m_k1,
            const void CONSTANT* p_b_grid_desc_k0_n_k1,
            const void CONSTANT* p_c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
            const void CONSTANT* p_block_2_ctile_map)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    const auto a_grid_desc_k0_m_k1 = *reinterpret_cast<const AK0MK1GridDesc*>(
        cast_pointer_to_generic_address_space(p_a_k0_m_k1_grid_desc));
    const auto b_grid_desc_k0_n_k1 = *reinterpret_cast<const BGridDesc_K0_N_K1*>(
        cast_pointer_to_generic_address_space(p_b_grid_desc_k0_n_k1));
    const auto c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
        *reinterpret_cast<const CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2*>(
            cast_pointer_to_generic_address_space(p_c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc));
    const auto block_2_ctile_map = *reinterpret_cast<const Block2CTileMap*>(
        cast_pointer_to_generic_address_space(p_block_2_ctile_map));

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      a_batch_stride,
                      c_batch_stride,
                      p_shared_block,
                      a_grid_desc_b_k0_m_k1,
                      b_grid_desc_k0_n_k1,
                      c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2,
                      block_2_ctile_map);
}
#endif

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename AGridDesc_B_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDesc_B_M_N,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t K1Value,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadSliceLengths_K0_M_K1,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K0_N_K1,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridStepHacks,
          typename BGridStepHacks,
          typename CGridStepHacks,
          typename AGridMoveSliceWindowStepHacks,
          typename BGridMoveSliceWindowStepHacks,
          bool CAccessOrderMRepeatNRepeat,
          bool ABlockLdsExtraM,
          bool BBlockLdsExtraN>
struct GridwiseBatchedGemm_sk0mk1_k0nk1_smn_xdlops_v2r3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};
    static constexpr auto I8 = Number<8>{};

    // K1 should be Number<...>
    static constexpr auto K1 = Number<K1Value>{};

    // A matrix in LDS memory, dst of blockwise copy
    __host__ __device__ static constexpr auto MakeABlockDesc_K0_M_K1()
    {
        constexpr auto max_lds_align = K1;
        if constexpr(ABlockLdsExtraM)
        {
            return make_naive_tensor_descriptor(
                make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
        }
        else
        {
            return make_naive_tensor_descriptor_aligned(
                make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
        }
    }

    // B matrix in LDS memory, dst of blockwise copy
    __host__ __device__ static constexpr auto MakeBBlockDesc_K0_N_K1()
    {
        constexpr auto max_lds_align = K1;

        if constexpr(BBlockLdsExtraN)
        {
            return make_naive_tensor_descriptor(
                make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
        }
        else
        {
            return make_naive_tensor_descriptor_aligned(
                make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
        }
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto a_block_desc_k0_m_k1 = MakeABlockDesc_K0_M_K1();
        constexpr auto b_block_desc_k0_n_k1 = MakeBBlockDesc_K0_N_K1();

        constexpr auto max_lds_align = K1;
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_block_desc_k0_n_k1.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size + b_block_space_size) * sizeof(FloatAB);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_B_K0_M_K1& a_grid_desc_b_k0_m_k1,
                  const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
                  const CGridDesc_B_M_N& c_grid_desc_b_m_n,
                  index_t M01,
                  index_t N01)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXDL * MRepeat) == 0) &&
                          (NPerBlock % (NRepeat * NPerXDL)) == 0,
                      "Invalid tuning param!");

        if(a_grid_desc_b_k0_m_k1.GetLength(I0) != c_grid_desc_b_m_n.GetLength(I0))
            return false;

        const auto M  = a_grid_desc_b_k0_m_k1.GetLength(I2);
        const auto N  = b_grid_desc_k0_n_k1.GetLength(I1);
        const auto K0 = a_grid_desc_b_k0_m_k1.GetLength(I1);

        if(!(M == c_grid_desc_b_m_n.GetLength(I1) && N == c_grid_desc_b_m_n.GetLength(I2) &&
             K0 == b_grid_desc_k0_n_k1.GetLength(I0) && K1 == a_grid_desc_b_k0_m_k1.GetLength(I3) &&
             K1 == b_grid_desc_k0_n_k1.GetLength(I2)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // check M01, N01
        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        if(!(M0 % M01 == 0 && N0 % N01 == 0))
            return false;

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CGridDesc_B_M_N& c_grid_desc_b_m_n)
    {
        const auto M = c_grid_desc_b_m_n.GetLength(I1);
        const auto N = c_grid_desc_b_m_n.GetLength(I2);

        const index_t grid_size = (M / MPerBlock) * (N / NPerBlock);

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainK0BlockLoop(index_t K0)
    {
        const bool has_main_k0_block_loop = (K0 / K0PerBlock) > 1;

        return has_main_k0_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_B_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_B_M_N& c_grid_desc_b_m_n)
    {
        using BlockwiseGemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(MakeABlockDesc_K0_M_K1()),
                                                                decltype(MakeBBlockDesc_K0_N_K1()),
                                                                MPerXDL,
                                                                NPerXDL,
                                                                MRepeat,
                                                                NRepeat,
                                                                K1>;

        return BlockwiseGemm::MakeCGridDescriptor_B_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_b_m_n);
    }

    __host__ __device__ static constexpr auto
    MakeAGridDescriptor_K0_M_K1(const AGridDesc_B_K0_M_K1& a_grid_desc_b_k0_m_k1, const int bb)
    {
        const auto K0 = a_grid_desc_b_k0_m_k1.GetLength(I1);
        const auto M  = a_grid_desc_b_k0_m_k1.GetLength(I2);

        const auto a_grid_desc_k0_m_k1 =
            transform_tensor_descriptor(a_grid_desc_b_k0_m_k1,
                                        make_tuple(make_freeze_transform(bb),
                                                   make_pass_through_transform(K0),
                                                   make_pass_through_transform(M),
                                                   make_pass_through_transform(K1)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        return a_grid_desc_k0_m_k1;
    }

    __host__ __device__ static constexpr auto 
    MakeCGridDesc_M_N(const CGridDesc_B_M_N& c_grid_desc_b_m_n,
                    const index_t bb)
    {
        const auto M = c_grid_desc_b_m_n.GetLength(I1);
        const auto N = c_grid_desc_b_m_n.GetLength(I2);

        const auto c_m_n_grid_desc = transform_tensor_descriptor(
            c_grid_desc_b_m_n,
            make_tuple(make_freeze_transform(bb), 
                       make_pass_through_transform(M),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return c_m_n_grid_desc;
    }

    using CMNGridDesc = decltype(MakeCGridDesc_M_N(CGridDesc_B_M_N{}, 0));

    using CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(MakeCGridDescriptor_B_M0_N0_M1_N1_M2_M3_M4_N2(CGridDesc_B_M_N{}));

    __host__ __device__ static constexpr auto 
    MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(
        const CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2& c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2, const int bb)
    {
        const auto M0 = c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        const auto N0 = c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
        const auto M1 = c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        const auto N1 = c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
        const auto M2 = c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        const auto M3 = c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        const auto M4 = c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);
        const auto N2 = c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I8);

        const auto c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            transform_tensor_descriptor(c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2,
                                        make_tuple(make_freeze_transform(bb),
                                                   make_pass_through_transform(M0),
                                                   make_pass_through_transform(N0),
                                                   make_pass_through_transform(M1),
                                                   make_pass_through_transform(N1),
                                                   make_pass_through_transform(M2),
                                                   make_pass_through_transform(M3),
                                                   make_pass_through_transform(M4),
                                                   make_pass_through_transform(N2)),
                                        make_tuple(Sequence<0>{},
                                                   Sequence<1>{},
                                                   Sequence<2>{},
                                                   Sequence<3>{},
                                                   Sequence<4>{},
                                                   Sequence<5>{},
                                                   Sequence<6>{},
                                                   Sequence<7>{},
                                                   Sequence<8>{}),
                                        make_tuple(Sequence<0>{},
                                                   Sequence<1>{},
                                                   Sequence<2>{},
                                                   Sequence<3>{},
                                                   Sequence<4>{},
                                                   Sequence<5>{},
                                                   Sequence<6>{},
                                                   Sequence<7>{}));


        return c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeBlock2CTileMap(const CGridDesc_B_M_N& c_grid_desc_b_m_n, index_t M01, index_t N01)
    {
        const auto M = c_grid_desc_b_m_n.GetLength(I1);
        const auto N = c_grid_desc_b_m_n.GetLength(I2);

        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        const auto M00 = M0 / M01;
        const auto N00 = N0 / N01;

        const auto m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}));

        const auto c_blockid_to_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

        const auto c_blockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  c_blockid_to_m00_m01_n00_n01_block_cluster_adaptor);

        return c_blockid_to_m0_n0_block_cluster_adaptor;
    }

    using Block2CTileMap = decltype(MakeBlock2CTileMap(CGridDesc_B_M_N{}, 1, 1));

    template <bool HasMainKBlockLoop>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid_,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid_,
        const index_t a_batch_stride,
        const index_t c_batch_stride,
        FloatAB* __restrict__ p_shared_block,
        const AGridDesc_B_K0_M_K1& a_grid_desc_b_k0_m_k1,
        const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
        const CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2& c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2,
        const Block2CTileMap& block_2_ctile_map)
    {
        const auto NumBatches  = a_grid_desc_b_k0_m_k1.GetLength(I0);
        const auto K0 = a_grid_desc_b_k0_m_k1.GetLength(I1);

        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_b_grid, b_grid_desc_k0_n_k1.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align        = K1;
        constexpr auto a_block_desc_k0_m_k1 = MakeABlockDesc_K0_M_K1();
        constexpr auto b_block_desc_k0_n_k1 = MakeBBlockDesc_K0_N_K1();

        // these 2 descriptors are reused inside the loop over batches
        const auto a_grid_desc_k0_m_k1 = MakeAGridDescriptor_K0_M_K1(a_grid_desc_b_k0_m_k1, 0);
        const auto c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2, 0);
        //
        // // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum_t::Set,
                                            Sequence<K0PerBlock, MPerBlock, K1>,
                                            ABlockTransferThreadSliceLengths_K0_M_K1,
                                            ABlockTransferThreadClusterLengths_K0_M_K1,
                                            ABlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(a_grid_desc_k0_m_k1),
                                            decltype(a_block_desc_k0_m_k1),
                                            ABlockTransferSrcAccessOrder,
                                            Sequence<1, 0, 2>,
                                            ABlockTransferSrcVectorDim,
                                            2,
                                            ABlockTransferSrcScalarPerVector,
                                            ABlockTransferDstScalarPerVector_K1,
                                            1,
                                            1,
                                            AThreadTransferSrcResetCoordinateAfterRun,
                                            true>(a_grid_desc_k0_m_k1,
                                                  make_multi_index(0, m_block_data_idx_on_grid, 0), 
                                                  a_block_desc_k0_m_k1, 
                                                  make_multi_index(0, 0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum_t::Set,
                                            Sequence<K0PerBlock, NPerBlock, K1>,
                                            BBlockTransferThreadSliceLengths_K0_N_K1,
                                            BBlockTransferThreadClusterLengths_K0_N_K1,
                                            BBlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(b_grid_desc_k0_n_k1),
                                            decltype(b_block_desc_k0_n_k1),
                                            BBlockTransferSrcAccessOrder,
                                            Sequence<1, 0, 2>,
                                            BBlockTransferSrcVectorDim,
                                            2,
                                            BBlockTransferSrcScalarPerVector,
                                            BBlockTransferDstScalarPerVector_K1,
                                            1,
                                            1,
                                            BThreadTransferSrcResetCoordinateAfterRun,
                                            true>(b_grid_desc_k0_n_k1,
                                                  make_multi_index(0, n_block_data_idx_on_grid, 0), 
                                                  b_block_desc_k0_n_k1, 
                                                  make_multi_index(0, 0, 0));


        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check

        auto blockwise_gemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_block_desc_k0_m_k1),
                                                                decltype(b_block_desc_k0_n_k1),
                                                                MPerXDL,
                                                                NPerXDL,
                                                                MRepeat,
                                                                NRepeat,
                                                                K1>{};

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();
        constexpr auto c_thread_desc = blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block = p_shared_block;
        FloatAB* p_b_block = p_shared_block + a_block_space_size;

        constexpr auto a_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_k0_m_k1_grid_step_hacks = AGridStepHacks{};
        constexpr auto b_k0_n_k1_grid_step_hacks = BGridStepHacks{};

        // hack to control index calculation when move slice window for A and B matrix for
        // threadwise copy
        constexpr auto a_k0_m_k1_grid_move_slice_window_step_hack = AGridMoveSliceWindowStepHacks{};
        constexpr auto b_k0_n_k1_grid_move_slice_window_step_hack = BGridMoveSliceWindowStepHacks{};

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_a_block, a_block_desc_k0_m_k1.GetElementSpaceSize());
        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_b_block, b_block_desc_k0_n_k1.GetElementSpaceSize());


        // variables for store from register to global memory
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
        constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
        constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
        constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc =
            make_naive_tensor_descriptor_packed(make_tuple(
                Number<M0>{}, Number<N0>{}, I1, I1, Number<M2>{}, I1, Number<M4>{}, I1));
        // calculate origin of thread output tensor on global memory
        //     blockwise GEMM c matrix starting index
        const auto c_thread_mtx_on_block =
            blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

        const index_t m_thread_data_on_grid =
            m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

        const index_t n_thread_data_on_grid =
            n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

        constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_grid_tensor_step_hacks = CGridStepHacks{};

        const auto m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto m_thread_data_on_grid_idx =
            m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                make_multi_index(m_thread_data_on_grid));

        const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

        const auto n_thread_data_on_grid_idx =
            n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_on_grid));

        auto c_thread_copy = ThreadwiseTensorSliceTransfer_v1r3<
            FloatAcc,
            FloatC,
            decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc),
            decltype(c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2),
            Sequence<M0, N0, I1, I1, M2, I1, M4, I1>,
            CThreadTransferSrcDstAccessOrder,
            CThreadTransferSrcDstVectorDim,
            CThreadTransferDstScalarPerVector,
            CGlobalMemoryDataOperation,
            1,
            true>{c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                  make_multi_index(m_thread_data_on_grid_idx[I0],
                                   n_thread_data_on_grid_idx[I0],
                                   m_thread_data_on_grid_idx[I1],
                                   n_thread_data_on_grid_idx[I1],
                                   m_thread_data_on_grid_idx[I2],
                                   m_thread_data_on_grid_idx[I3],
                                   m_thread_data_on_grid_idx[I4],
                                   n_thread_data_on_grid_idx[I2])};


        const FloatAB* p_a_grid = p_a_grid_;
        FloatAB* p_c_grid       = p_c_grid_;

        // const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        //     p_a_grid, a_grid_desc_k0_m_k1.GetElementSpaceSize());
        //
        // auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        //     p_c_grid, c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetElementSpaceSize());

        for (int bb = 0; bb < NumBatches; ++bb)
        // int bb = 0;
        // do
        {
            const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
                p_a_grid, a_grid_desc_k0_m_k1.GetElementSpaceSize());

            auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
                p_c_grid, c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetElementSpaceSize());

            // initialize c_thread_buf
            static_for<0, c_thread_desc.GetElementSpaceSize(), 1>{}([&](auto ii) {
                c_thread_buf(ii) = static_cast<FloatAcc>(0);
            });

            // preload data into LDS
            {
                a_blockwise_copy.RunRead(
                    a_grid_desc_k0_m_k1, a_grid_buf, a_k0_m_k1_grid_step_hacks);
                b_blockwise_copy.RunRead(
                    b_grid_desc_k0_n_k1, b_grid_buf, b_k0_n_k1_grid_step_hacks);

                a_blockwise_copy.RunWrite(a_block_desc_k0_m_k1, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc_k0_n_k1, b_block_buf);
            }

            // main body
            index_t k_block_data_begin = 0;

            if constexpr(HasMainKBlockLoop)
            {
                do
                {
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_k0_m_k1,
                                                        a_block_slice_copy_step,
                                                        a_k0_m_k1_grid_move_slice_window_step_hack);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_k0_n_k1,
                                                        b_block_slice_copy_step,
                                                        b_k0_n_k1_grid_move_slice_window_step_hack);

                    a_blockwise_copy.RunRead(
                        a_grid_desc_k0_m_k1, a_grid_buf, a_k0_m_k1_grid_step_hacks);

                    block_sync_lds();

                    b_blockwise_copy.RunRead(
                        b_grid_desc_k0_n_k1, b_grid_buf, b_k0_n_k1_grid_step_hacks);

                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                    block_sync_lds();

                    a_blockwise_copy.RunWrite(a_block_desc_k0_m_k1, a_block_buf);
                    b_blockwise_copy.RunWrite(b_block_desc_k0_n_k1, b_block_buf);

                    k_block_data_begin += K0PerBlock;
                } while(k_block_data_begin < (K0 - K0PerBlock));
            }

            // tail
            {
                block_sync_lds();

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
            }

            // output: register to global memory
            {

                c_thread_copy.Run(c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc,
                                  make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                  c_thread_buf,
                                  c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                  c_grid_buf,
                                  c_m0_n0_m1_n1_m2_m3_m4_n2_grid_tensor_step_hacks);
            }

            p_a_grid += a_batch_stride;
            p_c_grid += c_batch_stride;

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_k0_m_k1,
                                                make_multi_index(K0PerBlock-K0, 0, 0),
                                                a_k0_m_k1_grid_move_slice_window_step_hack);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_k0_n_k1,
                                                make_multi_index(K0PerBlock-K0, 0, 0),
                                                b_k0_n_k1_grid_move_slice_window_step_hack);
        }
        //     ++bb;
        // } while (bb < NumBatches);
    }
}; // struct GridwiseBatchedGemm_sk0mk1_k0nk1_smn_xdlops_v2r3

} // namespace ck
#endif
