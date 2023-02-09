// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_wmma(
            const FloatA* __restrict__ p_a_grid,
            const FloatB* __restrict__ p_b0_grid,
            FloatC* __restrict__ p_c_grid,
            const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
            const BGridDesc_K0_N_K1 b0_grid_desc_k0_l_k1,
            const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            // const
            // CGridDescriptor_MBlockxRepeat_MWave_MSubGroup_MAccVgprs_NBlockxRepeat_NWave_NThreadPerSubGroup
            //     c_grid_desc_mblockxrepeat_mwave_msubgroup_maccvgprs_nblockxrepeat_nwave_nthreadpersubgroup,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b0_grid,
                                                  p_c_grid,
                                                  p_shared,
                                                  a_grid_desc_k0_m_k1,
                                                  b0_grid_desc_k0_l_k1,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b0_grid;
    ignore = p_c_grid;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b0_grid_desc_k0_l_k1;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx1100__))
}

// Gemm0: A [M x K] x B0 [K x L] = Acc [M x L]
// Gemm1: Acc [M x L] x B1 [L x N] = C [M x N]
template <index_t BlockSize,
          typename FloatA,
          typename FloatB0,
          typename FloatB1,
          typename FloatAcc,
          typename FloatCShuffle,
          typename FloatC,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_K0_M_K1,
          typename B0GridDesc_K0_L_K1,
          typename B1GridDesc_L0_N_L1,
          typename CGridDesc_M_N,
          index_t Gemm0MPerBlock,
          index_t Gemm0LPerBlock,
          index_t Gemm0K0PerBlock,
          index_t Gemm0K1Value,
          index_t Gemm0MPerWmma,
          index_t Gemm0LPerWmma,
          index_t Gemm0MRepeat,
          index_t Gemm0LRepeat,
          index_t Gemm1MPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1L0PerBlock,
          index_t Gemm1L1Value,
          index_t Gemm1MPerWmma,
          index_t Gemm1NPerWmma,
          index_t Gemm1MRepeat,
          index_t Gemm1NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool ABlockLdsExtraM,
          typename B0BlockTransferThreadClusterLengths_K0_L_K1,
          typename B0BlockTransferThreadClusterArrangeOrder,
          typename B0BlockTransferSrcAccessOrder,
          index_t B0BlockTransferSrcVectorDim,
          index_t B0BlockTransferSrcScalarPerVector,
          index_t B0BlockTransferDstScalarPerVector_K1,
          bool B0ThreadTransferSrcResetCoordinateAfterRun,
          bool B0BlockLdsExtraN,
          typename B1BlockTransferThreadClusterLengths_L0_N_L1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_L1,
          bool B1ThreadTransferSrcResetCoordinateAfterRun,
          bool B1BlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          index_t NumGemmKPrefetchStage = 1,
          LoopScheduler LoopSched       = make_default_loop_scheduler(),
          PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseBatchedGemmSoftmaxGemm_Wmma_CShuffle
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto K1 = Number<Gemm0K1Value>{};
    static constexpr auto N1 = Number<Gemm1N1Value>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<decltype(
        GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    __host__ __device__ static constexpr auto GetABlockDescriptor_K0PerBlock_MPerBlock_K1()
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k0perblock_mperblock_k1 = [&]() {
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
        }();

        return a_block_desc_k0perblock_mperblock_k1;
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_K0PerBlock_NPerBlock_K1()
    {
        constexpr auto max_lds_align = K1;

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_k0perblock_nperblock_k1 = [&]() {
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
        }();

        return b_block_desc_k0perblock_nperblock_k1;
    }

    __host__ __device__ static constexpr auto
    // *Caution Here repeat is shuffle repeat
    GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
    {
        constexpr index_t MWave = MPerBlock / (MRepeat * MPerWmma);
        constexpr index_t NWave = NPerBlock / (NRepeat * NPerWmma);

        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMRepeatPerShuffle * MWave * MPerWmma>{},
                           I1,
                           Number<CShuffleNRepeatPerShuffle * NWave * NPerWmma>{}));

        return c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_k0perblock_mperblock_k1 =
            GetABlockDescriptor_K0PerBlock_MPerBlock_K1();

        constexpr auto b_block_desc_k0perblock_nperblock_k1 =
            GetBBlockDescriptor_K0PerBlock_NPerBlock_K1();

        constexpr auto max_lds_align = K1;

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_k0perblock_mperblock_k1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_k0perblock_nperblock_k1.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size_aligned * sizeof(FloatA) +
                b_block_space_size_aligned * sizeof(FloatB));
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                  const BGridDesc_K0_N_K1& b0_grid_desc_k0_l_k1,
                  const CGridDesc_M_N& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerWmma * MRepeat) == 0) &&
                          (NPerBlock % (NRepeat * NPerWmma)) == 0,
                      "Invalid tuning param!");

        const auto M  = a_grid_desc_k0_m_k1.GetLength(I1);
        const auto N  = b0_grid_desc_k0_l_k1.GetLength(I1);
        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1) &&
             K0 == b0_grid_desc_k0_l_k1.GetLength(I0) && K1 == a_grid_desc_k0_m_k1.GetLength(I2) &&
             K1 == b0_grid_desc_k0_l_k1.GetLength(I2)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // check gridwise gemm pipeline
        const auto num_k_loop = K0 / K0PerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / (K0PerBlock * K1);

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap(
        const CGridDesc_M_N& c_grid_desc_m_n, index_t /* M01 */, index_t /* N01 */)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    using CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}))>;
    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}, 1, 1))>;

    template <bool HasMainKBlockLoop, typename C0MatrixMask, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const FloatA* __restrict__ p_a_grid,
                               const FloatB0* __restrict__ p_b0_grid,
                               const FloatB1* __restrict__ p_b1_grid,
                               FloatC* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                               const B0GridDesc_K0_L_K1& b0_grid_desc_k0_l_k1,
                               const B1GridDesc_L0_N_L1& b1_grid_desc_l0_n_l1,
                               const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const AElementwiseOperation& a_element_op,
                               const B0ElementwiseOperation& b0_element_op,
                               const AccElementwiseOperation& acc_element_op,
                               const B1ElementwiseOperation& b1_element_op,
                               const CElementwiseOperation& c_element_op,
                               const C0MatrixMask& c0_matrix_mask,
                               const Block2CTileMap& block_2_ctile_map)
    {
        // clang-format off
/*******************************************************************************/
// Memory buffer zone.
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_k0_m_k1.GetElementSpaceSize());
        const auto b0_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b0_grid, b0_grid_desc_k0_l_k1.GetElementSpaceSize());
        const auto b1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b1_grid, b1_grid_desc_l0_n_l1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

/*******************************************************************************/
// BlockIdx.x -> [BlockId.m, BlockId.n]
        const auto block_work_idx = block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        { return; }

        // Store BlockId into SGPR
        const index_t m_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

/*******************************************************************************/
// set up Gemm0
/*******************************************************************************/

/*******************************************************************************/
// BlockLevel, A/B Matrix ThreadMapping in LDS, As Destinaion of BlockWise_Copy
        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);
        constexpr auto max_lds_align = K1;
        constexpr auto a_block_desc_k0perblock_mperblock_k1 = GetABlockDescriptor_K0PerBlock_MPerBlock_K1();
        constexpr auto b_block_desc_k0perblock_nperblock_k1 = GetBBlockDescriptor_K0PerBlock_NPerBlock_K1();

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<        ThisThreadBlock,
/* typename SrcElementwiseOperation,              */    AElementwiseOperation,
/* typename DstElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
/* InMemoryDataOperationEnum DstInMemOp,          */    InMemoryDataOperationEnum::Set,
/* typename BlockSliceLengths,                    */    Sequence<K0PerBlock, MPerBlock, K1>,
/* typename ThreadClusterLengths,                 */    ABlockTransferThreadClusterLengths_K0_M_K1,
/* typename ThreadClusterArrangeOrder,            */    ABlockTransferThreadClusterArrangeOrder,
/* typename SrcData,                              */    FloatA,
/* typename DstData,                              */    FloatA,
/* typename SrcDesc,                              */    decltype(a_grid_desc_k0_m_k1),
/* typename DstDesc,                              */    decltype(a_block_desc_k0perblock_mperblock_k1),
/* typename SrcDimAccessOrder,                    */    ABlockTransferSrcAccessOrder,
/* typename DstDimAccessOrder,                    */    Sequence<0, 1, 2>,
/* index_t SrcVectorDim,                          */    ABlockTransferSrcVectorDim,
/* index_t DstVectorDim,                          */    2,
/* index_t SrcScalarPerVector,                    */    ABlockTransferSrcScalarPerVector,
/* index_t DstScalarPerVector,                    */    ABlockTransferDstScalarPerVector_K1,
/* index_t SrcScalarStrideInVector,               */    1,
/* index_t DstScalarStrideInVector,               */    1,
/* bool ThreadTransferSrcResetCoordinateAfterRun, */    AThreadTransferSrcResetCoordinateAfterRun,
/* bool ThreadTransferDstResetCoordinateAfterRun, */    true>(
                a_grid_desc_k0_m_k1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_k0perblock_mperblock_k1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<K0PerBlock, NPerBlock, K1>,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                FloatB,
                                                FloatB,
                                                decltype(b0_grid_desc_k0_l_k1),
                                                decltype(b_block_desc_k0perblock_nperblock_k1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<0, 1, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                true>(
                b0_grid_desc_k0_l_k1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_k0perblock_nperblock_k1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

/*******************************************************************************/
        // Gemm0
        constexpr auto WmmaK = 16;
        constexpr auto KPack = math::integer_least_multiple(K1, WmmaK);

        auto blockwise_gemm0 =
            BlockwiseGemmWMMA_k0mk1_k0nk1_m0m1m2n0n1n2m3_CShuffle_FIFO<BlockSize,
                                                         FloatA,
                                                         FloatB,
                                                         FloatAcc,
                                                         decltype(a_block_desc_k0perblock_mperblock_k1),
                                                         decltype(b_block_desc_k0perblock_nperblock_k1),
                                                         MPerWmma,
                                                         NPerWmma,
                                                         MRepeat,
                                                         NRepeat,
                                                         KPack>{};

        // Prepare Register for A*B0 matrix
        auto acc_thread_buf = blockwise_gemm.GetCThreadBuffer();

/*******************************************************************************/
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(a_block_desc_k0perblock_mperblock_k1.GetElementSpaceSize(), max_lds_align);
        // LDS allocation for A and B: be careful of alignment
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(static_cast<FloatA*>(p_shared), a_block_desc_k0perblock_mperblock_k1.GetElementSpaceSize());
        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(static_cast<FloatB*>(p_shared) + a_block_space_size_aligned, b_block_desc_k0perblock_nperblock_k1.GetElementSpaceSize());
        
        // Shift Per SUB_K
        constexpr auto a_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);
        const auto a_block_reset_copy_step = make_multi_index(-a_grid_desc_k0_m_k1.GetLength(I0), 0, 0);
        const auto b_block_reset_copy_step = make_multi_index(-b0_grid_desc_k0_l_k1.GetLength(I0), LPerBlock, 0);
/*******************************************************************************/
// softmax
/*******************************************************************************/
        auto workspace_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(static_cast<FloatAcc*>(p_shared), math::integer_least_multiple(BlockSize, max_lds_align));
        // get acc0 8D thread cluster
        constexpr auto thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4 =
            blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths() /
            blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();
        constexpr auto tm0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I0);
        constexpr auto tn0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I1);
        constexpr auto tm1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I2);
        constexpr auto tn1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I3);
        constexpr auto tm2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I4);
        constexpr auto tn2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I5);
        constexpr auto tn3 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I6);
        constexpr auto tn4 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I7);

        // get acc0 thread map
        constexpr auto m0_n_m1_to_m_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(tm0 * tm1, tm2)),
                       make_pass_through_transform(I1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        constexpr auto threadid_to_m0_n_m1_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(tm0 * tm1, tn0 * tn1 * tn2 * tn3 * tn4, tm2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));
        const auto threadid_to_m_n_thread_cluster_adaptor =
            chain_tensor_adaptors(m0_n_m1_to_m_n_adaptor, threadid_to_m0_n_m1_adaptor);

        // get acc0 2D thread cluster & 2D thread slice
        constexpr auto thread_cluster_desc_m_n = make_naive_tensor_descriptor_packed(
            make_tuple(tm0 * tm1 * tm2, tn0 * tn1 * tn2 * tn3 * tn4));
        constexpr auto thread_slice_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(m0 * m1 * m2, n0 * n1 * n2 * n3 * n4));

        auto blockwise_softmax = BlockwiseSoftmax<BlockSize,
                                                  FloatGemmAcc,
                                                  decltype(threadid_to_m_n_thread_cluster_adaptor),
                                                  decltype(thread_cluster_desc_m_n),
                                                  decltype(thread_slice_desc_m_n)>{};
        
        // Initialize running sum and max of exponentiating row vectors
        using SoftmaxBuf = typename decltype(blockwise_softmax)::BufferType;
        SoftmaxBuf running_sum, running_sum_new, running_max, running_max_new;
        running_sum     = 0;
        running_sum_new = 0;
        running_max     = NumericLimits<FloatGemmAcc>::Lowest();
        running_max_new = NumericLimits<FloatGemmAcc>::Lowest();
/*******************************************************************************/
// set up Gemm1
/*******************************************************************************/
        // B1 matrix in LDS memory, dst of blockwise copy
        constexpr auto b1_block_desc_l0perblock_nperblock_l1 = GetB1BlockDescriptor_L0PerBlock_NPerBlock_L1();

        // A1 matrix blockwise copy
        auto a1_blockwise_copy = ThreadwiseTensorSliceTransfer_StaticToStatic_InterRow<
            FloatAcc,
            FloatA,
            decltype(acc_thread_desc_k0_m_k1),
            decltype(a1_thread_desc_k0_m_k1),
            tensor_operation::element_wise::PassThrough,
            Sequence<A1ThreadSliceK0, A1ThreadSliceM, A1ThreadSliceK1>,
            Sequence<1, 0, 2>,
            2,
            n4,
        //  dst Rowlane
        //  0x76543210  0xfedcba98
        //  src Rowlane
            0x76543210, 0xfedcba98>{tensor_operation::element_wise::PassThrough{}};
        
        // B1 matrix blockwise copy
        auto b1_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B1K0, Gemm1NPerBlock, B1K1>,
                                                B1BlockTransferThreadClusterLengths_BK0_N_BK1,
                                                B1BlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatAB,
                                                decltype(b1_grid_desc_bk0_n_bk1),
                                                decltype(b1_block_desc_bk0_n_bk1),
                                                B1BlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                B1BlockTransferSrcVectorDim,
                                                2,
                                                B1BlockTransferSrcScalarPerVector,
                                                B1BlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                B1ThreadTransferSrcResetCoordinateAfterRun,
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                b1_grid_desc_bk0_n_bk1,
                make_multi_index(0, gemm1_n_block_data_idx_on_grid, 0),
                b1_element_op,
                b1_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});
        
        auto a1_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(a1_thread_desc_k0_m_k1.GetElementSpaceSize());
        auto b1_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(static_cast<FloatB*>(p_shared), b_block_desc_k0perblock_nperblock_k1.GetElementSpaceSize());

        auto blockwise_gemm1 =
            BlockwiseGemmWMMA_k0mk1_k0nk1_m0m1m2n0n1n2m3_CShuffle<BlockSize,
                                                                  FloatA,
                                                                  FloatB,
                                                                  FloatAcc,
                                                                  decltype(a1_thread_desc_k0perblock_mperblock_k1),
                                                                  decltype(b1_block_desc_k0perblock_nperblock_k1),
                                                                  MPerWmma,
                                                                  NPerWmma,
                                                                  MRepeat,
                                                                  NRepeat,
                                                                  KPack>{make_tuple(0, 0, 0, 0, 0)};

        auto acc1_thread_buf = blockwise_gemm1.GetCThreadBuffer();

        const index_t num_gemm1_k_block_outer_loop = b_grid_desc_bk0_n_bk1.GetLength(I1) / NPerBlock;
        constexpr index_t num_gemm1_k_block_inner_loop = NPerBlock / Gemm1KPerBlock;

        // Initialize C
        StaticBuffer<AddressSpaceEnum::Vgpr, FloatAcc, acc1_thread_buf.Size(), true> c_thread_buf;
        c_thread_buf.Clear();

/*******************************************************************************/
        // Flash Attention
        // Dao, Tri, et al. "Flashattention: Fast and memory-efficient exact attention with io-awareness." arXiv preprint arXiv:2205.14135 (2022).
        index_t gemm1_k_block_outer_index = 0;
        // Outer loop, along GEMM_L
        // Inner loop, along GEMM_K
        do{
            // gemm0 start, A-B swaped
            const index_t K0BlockMainLoop = __builtin_amdgcn_readfirstlane(K0 / K0PerBlock);
            GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc_k0_m_k1,
                                                              a_block_desc_k0perblock_mperblock_k1,
                                                              a_blockwise_copy,
                                                              a_grid_buf,
                                                              a_block_buf,
                                                              a_block_slice_copy_step,
                                                              b0_grid_desc_k0_l_k1,
                                                              b_block_desc_k0perblock_nperblock_k1,
                                                              b_blockwise_copy,
                                                              b0_grid_buf,
                                                              b_block_buf,
                                                              b_block_slice_copy_step,
                                                              blockwise_gemm,
                                                              acc_thread_buf,
                                                              K0BlockMainLoop);
            // do MNK padding or upper triangular masking
            if constexpr(MaskOutUpperTriangle || PadN)
            {
                // 8d thread_desc in thread scope
                constexpr auto c_thread_lengths =
                    blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

                // 8d block_desc in block scope
                constexpr auto c_block_lengths =
                    blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

                constexpr auto M0 = c_block_lengths[I0];
                constexpr auto N0 = c_block_lengths[I1];
                constexpr auto M1 = c_block_lengths[I2];
                constexpr auto N1 = c_block_lengths[I3];
                constexpr auto M2 = c_block_lengths[I4];
                constexpr auto N2 = c_block_lengths[I5];
                constexpr auto N3 = c_block_lengths[I6];
                constexpr auto N4 = c_block_lengths[I7];

                // works like multi-dimension static_for (static_ford), but provides both the linear
                // index as well as n-d index
                using Acc0TileIterator = SpaceFillingCurve<
                    decltype(c_thread_lengths),
                    typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
                    typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
                    false>; // SnakeCurved

                auto acc0_thread_origin = blockwise_gemm.CalculateCThreadOriginDataIndex8D(
                    Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

                constexpr auto block_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                    make_tuple(make_unmerge_transform(make_tuple(M0, M1, M2)),
                               make_unmerge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));

                static_for<0, Acc0TileIterator::GetNumOfAccess(), 1>{}([&](auto i) {
                    auto acc0_thread_idx = Acc0TileIterator::GetIndex(i) + acc0_thread_origin;
                    auto m_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
                    auto n_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
                    auto m_global = m_local + m_block_data_idx_on_grid;
                    auto n_global = n_local + n_block_data_idx_on_grid;
                    if(c0_matrix_mask.IsMaskedElement(m_global, n_global))
                    {
                        acc_thread_buf(i) = -ck::NumericLimits<float>::Infinity();
                    }
                    else
                    {
                        acc_element_op(acc_thread_buf(i), acc_thread_buf[i]);
                    }
                });
            }
            else
            {   static_for<0, acc_thread_buf.Size(), 1>{}(
                    [&](auto i) { acc_element_op(acc_thread_buf(i), acc_thread_buf[i]); });
            }


            block_sync_lds(); 
            // gemm0 end

            // Tiled softmax start
            // softmax
            SoftmaxBuf& max = blockwise_softmax.max_value_buf;
            SoftmaxBuf& sum = blockwise_softmax.sum_value_buf;

            blockwise_softmax.Run(acc_thread_buf, workspace_buf);

            // TODO: may convert to log domain
            running_max_new = mathext::max(max, running_max);
            running_sum_new = mathext::exp(running_max - running_max_new) * running_sum +
                              mathext::exp(max - running_max_new) * sum;
            
            // gemm1
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // Initialize acc1
                acc1_thread_buf.Clear();

                // preload data into LDS
                b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                     b1_block_slice_copy_step);

                block_sync_lds(); // wait for reduction LDS read

                b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);

                // main body
                if constexpr(num_gemm1_k_block_inner_loop > 1)
                {
                    static_for<0, num_gemm1_k_block_inner_loop - 1, 1>{}([&](auto i) {
                        // Data cast from FloatAcc to FloatA happen here
                        a1_blockwise_copy.Run(acc_thread_desc_k0_m_k1,
                                              make_tuple(Number<i * A1ThreadSliceK0>{}, I0, I0),
                                              acc_thread_buf,
                                              a1_thread_desc_k0_m_k1,
                                              make_tuple(I0, I0, I0),
                                              a1_thread_buf);

                        b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                        block_sync_lds();

                        gemm1_blockwise_gemm.Run(a1_thread_buf, b1_block_buf, acc1_thread_buf);

                        block_sync_lds();

                        b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                             b1_block_slice_copy_step);

                        b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);
                    });
                }
                // tail
                {
                    a1_blockwise_copy.Run(
                        acc_thread_desc_k0_m_k1,
                        make_tuple(
                            Number<(num_gemm1_k_block_inner_loop - 1) * A1ThreadSliceK0>{}, I0, I0),
                        acc_thread_buf,
                        a1_thread_desc_k0_m_k1,
                        make_tuple(I0, I0, I0),
                        a1_thread_buf);

                    block_sync_lds();

                    gemm1_blockwise_gemm.Run(a1_thread_buf, b1_block_buf, acc1_thread_buf);
                }
            } // end gemm1

            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                gemm1_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
            constexpr auto cm0 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
            constexpr auto cn0 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
            constexpr auto cm1 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
            constexpr auto cn1 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
            constexpr auto cm2 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
            constexpr auto cn2 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
            constexpr auto cn3 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
            constexpr auto cn4 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);
            constexpr auto c_thread_slice_desc_m_n = make_naive_tensor_descriptor_packed(
                make_tuple(cm0 * cm1 * cm2, cn0 * cn1 * cn2 * cn3 * cn4));
            constexpr auto c_thread_buf_slice_m = c_thread_slice_desc_m_n.GetLength(I0);
            constexpr auto c_thread_buf_slice_n = c_thread_slice_desc_m_n.GetLength(I1);

            static_for<0, c_thread_buf_slice_m, 1>{}([&](auto iM) {
                static_for<0, c_thread_buf_slice_n, 1>{}([&](auto iN) {
                    auto I = Number<c_thread_slice_desc_m_n.CalculateOffset(make_tuple(iM, iN))>{};
                    FloatGemmAcc acc1 = acc1_thread_buf[I]; // P*V
                    FloatGemmAcc c    = c_thread_buf[I];    // O
                    FloatGemmAcc c_new =
                        (running_sum[iM] * math::exp(running_max[iM] - running_max_new[iM]) * c +
                         math::exp(max[iM] - running_max_new[iM]) * acc1) /
                        running_sum_new[iM]; 

                    c_thread_buf(I) = c_new; // O_new
                });
            });

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1,
                                                a_block_reset_copy_step); // rewind K
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_bk0_n_bk1,
                                                b_block_reset_copy_step); // rewind K and step N

            // update before next j iteration
            running_max = running_max_new;
            running_sum = running_sum_new;

            block_sync_lds(); // wait for gemm1 LDS read
        }while(++gemm1_k_block_outer_index < num_gemm1_k_block_outer_loop);
/*******************************************************************************/
        // write out to C, implement shuffle
        {
            constexpr auto c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =  
            blockwise_gemm.GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

            // This API Provide All dimension (size) you need
            constexpr auto c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp =
                blockwise_gemm.GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

            constexpr auto MWave              = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I1);
            constexpr auto MSubGroup          = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I2);
            constexpr auto NWave              = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I4);
            constexpr auto NThreadPerSubGroup = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I5);
            constexpr auto MAccVgprs          = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I6);

            // LDS descriptor, shuffle and write out in MRepeat x NRepeat times
            constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
                GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatCShuffle*>(p_shared),
                c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat.GetElementSpaceSize());

            constexpr auto c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs = transform_tensor_descriptor(
                c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMRepeatPerShuffle>{}, // MRepeat per shuffle repeat
                        MWave,                               // MWave
                        MSubGroup,                           // MSubGroup * MAccVgprs = MPerWmma
                        MAccVgprs)),
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNRepeatPerShuffle>{}, // NRepeat per shuffle repeat
                        NWave,                               // NWave
                        NThreadPerSubGroup))),               // NThreadPerSubGroup = NPerWmma
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<>{}, Sequence<0, 1, 2, 6>{}, Sequence<>{}, Sequence<3, 4, 5>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block = blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(MRepeat, MWave, MSubGroup, MAccVgprs))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(NRepeat, NWave, NThreadPerSubGroup))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));
            
            const auto m_thread_data_on_block_idx = m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor.CalculateBottomIndex(
                make_multi_index(m_thread_data_on_block));
            
            const auto n_thread_data_on_block_idx = n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                                   FloatCShuffle,
                                                   decltype(c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs),
                                                   decltype(c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMRepeatPerShuffle,
                                                            I1,
                                                            I1,
                                                            CShuffleNRepeatPerShuffle,
                                                            I1,
                                                            I1,
                                                            MAccVgprs>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                   6,
                                                   1, // vector write pixel
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                    make_multi_index(0,
                                     m_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     0,
                                     n_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3]),
                    ck::tensor_operation::element_wise::PassThrough{}};

            // shuffle: blockwise copy C from LDS to global
            auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                         1,
                         CShuffleNRepeatPerShuffle * NWave * NPerWmma>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                FloatCShuffle,        // typename SrcData,
                FloatC,               // typename DstData,
                decltype(c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0),
                 c_element_op};

            // space filling curve for local reg & global memory
            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MRepeat, 1, 1, NRepeat, 1, 1, MAccVgprs>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6>,
                                  Sequence<CShuffleMRepeatPerShuffle,
                                           1,
                                           1,
                                           CShuffleNRepeatPerShuffle,
                                           1,
                                           1,
                                           MAccVgprs>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                                           1,
                                           CShuffleNRepeatPerShuffle * NWave * NPerWmma>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                                              c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                c_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                    c_shuffle_block_buf,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    c_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);
                    // move on C
                    c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                }
            });
        }
        // clang-format on
    }
};

} // namespace ck
