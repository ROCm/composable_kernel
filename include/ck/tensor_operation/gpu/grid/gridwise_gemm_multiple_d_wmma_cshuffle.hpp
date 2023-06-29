// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseOp,
          typename ADataType,
          typename BDataType,
          typename DsPointer,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_conv_fwd_multiple_d_wmma_cshuffle(
            const ADataType* __restrict__ p_a_grid,
            const BDataType* __restrict__ p_b_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const index_t batch_count,
            const AGridDesc_AK0_M_AK1 a_grid_desc_k0_m_k1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_k0_n_k1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock_,
            const Block2CTileMap block_2_ctile_map,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    __shared__ char p_shared[GridwiseOp::GetSharedMemoryNumberOfByte()];

    DsPointer p_ds_grid_grp;

    static constexpr index_t NumDTensor =
        DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock::Size();

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    GridwiseOp::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                p_b_grid + b_batch_offset,
                                                p_ds_grid_grp,
                                                p_e_grid + e_batch_offset,
                                                p_shared,
                                                a_grid_desc_k0_m_k1,
                                                b_grid_desc_k0_n_k1,
                                                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                                a_element_op,
                                                b_element_op,
                                                cde_element_op,
                                                block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = batch_count;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock_;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

template <typename GridwiseOp,
          typename ADataType,
          typename BDataType,
          typename DsPointer,
          typename EDataType,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename ComputePtrOffsetOfBatch,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_contraction_multiple_d_wmma_cshuffle(
            const ADataType* __restrict__ p_a_grid,
            const BDataType* __restrict__ p_b_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            const index_t batch_count,
            const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
            const BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
            const Block2CTileMap block_2_etile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__))
    // printf("entry kernel launch");
    __shared__ char p_shared[GridwiseOp::GetSharedMemoryNumberOfByte()];

    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    // printf("before compute_ptr_offset call");
    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    static constexpr index_t NumDTensor =
        DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock::Size();

    DsPointer p_ds_grid_grp;

    // printf("before allocate pointer d");

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    // printf("before entry");

    GridwiseOp::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                p_b_grid + b_batch_offset,
                                                p_ds_grid_grp,
                                                p_e_grid + e_batch_offset,
                                                p_shared,
                                                a_grid_desc_k0_m_k1,
                                                b_grid_desc_k0_n_k1,
                                                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                a_element_op,
                                                b_element_op,
                                                cde_element_op,
                                                block_2_etile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = batch_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = block_2_etile_map;
    ignore = compute_ptr_offset_of_batch;
#endif
}

template <typename GridwiseOp,
          typename ADataType,
          typename BDataType,
          typename DsPointer,
          typename EDataType,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_mupltipe_d_wmma_cshuffle(
            const ADataType* __restrict__ p_a_grid,
            const BDataType* __restrict__ p_b_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
            const BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__))
    __shared__ char p_shared[GridwiseOp::GetSharedMemoryNumberOfByte()];

    GridwiseOp::template Run<HasMainKBlockLoop>(p_a_grid,
                                                p_b_grid,
                                                p_ds_grid,
                                                p_e_grid,
                                                p_shared,
                                                a_grid_desc_k0_m_k1,
                                                b_grid_desc_k0_n_k1,
                                                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                a_element_op,
                                                b_element_op,
                                                cde_element_op,
                                                block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx1100__ ))
}

template < // DataType Family
    typename ADataType,
    typename BDataType,
    typename AccDataType,
    typename CShuffleDataType,
    typename DsDataType,
    typename EDataType,
    // InMemory Data Descriptor
    typename AGridDesc_K0_M_K1,
    typename BGridDesc_K0_N_K1,
    typename DsGridDesc_M_N,
    typename EGridDesc_M_N,
    // ElementwiseOp Family
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CDEElementwiseOperation,
    InMemoryDataOperationEnum EGlobalMemoryDataOperation,
    // Tiling Family
    index_t MPerBlock,
    index_t NPerBlock,
    index_t K0PerBlock,
    index_t MPerWmma,
    index_t NPerWmma,
    index_t K1Value,
    index_t MRepeat,
    index_t NRepeat,
    // ThreadCluster Family
    index_t BlockSize,
    typename ABlockTransferThreadClusterLengths_K0_M_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    index_t ABlockTransferSrcVectorDim,
    index_t ABlockTransferSrcScalarPerVector,
    index_t ABlockTransferDstScalarPerVector_K1,
    bool AThreadTransferSrcResetCoordinateAfterRun,
    bool ABlockLdsExtraM,
    typename BBlockTransferThreadClusterLengths_K0_N_K1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    index_t BBlockTransferSrcVectorDim,
    index_t BBlockTransferSrcScalarPerVector,
    index_t BBlockTransferDstScalarPerVector_K1,
    bool BThreadTransferSrcResetCoordinateAfterRun,
    bool BBlockLdsExtraN,
    index_t CShuffleMRepeatPerShuffle,
    index_t CShuffleNRepeatPerShuffle,
    typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
    index_t NumGemmKPrefetchStage = 1,
    LoopScheduler LoopSched       = make_default_loop_scheduler(),
    PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseGemmMultipleD_k0mk1_k0nk1_mn_wmma_cshuffle
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto K1 = Number<K1Value>{};

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

    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_k0perblock_mperblock_k1 =
            GetABlockDescriptor_K0PerBlock_MPerBlock_K1();

        constexpr auto b_block_desc_k0perblock_nperblock_k1 =
            GetBBlockDescriptor_K0PerBlock_NPerBlock_K1();

        constexpr auto cshuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

        constexpr auto max_lds_align = K1;

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_k0perblock_mperblock_k1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_k0perblock_nperblock_k1.GetElementSpaceSize(), max_lds_align);

        constexpr auto c_block_space_size_aligned = math::integer_least_multiple(
            cshuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat.GetElementSpaceSize(),
            max_lds_align);

        return math::max((a_block_space_size_aligned * sizeof(ADataType) +
                          b_block_space_size_aligned * sizeof(BDataType)),
                         c_block_space_size_aligned * sizeof(CShuffleDataType));
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                  const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
                  const DsGridDesc_M_N& ds_grid_desc_m_n,
                  const EGridDesc_M_N& e_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerWmma * MRepeat) == 0) &&
                          (NPerBlock % (NRepeat * NPerWmma)) == 0,
                      "Invalid tuning param!");

        const auto M  = a_grid_desc_k0_m_k1.GetLength(I1);
        const auto N  = b_grid_desc_k0_n_k1.GetLength(I1);
        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);

        bool valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            valid = valid && (M == ds_grid_desc_m_n[i].GetLength(I0) &&
                              N == ds_grid_desc_m_n[i].GetLength(I1));
        });

        if(!valid)
        {
            return false;
        }

        if(!(M == e_grid_desc_m_n.GetLength(I0) && N == e_grid_desc_m_n.GetLength(I1) &&
             K0 == b_grid_desc_k0_n_k1.GetLength(I0) && K1 == a_grid_desc_k0_m_k1.GetLength(I2) &&
             K1 == b_grid_desc_k0_n_k1.GetLength(I2)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // check gridwise gemm pipeline
        const auto num_k_loop = K0 / K0PerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(e_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(a_grid_desc_k0_m_k1.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
             b_grid_desc_k0_n_k1.GetElementSpaceSize() * sizeof(BDataType) <= TwoGB &&
             e_grid_desc_m_n.GetElementSpaceSize() * sizeof(EDataType) <= TwoGB))
        {
            return false;
        }

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / (K0PerBlock * K1);

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    // E desc for destination in blockwise copy
    template <typename EGridDesc_M_N_>
    __host__ __device__ static constexpr auto
    MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const EGridDesc_M_N_& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            e_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return e_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // Ds desc for source in blockwise copy
    template <typename DsGridDesc_M_N_>
    __host__ __device__ static constexpr auto
    MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const DsGridDesc_M_N_& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n[i]);
            },
            Number<NumDTensor>{});
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap(
        const EGridDesc_M_N& e_grid_desc_m_n, index_t /* M01 */, index_t /* N01 */)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, EGridDesc_M_N>(
            e_grid_desc_m_n);
    }

    using DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DsGridDesc_M_N{}))>;
    using EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock  = remove_cvref_t<decltype(
        MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(EGridDesc_M_N{}))>;
    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(EGridDesc_M_N{}, 1, 1))>;
    using DsGridPointer = decltype(MakeDsGridPointer());

    template <bool HasMainKBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               DsGridPointer p_ds_grid,
                               EDataType* __restrict__ p_e_grid,
                               void* __restrict__ p_shared,
                               const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                               const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
                               const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CDEElementwiseOperation& cde_element_op,
                               const Block2CTileMap& block_2_ctile_map)
    {
        // printf("safe entry");
        // clang-format off
/*******************************************************************************/
// Memory buffer zone.
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_k0_m_k1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_k0_n_k1.GetElementSpaceSize());
        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});
        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

/*******************************************************************************/
// BlockIdx.x -> [BlockId.m, BlockId.n]
        const auto block_work_idx = block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        { return; }

        // Store BlockId into SGPR
        const index_t m_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

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
/* typename SrcData,                              */    ADataType,
/* typename DstData,                              */    ADataType,
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
                                                BDataType,
                                                BDataType,
                                                decltype(b_grid_desc_k0_n_k1),
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
                b_grid_desc_k0_n_k1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_k0perblock_nperblock_k1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

/*******************************************************************************/
        // GEMM
        constexpr auto WmmaK = 16;
        constexpr auto KPack = math::integer_least_multiple(K1, WmmaK);

        auto blockwise_gemm =
            BlockwiseGemmWMMA_k0mk1_k0nk1_m0m1m2n0n1n2m3_CShuffle<BlockSize,
                                                         ADataType,
                                                         BDataType,
                                                         AccDataType,
                                                         decltype(a_block_desc_k0perblock_mperblock_k1),
                                                         decltype(b_block_desc_k0perblock_nperblock_k1),
                                                         MPerWmma,
                                                         NPerWmma,
                                                         MRepeat,
                                                         NRepeat,
                                                         KPack>{};

        // Prepare Register for C matrix
        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

/*******************************************************************************/
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(a_block_desc_k0perblock_mperblock_k1.GetElementSpaceSize(), max_lds_align);
        // LDS allocation for A and B: be careful of alignment
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(static_cast<ADataType*>(p_shared), a_block_desc_k0perblock_mperblock_k1.GetElementSpaceSize());
        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(static_cast<BDataType*>(p_shared) + a_block_space_size_aligned, b_block_desc_k0perblock_nperblock_k1.GetElementSpaceSize());
        
        // Shift Per SUB_K
        constexpr auto a_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);

        // gridwise GEMM pipeline
        const index_t K0BlockMainLoop = __builtin_amdgcn_readfirstlane(K0 / K0PerBlock);
        GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc_k0_m_k1,
                                                          a_block_desc_k0perblock_mperblock_k1,
                                                          a_blockwise_copy,
                                                          a_grid_buf,
                                                          a_block_buf,
                                                          a_block_slice_copy_step,
                                                          b_grid_desc_k0_n_k1,
                                                          b_block_desc_k0perblock_nperblock_k1,
                                                          b_blockwise_copy,
                                                          b_grid_buf,
                                                          b_block_buf,
                                                          b_block_slice_copy_step,
                                                          blockwise_gemm,
                                                          c_thread_buf,
                                                          K0BlockMainLoop);
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
                static_cast<CShuffleDataType*>(p_shared),
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
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   CShuffleDataType,
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
            
            // tuple of reference to C/Ds tensor descriptors
            const auto c_ds_desc_refs = concat_tuple_of_reference(
                tie(c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat),
                generate_tie(
                    [&](auto i) -> const auto& // return type should be reference
                    { return ds_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                    Number<NumDTensor>{}));
            
            // tuple of reference to C/Ds tensor buffers
            const auto c_ds_buf_refs = concat_tuple_of_reference(
                tie(c_shuffle_block_buf),
                generate_tie(
                    [&](auto i) -> const auto& // return type should be reference
                    { return ds_grid_buf[i]; },
                    Number<NumDTensor>{}));

            // tuple of starting index of C/Ds blockwise copy
            const auto idx_c_ds_block_begin = container_concat(
                make_tuple(make_multi_index(0, 0, 0, 0)),
                generate_tuple(
                    [&](auto) {
                        return make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0);
                    },
                    Number<NumDTensor>{}));

            // shuffle: blockwise copy C from LDS to global
            auto cde_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v7<
                ThisThreadBlock,            // ThreadGroup
                decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType{})),
                Tuple<EDataType>,
                decltype(c_ds_desc_refs),
                decltype(tie(e_grid_desc_mblock_mperblock_nblock_nperblock)),
                CDEElementwiseOperation,    // ElementwiseOperation,
                Sequence<static_cast<index_t>(EGlobalMemoryDataOperation)>, // DstInMemOp,
                Sequence<1,
                         CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                         1,
                         CShuffleNRepeatPerShuffle * NWave * NPerWmma>, // BlockSliceLengths,
                CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CDEShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                sequence_merge_t<
                    Sequence<true>,
                    uniform_sequence_gen_t<NumDTensor,
                                           false>>,   // bool ThreadTransferSrcResetCoordinateAfterRun,
                Sequence<false>> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_ds_desc_refs,
                 idx_c_ds_block_begin,
                 tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                 make_tuple(make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0)),
                 cde_element_op};

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
            constexpr auto sfc_cde_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                                           1,
                                           CShuffleNRepeatPerShuffle * NWave * NPerWmma>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_cde_global.GetNumOfAccess(), "wrong!");

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
                cde_shuffle_block_copy_lds_to_global.Run(
                    c_ds_desc_refs,
                    c_ds_buf_refs,
                    tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                    tie(e_grid_buf));

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto cde_global_step = sfc_cde_global.GetForwardStep(access_id);
                    // move on Ds
                    static_for<0, NumDTensor, 1>{}([&](auto i) {
                        cde_shuffle_block_copy_lds_to_global.MoveSrcSliceWindow(
                            c_ds_desc_refs, i + I1, cde_global_step);
                    });

                    // move on E
                    cde_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                        I0,
                        cde_global_step);
                }
            });
        }
        // clang-format on
    }
};

} // namespace ck
