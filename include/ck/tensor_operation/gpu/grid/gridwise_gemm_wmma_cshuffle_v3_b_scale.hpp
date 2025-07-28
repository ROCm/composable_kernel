// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/env.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp"

namespace ck {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t ScaleBlockN, // scale N
          index_t ScaleBlockK, // scale K
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v4,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct GridwiseGemm_wmma_cshuffle_v3_b_scale
    : GridwiseGemm_wmma_cshuffle_v3_base<
          ALayout,
          BLayout,
          CLayout,
          ADataType,
          BDataType,
          AccDataType,
          CShuffleDataType,
          CDataType,
          AElementwiseOperation,
          BElementwiseOperation,
          CElementwiseOperation,
          GemmSpec,
          BlockSize,
          MPerBlock,
          NPerBlock,
          KPerBlock,
          AK1Value,
          BK1Value,
          MPerWmma,
          NPerWmma,
          MRepeat,
          NRepeat,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          ABlockTransferThreadClusterArrangeOrder,
          ABlockTransferSrcAccessOrder,
          ABlockTransferSrcVectorDim,
          ABlockTransferSrcScalarPerVector,
          ABlockTransferDstScalarPerVector_AK1,
          AThreadTransferSrcResetCoordinateAfterRun,
          ABlockLdsExtraM,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          BBlockTransferThreadClusterArrangeOrder,
          BBlockTransferSrcAccessOrder,
          BBlockTransferSrcVectorDim,
          BBlockTransferSrcScalarPerVector,
          BBlockTransferDstScalarPerVector_BK1,
          BThreadTransferSrcResetCoordinateAfterRun,
          BBlockLdsExtraN,
          CShuffleMRepeatPerShuffle,
          CShuffleNRepeatPerShuffle,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlkGemmPipeSched,
          BlkGemmPipelineVer,
          ComputeTypeA,
          ComputeTypeB,
          PermuteA,
          PermuteB>
{
    using BScaleType = ck::half_t;

    using Base = GridwiseGemm_wmma_cshuffle_v3_base<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1Value,
        BK1Value,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB>;

    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;
    using Base::I4;
    using Base::I5;
    using Base::I6;
    using Base::I7;

    using Base::AK0Number;
    using Base::AK1Number;
    using Base::BK0Number;
    using Base::BK1Number;

    using Base::APackedSize;
    using Base::BPackedSize;

    using Base::CalculateAK0Padded;
    using Base::CalculateBK0Padded;
    using Base::CalculateKPadded;
    using Base::CalculateKRead;
    using Base::CalculateMBlock;
    using Base::CalculateMPadded;
    using Base::CalculateNBlock;
    using Base::CalculateNPadded;
    using Base::MakeAGridDescriptor_AK0_M_AK1;
    using Base::MakeBGridDescriptor_BK0_N_BK1;
    using Base::MakeCGridDescriptor_M_N;

    using Base::GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat;

    using Base::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1;
    using Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1;

    struct Problem
    {
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         index_t StrideA_,
                         index_t StrideB_,
                         index_t StrideC_,
                         index_t StrideScaleB_,
                         index_t KBatch_)
            : M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
              StrideScaleB{StrideScaleB_},
              KBatch{KBatch_},
              MPadded{CalculateMPadded(M_)},
              NPadded{CalculateNPadded(N_)},
              KRead{CalculateKRead(K_, KBatch_)},
              KPadded{CalculateKPadded(K_, KBatch_)},
              AK0{CalculateAK0Padded(K_, KBatch_)},
              BK0{CalculateBK0Padded(K_, KBatch_)},
              MBlock{CalculateMBlock(M_)},
              NBlock{CalculateNBlock(N_)}
        {
        }

        __host__ void Print() const
        {
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "SScaleB:" << StrideScaleB << ", " << "MP:" << MPadded << ", "
                      << "NP:" << NPadded << ", " << "KRead:" << KRead << ", " << "KP:" << KPadded
                      << ", " << "AK0:" << AK0 << ", " << "BK0:" << BK0 << ", "
                      << "MBlock: " << MBlock << ", " << "NBlock: " << NBlock << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t StrideScaleB;
        index_t KBatch;
        index_t MPadded;
        index_t NPadded;
        index_t KRead;
        index_t KPadded;
        index_t AK0;
        index_t BK0;
        index_t MBlock;
        index_t NBlock;
    };

    // Argument
    struct Argument : public tensor_operation::device::BaseArgument, public Problem
    {
        __host__ Argument(const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          CDataType* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_,
                          index_t StrideScaleB_,
                          const BScaleType* p_b_scale_grid_,
                          index_t k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CElementwiseOperation c_element_op_,
                          bool is_reduce_ = false)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideC_, StrideScaleB_, k_batch_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_},
              p_b_scale_grid{p_b_scale_grid_},
              a_element_op{a_element_op_},
              b_element_op{b_element_op_},
              c_element_op{c_element_op_},
              is_reduce(is_reduce_)
        {
        }

        __host__ __device__ inline bool IsReduceAdd() const
        {
            return (Problem::KBatch > 1) && is_reduce;
        }

        __host__ __device__ inline bool IsAtomicAdd() const
        {
            return (Problem::KBatch > 1) && (!is_reduce);
        }

        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        CDataType* p_c_grid;

        const BScaleType* p_b_scale_grid;
        const AElementwiseOperation a_element_op;
        const BElementwiseOperation b_element_op;
        const CElementwiseOperation c_element_op;
        bool is_reduce;
    };

    struct SplitKBatchOffset
    {

        __device__ SplitKBatchOffset(Argument& karg)
        {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = blockIdx.z * karg.KRead / APackedSize;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = blockIdx.z * karg.KRead * karg.StrideA;
            }

            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = blockIdx.z * karg.KRead * karg.StrideB;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                if constexpr(!PermuteB)
                {
                    b_k_split_offset = blockIdx.z * karg.KRead / BPackedSize;
                }
                else
                {
                    const int k0_offset = karg.KRead * karg.N;
                    b_k_split_offset    = blockIdx.z * k0_offset / BPackedSize;
                }
            }

            // Calculate B scale offset
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                scale_k_split_offset = blockIdx.z * (karg.KRead / ScaleBlockK) * karg.StrideB;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                scale_k_split_offset = blockIdx.z * (karg.KRead / ScaleBlockK);
            }

            if(blockIdx.z < static_cast<uint32_t>(karg.KBatch - 1))
            {
                karg.K = karg.KRead;
            }
            else
            {
                karg.K = karg.K - karg.KRead * (karg.KBatch - 1);
            }

            if(karg.IsReduceAdd())
            {
                c_reduce_offset = blockIdx.z * karg.M * karg.N;
            }
            else
            {
                c_reduce_offset = 0;
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t scale_k_split_offset; // New member for scale matrix offset
        index_t c_reduce_offset;
    };

    using BlockwiseGemmPipe = typename Base::BlockwiseGemmPipe;

    // return block_id to C matrix tile idx (m0, n0) mapping
    // if arch = gfx942
    using Block2CTileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;
    // using Block2CTileMap = BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>;

    template <index_t NumberOfBuffers, typename BScaleGridDesc_BN_AK, typename BScaleType>
    __device__ static auto MakeBScale(const BScaleGridDesc_BN_AK& b_scale_grid_desc_bn_ak,
                                      const BScaleType* p_b_scale_grid,
                                      index_t block_n_id)
    {
        const auto b_scale_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_scale_grid, b_scale_grid_desc_bn_ak.GetElementSpaceSize());

        static constexpr auto wmma =
            WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>{};
        static constexpr auto KPerThread = wmma.selected_wmma.k_per_wmma;

        static constexpr auto ScaleSliceSizeN = NRepeat;
        static constexpr auto ScaleSliceSizeK = (KPerThread + ScaleBlockK - 1) / ScaleBlockK;

        constexpr auto b_scale_thread_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<ScaleSliceSizeN>{}, Number<ScaleSliceSizeK>{}));

        constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

        auto b_thread_offset_n = get_thread_local_1d_id() % NPerWmma +
                                 (get_thread_local_1d_id() / 32) % NWaves * NPerWmma;
        auto b_thread_offset_k = (get_thread_local_1d_id() % 32) / NPerWmma * KPerThread;

        auto b_scale_thread_copy =
            ThreadwiseTensorSliceTransfer_v2<BScaleType,
                                             BScaleType,
                                             decltype(b_scale_grid_desc_bn_ak),
                                             decltype(b_scale_thread_desc),
                                             Sequence<1, ScaleSliceSizeK>,
                                             Sequence<0, 1>,
                                             1,
                                             ScaleSliceSizeK,
                                             1,
                                             false>(
                b_scale_grid_desc_bn_ak,
                make_multi_index(block_n_id * NPerBlock / ScaleBlockN + b_thread_offset_n,
                                 b_thread_offset_k / ScaleBlockK));

        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_scale_thread_desc.GetElementSpaceSize());

        using BScale =
            typename BlockwiseGemmPipe::template BScale<ScaleSliceSizeN,
                                                        ScaleSliceSizeK,
                                                        NWaves,
                                                        ScaleBlockK,
                                                        NumberOfBuffers,
                                                        decltype(b_scale_grid_desc_bn_ak),
                                                        decltype(b_scale_thread_copy),
                                                        decltype(b_scale_grid_buf),
                                                        decltype(b_scale_thread_buf),
                                                        decltype(b_scale_thread_desc)>;

        return BScale{b_scale_grid_desc_bn_ak, b_scale_thread_copy, b_scale_grid_buf};
    }

    __device__ static index_t GetKBlockPerScale()
    {
        return (ScaleBlockK + KPerBlock - 1) / KPerBlock;
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(const ADataType* p_a_grid,
                               const BDataType* p_b_grid,
                               CDataType* p_c_grid,
                               const BScaleType* p_b_scale_grid,
                               void* p_shared,
                               const Problem& problem)
    {
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideA, problem.AK0);
        const auto b_grid_desc_bk0_n_bk1 = MakeBGridDescriptor_BK0_N_BK1(
            problem.K, problem.KPadded, problem.N, problem.NPadded, problem.StrideB, problem.BK0);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);

        // B Scale grid
        const auto b_scale_grid_desc_bn_ak = make_naive_tensor_descriptor(
            make_tuple(math::integer_divide_ceil(problem.N, ScaleBlockN),
                       math::integer_divide_ceil(problem.K, ScaleBlockK)),
            make_tuple(problem.StrideScaleB, 1));

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{problem.M, problem.N, 4};

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

        // BScale struct
        auto b_scale_struct = MakeBScale<1>(b_scale_grid_desc_bn_ak, p_b_scale_grid, block_n_id);

        const index_t num_k_block_per_scale = GetKBlockPerScale();

        Base::template Run<decltype(a_grid_desc_ak0_m_ak1),
                           decltype(b_grid_desc_bk0_n_bk1),
                           decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                           decltype(b_scale_struct),
                           HasMainKBlockLoop,
                           CGlobalMemoryDataOperation,
                           TailNum>(p_a_grid,
                                    p_b_grid,
                                    p_c_grid,
                                    p_shared,
                                    a_grid_desc_ak0_m_ak1,
                                    b_grid_desc_bk0_n_bk1,
                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    block_m_id,
                                    block_n_id,
                                    num_k_block_per_scale,
                                    b_scale_struct);
    }

    // NOTE: Wrapper function to have __global__ function in common
    // between gemm_universal, b_scale, ab_scale, etc.
    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void
    Run(void* p_shared, const SplitKBatchOffset& splitk_batch_offset, const Argument& karg)
    {
        Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
            karg.p_b_grid + splitk_batch_offset.b_k_split_offset,
            karg.p_c_grid + splitk_batch_offset.c_reduce_offset,
            karg.p_b_scale_grid + splitk_batch_offset.scale_k_split_offset,
            p_shared,
            karg);
    }
};

} // namespace ck
