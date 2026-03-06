// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <iostream>
#include <ostream>
#endif

#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_reduce_wmma.hpp"
#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_welford_wmma.hpp"
#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma.hpp"
#include "ck/tensor_operation/gpu/grid/epilogue_direct_store.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles_preshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles_interleave.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"

namespace ck {

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_gemm_wmma_cshuffle_v3(typename GridwiseGemm::Argument karg)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using e_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<e_data_type, ck::half_t> ||
                    std::is_same_v<e_data_type, ck::bhalf_t>)))
    {
#endif
        using EpilogueType =
            typename std::conditional<GridwiseGemm::IsBWaveTransferApplicable &&
                                          GridwiseGemm::UseDirectStore,
                                      typename GridwiseGemm::EpilogueDirectStore,
                                      typename GridwiseGemm::EpilogueCShuffle>::type;

        constexpr index_t LDS_size =
            GridwiseGemm::template GetSharedMemoryNumberOfByte<EpilogueType>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        auto epilogue_args = EpilogueType{};

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared, splitk_batch_offset, karg, epilogue_args);

#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
#endif
}

template <typename GridwiseGemm,
          typename ComputePtrOffsetOfStridedBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          bool IsBScaled           = false,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_batched_gemm_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg, // This works for now but it actually receives a
                                              // DeviceBatchedGemm_Wmma_CShuffleV3::Argument
                                              // argument through implicit conversion to base class!
        const ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using c_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(CGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<c_data_type, ck::half_t> ||
                    std::is_same_v<c_data_type, ck::bhalf_t>)))
    {
#endif
        // The normal approach to batching would be to increase the grid size by just stretching out
        // the grid Z dimension (which is the outermost dimension), but this depends on lower level
        // functions not directly using the Z dimension for other calculations. As it turns out, k
        // batching does rely directly on blockIdx.Z through SplitKBatchOffset. Therefore, for now
        // we will use the grid Y dimension for batching. This may be a bit fragile.
        const index_t g_idx = amd_wave_read_first_lane(blockIdx.y);

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t c_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx));

        using EpilogueType =
            typename std::conditional<GridwiseGemm::IsBWaveTransferApplicable &&
                                          GridwiseGemm::UseDirectStore,
                                      typename GridwiseGemm::EpilogueDirectStore,
                                      typename GridwiseGemm::EpilogueCShuffle>::type;

        constexpr index_t LDS_size =
            GridwiseGemm::template GetSharedMemoryNumberOfByte<EpilogueType>();

        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        // shift A matrices pointer for splitk
        typename GridwiseGemm::AsGridPointer p_as_grid_shift;
        static_for<0, GridwiseGemm::NumATensor, 1>{}([&](auto i) {
            using ADataType_ =
                remove_cvref_t<tuple_element_t<i.value, typename GridwiseGemm::AsDataType_>>;
            p_as_grid_shift(i) = static_cast<const ADataType_*>(karg.p_as_grid[i]) +
                                 splitk_batch_offset.a_k_split_offset[i] + a_batch_offset;
        });

        // shift B matrices pointer for splitk
        typename GridwiseGemm::BsGridPointer p_bs_grid_shift;
        static_for<0, GridwiseGemm::NumBTensor, 1>{}([&](auto i) {
            using BDataType_ =
                remove_cvref_t<tuple_element_t<i.value, typename GridwiseGemm::BsDataType_>>;
            p_bs_grid_shift(i) = static_cast<const BDataType_*>(karg.p_bs_grid[i]) +
                                 splitk_batch_offset.b_k_split_offset[i] + b_batch_offset;
        });

        auto epilogue_args = EpilogueType{};

        if constexpr(IsBScaled)
        {
            const long_index_t b_scale_batch_offset =
                amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetScaleBPtrOffset(g_idx));

            GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
                p_as_grid_shift,
                p_bs_grid_shift,
                karg.p_ds_grid,
                karg.p_e_grid + splitk_batch_offset.c_reduce_offset + c_batch_offset,
                karg.p_a_scale_grid,
                karg.p_b_scale_grid + b_scale_batch_offset +
                    splitk_batch_offset.scale_b_k_split_offset,
                p_shared,
                karg,
                karg.a_element_op,
                karg.b_element_op,
                karg.cde_element_op,
                epilogue_args);
        }
        else
        {
            GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
                p_as_grid_shift,
                p_bs_grid_shift,
                karg.p_ds_grid,
                karg.p_e_grid + splitk_batch_offset.c_reduce_offset + c_batch_offset,
                p_shared,
                karg,
                karg.a_element_op,
                karg.b_element_op,
                karg.cde_element_op,
                epilogue_args);
        }
#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
    ignore = compute_ptr_offset_of_batch;
#endif
}

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_gemm_b_preshuffle_wmma_cshuffle_v3(typename GridwiseGemm::Argument karg)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using e_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<e_data_type, ck::half_t> ||
                    std::is_same_v<e_data_type, ck::bhalf_t>)))
    {
#endif
        constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
            typename GridwiseGemm::EpilogueCShuffle>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        const index_t num_k_per_block = math::integer_divide_ceil(karg.K, GridwiseGemm::KPack);
        const index_t k_id            = blockIdx.z * num_k_per_block;

        auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared,
            splitk_batch_offset,
            karg,
            epilogue_args,
            0, /* A_k_id == 0 (we shift the pointer for splitk) */
            k_id);

#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
#endif
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
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
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename ComputeTypeA,
          typename ComputeTypeB,
          bool PermuteA,
          bool PermuteB,
          bool IsBPreShuffled          = false,
          bool ForceThreadTileTransfer = false, // only needed for convolution (limitation)
          bool IsFusedKernel           = false>
struct GridwiseGemm_wmma_cshuffle_v3_base
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    using LDSTypeA =
        typename std::conditional<(NumATensor > 1),
                                  ComputeTypeA,
                                  remove_cvref_t<tuple_element_t<0, AsDataType>>>::type;
    using LDSTypeB =
        typename std::conditional<(NumBTensor > 1),
                                  ComputeTypeB,
                                  remove_cvref_t<tuple_element_t<0, BsDataType>>>::type;

    static constexpr auto EShuffleBlockTransferScalarPerVector =
        CDEShuffleBlockTransferScalarPerVectors{}[I0];

    // K1 should be Number<...>
    static constexpr auto AK0Number = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0Number = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1Number = Number<AK1Value>{};
    static constexpr auto BK1Number = Number<BK1Value>{};

    static constexpr index_t KPerWmmaBlk =
        WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>::selected_wmma
            .k_per_blk;

    static constexpr index_t KInnerA = ck::math::integer_divide_ceil(AK1Value, KPerWmmaBlk);

    static constexpr index_t KInnerB = ck::math::integer_divide_ceil(BK1Value, KPerWmmaBlk);

    static constexpr index_t KInner = IsBPreShuffled ? KInnerB : ck::math::min(KInnerA, KInnerB);

    static constexpr index_t KPack =
        KInner *
        WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>::selected_wmma
            .k_per_wmma;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;

    static constexpr index_t APackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<LDSTypeA>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    static constexpr index_t BPackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<LDSTypeB>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    static constexpr index_t WaveSize =
        WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>::selected_wmma
            .wave_size;

    __host__ __device__ static constexpr bool AWaveTransferApplicable()
    {
        return !ForceThreadTileTransfer && APackedSize == 1 &&
               ABlockTransferSrcScalarPerVector == 8 && ABlockTransferDstScalarPerVector_AK1 == 8 &&
               BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 && AK1Value == 8 &&
               !IsBPreShuffled;
    }

    __host__ __device__ static constexpr bool BWaveTransferApplicable()
    {
        return !ForceThreadTileTransfer && BPackedSize == 1 &&
               BBlockTransferSrcScalarPerVector == 8 && BBlockTransferDstScalarPerVector_BK1 == 8 &&
               BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 && BK1Value == 8;
    }

#ifdef __gfx12__
    static constexpr bool IsAWaveTransferApplicable = AWaveTransferApplicable();

    static constexpr bool IsBWaveTransferApplicable = BWaveTransferApplicable();

    static constexpr bool IsWaveTileInterleavedFitting =
        (NPerBlock / NPerWmma / NRepeat) * (KPerBlock / KPack) >= (BlockSize / WaveSize);

    // We need to investigate if it makes sense to remove cshuffle for smaller types
    // Currently we use direct store for NRepeat equal to 4 or 8. For 16 bit type we use at
    // least buffer store 64 bit for 16 contiguous threads -> 128 bytes in total (full cache line)
    static constexpr bool UseDirectStore = is_same_v<BLayout, tensor_layout::gemm::ColumnMajor> &&
                                           sizeof(ComputeTypeB) == 2 && sizeof(EDataType) == 2 &&
                                           NumDTensor == 0 && (NRepeat == 4 || NRepeat == 8) &&
                                           !IsFusedKernel && IsWaveTileInterleavedFitting;
#else
    static constexpr bool IsAWaveTransferApplicable = false;
    static constexpr bool IsBWaveTransferApplicable = false;
    static constexpr bool UseDirectStore            = false;
#endif

    static constexpr bool UseBlockPaddingA =
        ABlockLdsExtraM || BlkGemmPipelineVer == BlockGemmPipelineVersion::v4;
    using ATransfer = typename std::conditional<
        IsAWaveTransferApplicable,
        ABTransferWaveTiles<ALayout,
                            tensor_layout::gemm::RowMajor,
                            LDSTypeA,
                            BlockSize,
                            MPerBlock,
                            KPerBlock,
                            MPerWmma,
                            KPack,
                            AK1Value,
                            WaveSize>,
        ABTransferThreadTiles<ALayout,
                              tensor_layout::gemm::RowMajor,
                              LDSTypeA,
                              BlockSize,
                              MPerBlock,
                              KPerBlock,
                              MPerWmma,
                              AK1Value,
                              KPack,
                              KInner,
                              KPerWmmaBlk,
                              UseBlockPaddingA,
                              PermuteA,
                              ABlockTransferThreadClusterLengths_AK0_M_AK1,
                              ABlockTransferThreadClusterArrangeOrder,
                              ABlockTransferSrcAccessOrder,
                              ABlockTransferSrcVectorDim,
                              ABlockTransferSrcScalarPerVector,
                              ABlockTransferDstScalarPerVector_AK1,
                              AThreadTransferSrcResetCoordinateAfterRun>>::type;

    static constexpr bool UseBlockPaddingB =
        BBlockLdsExtraN || BlkGemmPipelineVer == BlockGemmPipelineVersion::v4;
    using BTransfer = typename std::conditional<
        IsBPreShuffled,
        ABTransferThreadTilesPreShuffle<BLayout,
                                        tensor_layout::gemm::ColumnMajor,
                                        NPerBlock,
                                        KPerBlock,
                                        NPerWmma,
                                        NPerBlock / NPerWmma / NRepeat,
                                        BK1Value,
                                        WaveSize,
                                        KPack,
                                        BBlockTransferSrcScalarPerVector,
                                        BThreadTransferSrcResetCoordinateAfterRun>,
        typename std::conditional<
            IsBWaveTransferApplicable,
            typename std::conditional<
                UseDirectStore,
                ABTransferWaveTilesInterleave<BLayout,
                                              tensor_layout::gemm::ColumnMajor,
                                              LDSTypeB,
                                              BlockSize,
                                              NPerBlock,
                                              KPerBlock,
                                              NPerWmma,
                                              KPack,
                                              BK1Value,
                                              WaveSize,
                                              NPerBlock / NPerWmma / NRepeat>,
                ABTransferWaveTiles<BLayout,
                                    tensor_layout::gemm::ColumnMajor,
                                    LDSTypeB,
                                    BlockSize,
                                    NPerBlock,
                                    KPerBlock,
                                    NPerWmma,
                                    KPack,
                                    BK1Value,
                                    WaveSize>>::type,
            ABTransferThreadTiles<BLayout,
                                  tensor_layout::gemm::ColumnMajor,
                                  LDSTypeB,
                                  BlockSize,
                                  NPerBlock,
                                  KPerBlock,
                                  NPerWmma,
                                  BK1Value,
                                  KPack,
                                  KInner,
                                  KPerWmmaBlk,
                                  UseBlockPaddingB,
                                  PermuteB,
                                  BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                  BBlockTransferThreadClusterArrangeOrder,
                                  BBlockTransferSrcAccessOrder,
                                  BBlockTransferSrcVectorDim,
                                  BBlockTransferSrcScalarPerVector,
                                  BBlockTransferDstScalarPerVector_BK1,
                                  BThreadTransferSrcResetCoordinateAfterRun>>::type>::type;

    static_assert(!(is_same_v<remove_cvref_t<LDSTypeB>, pk_i4_t> &&
                    GemmSpec != tensor_operation::device::GemmSpecialization::Default),
                  "pk_i4_t does not support padding");

    static_assert(!PermuteA, "PermuteA is not supported");

    // return block_id to C matrix tile idx (m0, n0) mapping
    using Block2CTileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;

    // Calculate grid size taking into account splitk (KBatch)
    // 2D grid (x,z)
    __host__ __device__ static auto CalculateGridSize(index_t M, index_t N, index_t KBatch)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), 1, KBatch);
    }

    // Calculate grid size taking into account splitk (KBatch) and multiple groups (Batch)
    // 3D grid (x,y,z)
    __host__ __device__ static auto
    CalculateGridSize(index_t M, index_t N, index_t KBatch, index_t Batch)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), KBatch, Batch);
    }

    __host__ __device__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateKPadded(index_t K)
    {
        return math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
    }

    __host__ __device__ static auto CalculateAK0Padded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / AK1Value);
    }

    __host__ __device__ static auto CalculateBK0Padded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / BK1Value);
    }

    __host__ __device__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * KPerBlock;
    }

    __host__ __device__ static auto CalculateKRead(index_t K, index_t K_Batch = 1)
    {
        constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
        auto K_t                = K_Batch * KReadVec;
        return (K + K_t - 1) / K_t * KReadVec;
    }

    __host__ __device__ static auto CalculateMBlock(index_t M)
    {
        return math::integer_divide_ceil(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNBlock(index_t N)
    {
        return math::integer_divide_ceil(N, NPerBlock);
    }

    static constexpr auto MakeAsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;

                return static_cast<const ADataType_*>(nullptr);
            },
            Number<NumATensor>{});
    }

    static constexpr auto MakeBsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;

                return static_cast<const BDataType_*>(nullptr);
            },
            Number<NumBTensor>{});
    }

    using AsGridPointer = decltype(MakeAsGridPointer());
    using BsGridPointer = decltype(MakeBsGridPointer());

    __host__ __device__ static auto MakeAGridDescriptor_M_K(index_t M, index_t K, index_t StrideA)
    {
        if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
        {
            return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
        }
        else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
        {
            return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
        }
    }

    __host__ __device__ static auto MakeBGridDescriptor_N_K(index_t N, index_t K, index_t StrideB)
    {
        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(I1, StrideB));
        }
        else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(StrideB, I1));
        }
    }

    template <typename BaseDescriptors_M_K>
    __host__ __device__ static auto
    MakeAsGridDescriptor_AK0_M_AK1(const BaseDescriptors_M_K& base_descs,
                                   const index_t M,
                                   const index_t MPad,
                                   const index_t K,
                                   const index_t KPad,
                                   const std::array<index_t, NumATensor>& StrideAs,
                                   const index_t AK0)
    {
        constexpr bool padM = GemmSpec == GemmSpecialization::MKPadding ||
                              GemmSpec == GemmSpecialization::MNKPadding ||
                              GemmSpec == GemmSpecialization::MPadding ||
                              GemmSpec == GemmSpecialization::MNPadding;
        constexpr bool padK = GemmSpec == GemmSpecialization::MKPadding ||
                              GemmSpec == GemmSpecialization::MNKPadding ||
                              GemmSpec == GemmSpecialization::KPadding ||
                              GemmSpec == GemmSpecialization::NKPadding;
        return generate_tuple(
            [&](auto i) {
                return ATransfer::template MakeGridDescriptor<padM, padK>(
                    base_descs[i], M, MPad, K, KPad, StrideAs[i], AK0);
            },
            Number<NumATensor>{});
    }

    template <typename GridDescBase>
    __device__ static auto MakeAGridDescriptor_AK0_M_AK1(const GridDescBase& base_desc)
    {
        const auto M = base_desc.GetLength(I0);
        const auto K = base_desc.GetLength(I1);

        const auto AK0 = K / AK1Value;

        constexpr bool padM = false;
        constexpr bool padK = false;
        return ATransfer::template MakeGridDescriptor<padM, padK>(base_desc, M, M, K, K, 0, AK0);
    }

    template <typename BaseDescriptors_M_K>
    __host__ __device__ static auto
    MakeAsGridDescriptor_AK0_M_AK1(const BaseDescriptors_M_K& base_descs, const index_t KBatch = 1)
    {
        const index_t M = base_descs.At(I0).GetLength(I0);
        const index_t K = base_descs.At(I0).GetLength(I1);

        const index_t MPad = CalculateMPadded(M);
        const index_t KPad = CalculateKPadded(K, KBatch);

        const index_t AK0 = CalculateAK0Padded(K, KBatch);

        return MakeAsGridDescriptor_AK0_M_AK1(base_descs, M, MPad, K, KPad, {}, AK0);
    }

    __host__ __device__ static auto
    MakeAsGridDescriptor_AK0_M_AK1(const index_t M,
                                   const index_t MPad,
                                   const index_t K,
                                   const index_t KPad,
                                   const std::array<index_t, NumATensor>& StrideAs,
                                   const index_t AK0)
    {
        const auto base_descs =
            generate_tuple([&](auto i) { return MakeAGridDescriptor_M_K(M, K, StrideAs[i]); },
                           Number<NumATensor>{});
        return MakeAsGridDescriptor_AK0_M_AK1(base_descs, M, MPad, K, KPad, StrideAs, AK0);
    }

    template <typename BaseDescriptors_N_K>
    __host__ __device__ static auto
    MakeBsGridDescriptor_BK0_N_BK1(const BaseDescriptors_N_K& base_descs,
                                   const index_t K,
                                   const index_t KPad,
                                   const index_t N,
                                   const index_t NPad,
                                   const std::array<index_t, NumBTensor>& StrideBs,
                                   const index_t BK0)
    {
        constexpr bool padN = GemmSpec == GemmSpecialization::NKPadding ||
                              GemmSpec == GemmSpecialization::MNKPadding ||
                              GemmSpec == GemmSpecialization::NPadding ||
                              GemmSpec == GemmSpecialization::MNPadding;
        constexpr bool padK = GemmSpec == GemmSpecialization::NKPadding ||
                              GemmSpec == GemmSpecialization::MNKPadding ||
                              GemmSpec == GemmSpecialization::KPadding ||
                              GemmSpec == GemmSpecialization::MKPadding;
        return generate_tuple(
            [&](auto i) {
                return BTransfer::template MakeGridDescriptor<padN, padK>(
                    base_descs[i], N, NPad, K, KPad, StrideBs[i], BK0);
            },
            Number<NumBTensor>{});
    }

    template <typename GridDescBase>
    __device__ static auto MakeBGridDescriptor_BK0_N_BK1(const GridDescBase& base_desc)
    {
        const auto N = base_desc.GetLength(I0);
        const auto K = base_desc.GetLength(I1);

        const auto BK0 = K / BK1Value;

        constexpr bool padN = false;
        constexpr bool padK = false;
        return BTransfer::template MakeGridDescriptor<padN, padK>(base_desc, N, N, K, K, 0, BK0);
    }

    template <typename BaseDescriptors_N_K>
    __host__ __device__ static auto
    MakeBsGridDescriptor_BK0_N_BK1(const BaseDescriptors_N_K& base_descs, const index_t KBatch = 1)
    {
        const index_t N = base_descs.At(I0).GetLength(I0);
        const index_t K = base_descs.At(I0).GetLength(I1);

        const index_t NPad = CalculateNPadded(N);
        const index_t KPad = CalculateKPadded(K, KBatch);

        const index_t BK0 = CalculateBK0Padded(K, KBatch);

        return MakeBsGridDescriptor_BK0_N_BK1(base_descs, K, KPad, N, NPad, {}, BK0);
    }

    __host__ __device__ static auto
    MakeBsGridDescriptor_BK0_N_BK1(const index_t K,
                                   const index_t KPad,
                                   const index_t N,
                                   const index_t NPad,
                                   const std::array<index_t, NumBTensor>& StrideBs,
                                   const index_t BK0)
    {

        const auto base_descs =
            generate_tuple([&](auto i) { return MakeBGridDescriptor_N_K(N, K, StrideBs[i]); },
                           Number<NumBTensor>{});
        return MakeBsGridDescriptor_BK0_N_BK1(base_descs, K, KPad, N, NPad, StrideBs, BK0);
    }

    __host__ __device__ static constexpr auto MakeAWmmaTileDescriptor()
    {
        constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWmma);

        return ATransfer::template MakeWmmaTileDescriptor<MRepeat, MWaves>();
    }

    __host__ __device__ static constexpr auto MakeBWmmaTileDescriptor()
    {
        constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

        return BTransfer::template MakeWmmaTileDescriptor<NRepeat, NWaves>();
    }

    template <typename DELayout>
    __host__ __device__ static auto
    MakeDEGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideDE)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, DELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideDE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, DELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideDE));
            }
        }();

        // pad M and N
        return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                           make_tuple(make_right_pad_transform(M, MPad - M),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}));
        // TODO: Investigate why this path is not used in the original
        // gridwise_gemm_xdl_cshuffle_v3.hpp
#if 0

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(M, MPad - M),
                                                          make_right_pad_transform(N, NPad - N)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(M, MPad - M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
#endif
    }

    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    using DsGridPointer = decltype(MakeDsGridPointer());

    __host__ __device__ static auto MakeDsGridDescriptor_M_N(
        index_t M, index_t MPad, index_t N, index_t NPad, std::array<index_t, NumDTensor> StrideDs)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                return MakeDEGridDescriptor_M_N<DLayout>(M, MPad, N, NPad, StrideDs[i]);
            },
            Number<NumDTensor>{});
    }

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const DsGridDesc& ds_grid_desc_m_n,
                                                           index_t MBlock,
                                                           index_t NBlock)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ds_grid_desc_m_n[i], MBlock, NBlock);
            },
            Number<NumDTensor>{});
    }

    using BlockwiseGemmPipe =
        remove_cvref_t<decltype(BlockGemmPipeline_Selector<BlkGemmPipelineVer,
                                                           BlkGemmPipeSched,
                                                           BlockSize,
                                                           LDSTypeA,
                                                           LDSTypeB,
                                                           ComputeTypeA,
                                                           ComputeTypeB,
                                                           AccDataType,
                                                           decltype(MakeAWmmaTileDescriptor()),
                                                           decltype(MakeBWmmaTileDescriptor()),
                                                           ABlockTransferSrcScalarPerVector,
                                                           BBlockTransferSrcScalarPerVector,
                                                           MPerBlock,
                                                           NPerBlock,
                                                           KPerBlock,
                                                           MPerWmma,
                                                           NPerWmma,
                                                           MRepeat,
                                                           NRepeat,
                                                           KPack,
                                                           KInner,
                                                           false,
                                                           IsBPreShuffled>())>;

    // Used to create obj in global function and pass it to Run method
    using EpilogueCShuffle =
        EpilogueCShuffle<DsDataType,
                         EDataType,
                         AccDataType,
                         CShuffleDataType,
                         MPerBlock,
                         NPerBlock,
                         MPerWmma,
                         NPerWmma,
                         MRepeat,
                         NRepeat,
                         CShuffleMRepeatPerShuffle,
                         CShuffleNRepeatPerShuffle,
                         CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                         CDEShuffleBlockTransferScalarPerVectors,
                         CDEElementwiseOperation,
                         ThisThreadBlock,
                         BlockwiseGemmPipe>;

    using EpilogueDirectStore = EpilogueDirectStore<DsDataType,
                                                    EDataType,
                                                    AccDataType,
                                                    MRepeat,
                                                    NRepeat,
                                                    CDEElementwiseOperation,
                                                    BlockwiseGemmPipe>;

    using EpilogueWelfordCShuffle = EpilogueWelfordCShuffle<
        DsDataType,
        EDataType,
        AccDataType,
        CShuffleDataType,
        MPerBlock,
        NPerBlock,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        CDEElementwiseOperation,
        ThisThreadBlock,
        BlockwiseGemmPipe,
        BlockSize>;

    template <typename ReduceTrait>
    using EpilogueReduceCShuffle = EpilogueReduceCShuffle<
        DsDataType,
        EDataType,
        AccDataType,
        CShuffleDataType,
        MPerBlock,
        NPerBlock,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        CDEElementwiseOperation,
        ThisThreadBlock,
        BlockwiseGemmPipe,
        GemmSpec,
        BlockSize,
        ReduceTrait>;

    template <typename DEGridDesc>
    __host__ __device__ static constexpr auto
    MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const DEGridDesc& de_grid_desc_m_n,
                                                           index_t MBlock,
                                                           index_t NBlock)
    {
        const auto de_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            de_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return de_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // Conditions for Wave Transfer with transpose:
    // - 16 bit type: K % 8 == 0 (4 subtiles of 8x8)
    // - 8 bit type: K % 8 == 0 and M % 16 == 0 (2 subtiles of 8x16)
    __host__ static constexpr bool CheckValidityAWaveTransfer(const index_t& M, const index_t& K)
    {
        if constexpr(AWaveTransferApplicable() &&
                     !(is_same<tensor_layout::gemm::RowMajor, ALayout>::value))
        {
            if(!(K % ABlockTransferDstScalarPerVector_AK1 == 0))
            {
                return false;
            }
            bool pass = true;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                pass &= !(sizeof(ADataType_) == 1 &&
                          !(M % (2 * ABlockTransferSrcScalarPerVector) == 0));
            });
            return pass;
        }
        else
        {
            return true;
        }
    }

    __host__ static constexpr bool CheckValidityBWaveTransfer(const index_t& N, const index_t& K)
    {
        if constexpr(BWaveTransferApplicable() &&
                     !(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value))
        {
            if(!(K % BBlockTransferDstScalarPerVector_BK1 == 0))
            {
                return false;
            }
            bool pass = true;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                pass &= !(sizeof(BDataType_) == 1 &&
                          !(N % (2 * BBlockTransferSrcScalarPerVector) == 0));
            });
            return pass;
        }
        else
        {
            return true;
        }
    }

    __host__ __device__ static constexpr bool
    CheckValidity(const index_t M,
                  const index_t N,
                  const index_t K,
                  const index_t StrideA,
                  const index_t StrideB,
                  const std::array<index_t, NumDTensor> StrideDs,
                  const index_t StrideE,
                  const index_t KBatch)
    {

        ignore              = StrideDs;
        const auto M_padded = CalculateMPadded(M);
        const auto N_padded = CalculateMPadded(N);
        const auto K_padded = CalculateKPadded(K, KBatch);

        const auto e_grid_desc_m_n =
            MakeDEGridDescriptor_M_N<ELayout>(M, M_padded, N, N_padded, StrideE);

        const index_t AK0 = CalculateAK0Padded(K, KBatch);
        const index_t BK0 = CalculateBK0Padded(K, KBatch);

        const auto a_grid_desc_ak0_m_ak1 = MakeAsGridDescriptor_AK0_M_AK1(
            M, M_padded, K, K_padded, std::array<index_t, 1>{StrideA}, AK0);

        const auto b_grid_desc_bk0_n_bk1 = MakeBsGridDescriptor_BK0_N_BK1(
            K, K_padded, N, N_padded, std::array<index_t, 1>{StrideB}, BK0);

        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        const auto& a_desc = a_grid_desc_ak0_m_ak1.At(I0);
        const auto& b_desc = b_grid_desc_bk0_n_bk1.At(I0);

        if(!(a_desc.GetElementSpaceSize() * sizeof(LDSTypeA) <= TwoGB &&
             b_desc.GetElementSpaceSize() * sizeof(LDSTypeB) <= TwoGB &&
             e_grid_desc_m_n.GetElementSpaceSize() * sizeof(EDataType) <= TwoGB))
        {
            return false;
        }

        return true;
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Argument>
    __host__ static constexpr bool CheckValidity(const Argument& karg,
                                                 bool allow_short_v3_pipe = false)
    {
        static_assert((MPerBlock % (MPerWmma * MRepeat) == 0) &&
                          (NPerBlock % (NPerWmma * NRepeat)) == 0,
                      "Invalid tuning param!");

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     !(is_same<tensor_layout::gemm::RowMajor, ALayout>::value))
        {
            if(!(karg.M % MPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M value is not a multiple of MPerBlock! M: " << karg.M << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     (is_same<tensor_layout::gemm::RowMajor, BLayout>::value))
        {
            if(!(karg.N % NPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N value is not a multiple of NPerBlock! N: " << karg.N << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {

            auto K_t = karg.KBatch * KPerBlock;
            if(!(karg.K % K_t == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K value is not a multiple of K_Batch * K0PerBlock * K1! "
                                 "K_Batch:"
                              << karg.KBatch << " " << "K0PerBlock:" << KPerBlock << " "
                              << "K1:" << AK1Number << " " << "K:" << karg.K << " " << __FILE__
                              << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
            auto K_t                = karg.KBatch * KReadVec;
            auto KReadPadSplited    = math::integer_divide_ceil(karg.K, K_t) * KReadVec;
            if((KReadPadSplited * (karg.KBatch - 1)) >= karg.K)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K value too low for combination of AK1/BK1/KBatch. AK1: "
                              << AK1Number << ", BK1: " << BK1Number << ", KBatch: " << karg.KBatch
                              << ", K: " << karg.K << " " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << karg.K
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << karg.K
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ELayout>::value)
        {
            if(karg.N % EShuffleBlockTransferScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of "
                                 "EShuffleBlockTransferScalarPerVector ("
                              << EShuffleBlockTransferScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.M % EShuffleBlockTransferScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of "
                                 "EShuffleBlockTransferScalarPerVector ("
                              << EShuffleBlockTransferScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(is_same<remove_cvref_t<EDataType>, half_t>::value ||
                       is_same<remove_cvref_t<EDataType>, float>::value ||
                       is_same<remove_cvref_t<EDataType>, bhalf_t>::value ||
                       is_same<remove_cvref_t<EDataType>, int32_t>::value))
        {
            if(karg.IsAtomicAdd() && karg.KBatch > 1)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << " KBatch: " << karg.KBatch << " > 1 is not supported for this "
                              << "destination type (EDataType) " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = karg.AK0 / (KPerBlock / AK1Value);

        if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
        {
            if(num_k_loop <= BlockwiseGemmPipe::PrefetchStages)
            {
                if(!(allow_short_v3_pipe && BlkGemmPipelineVer == BlockGemmPipelineVersion::v3))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "Pipeline validation failed: num_k_loop (" << num_k_loop
                                  << ") <= PrefetchStages (" << BlockwiseGemmPipe::PrefetchStages
                                  << ") for pipeline version != v1." << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }

        if constexpr(is_same<remove_cvref_t<EDataType>, int8_t>::value)
        {
            if(karg.KBatch > 1)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "int8_t does not support KBatch > 1. KBatch: " << karg.KBatch
                              << " " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockHasHotloop(num_loop);
    }

    __host__ __device__ static constexpr TailNumber CalculateKBlockLoopTailNum(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockLoopTailNum(num_loop);
    }

    template <typename EpilogueType>
    __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = ATransfer::GetBlockDescriptor();
        constexpr auto b_block_desc_bk0_n_bk1 = BTransfer::GetBlockDescriptor();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        constexpr auto a_block_space_size_aligned =
            ATransfer::IsLDSNeeded()
                ? math::integer_least_multiple(a_block_desc_ak0_m_ak1.GetElementSpaceSize(),
                                               max_lds_align)
                : 0;

        constexpr auto b_block_space_size_aligned =
            BTransfer::IsLDSNeeded()
                ? math::integer_least_multiple(b_block_desc_bk0_n_bk1.GetElementSpaceSize(),
                                               max_lds_align)
                : 0;

        if constexpr(EpilogueType::IsLDSNeeded())
        {
            // LDS allocation for C shuffle in LDS
            constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
                EpilogueType::
                    GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

            constexpr auto c_block_size =
                c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat
                    .GetElementSpaceSize();

            return math::max((a_block_space_size_aligned * sizeof(LDSTypeA) / APackedSize +
                              b_block_space_size_aligned * sizeof(LDSTypeB) / BPackedSize),
                             c_block_size * sizeof(CShuffleDataType));
        }
        else
        {
            return a_block_space_size_aligned * sizeof(LDSTypeA) / APackedSize +
                   b_block_space_size_aligned * sizeof(LDSTypeB) / BPackedSize;
        }
    }

    // Note: arguments k_batch and k_id should be set if splitk is used
    // with implicit gemm (no pointer shift but shift using tensor descriptors)
    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename AScaleStruct,
              typename BScaleStruct,
              typename EpilogueArgument,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(AsGridPointer p_as_grid,
                               BsGridPointer p_bs_grid,
                               DsGridPointer p_ds_grid,
                               EDataType* p_e_grid,
                               void* p_shared,
                               const AGridDesc_AK0_M_K1& as_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_K1& bs_grid_desc_bk0_n_bk1,
                               const DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               AElementwiseOperation a_element_op,
                               BElementwiseOperation b_element_op,
                               CDEElementwiseOperation cde_element_op,
                               const index_t& block_m_id,
                               const index_t& block_n_id,
                               const index_t& num_k_block_per_scale,
                               AScaleStruct& a_scale_struct,
                               BScaleStruct& b_scale_struct,
                               EpilogueArgument& epilogue_args,
                               const index_t A_k_id  = 0,
                               const index_t B_k_id  = 0,
                               const index_t k_batch = 1)
    {
        const auto as_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_as_grid[i], as_grid_desc_ak0_m_ak1[i].GetElementSpaceSize());
            },
            Number<NumATensor>{});

        const auto bs_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_bs_grid[i], bs_grid_desc_bk0_n_bk1[i].GetElementSpaceSize());
            },
            Number<NumBTensor>{});

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = ATransfer::GetBlockDescriptor();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = BTransfer::GetBlockDescriptor();

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ATransfer::template GetBlockTransfer<AGridDesc_AK0_M_K1,
                                                 decltype(a_block_desc_ak0_m_ak1),
                                                 AsDataType,
                                                 AElementwiseOperation,
                                                 BlockwiseGemmPipe::GlobalBufferNum>(
                as_grid_desc_ak0_m_ak1, a_block_desc_ak0_m_ak1, a_element_op, block_m_id, A_k_id);

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BTransfer::template GetBlockTransfer<BGridDesc_BK0_N_K1,
                                                 decltype(b_block_desc_bk0_n_bk1),
                                                 BsDataType,
                                                 BElementwiseOperation,
                                                 BlockwiseGemmPipe::GlobalBufferNum>(
                bs_grid_desc_bk0_n_bk1, b_block_desc_bk0_n_bk1, b_element_op, block_n_id, B_k_id);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        // Cast after lds
        auto a_block_buf = ATransfer::GetBuffer(static_cast<LDSTypeA*>(p_shared),
                                                a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf =
            BTransfer::GetBuffer(reinterpret_cast<LDSTypeB*>(static_cast<char*>(p_shared) +
                                                             a_block_space_size_aligned *
                                                                 sizeof(LDSTypeA) / APackedSize),
                                 b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = ATransfer::GetBlockStep();
        constexpr auto b_block_slice_copy_step = BTransfer::GetBlockStep();

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            ATransfer::GetKDimension(as_grid_desc_ak0_m_ak1[I0]) / (KPerBlock * k_batch));

        blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(
            ATransfer::template get_first_element_workaround<NumATensor>(as_grid_desc_ak0_m_ak1),
            a_block_desc_ak0_m_ak1,
            a_blockwise_copy,
            ATransfer::template get_first_element_workaround<NumATensor>(as_grid_buf),
            a_block_buf,
            a_block_slice_copy_step,
            BTransfer::template get_first_element_workaround<NumBTensor>(bs_grid_desc_bk0_n_bk1),
            b_block_desc_bk0_n_bk1,
            b_blockwise_copy,
            BTransfer::template get_first_element_workaround<NumBTensor>(bs_grid_buf),
            b_block_buf,
            b_block_slice_copy_step,
            c_thread_buf,
            a_scale_struct,
            b_scale_struct,
            num_k_block_main_loop,
            num_k_block_per_scale);

        // Epilogue:
        //  - CShuffle / direct store
        //  - Multiple Ds
        //  - Fused operations
        epilogue_args.template Run<EGlobalMemoryDataOperation>(
            c_thread_buf,
            p_ds_grid,
            p_e_grid,
            p_shared,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock,
            cde_element_op,
            block_m_id,
            block_n_id);
    }
};

} // namespace ck
