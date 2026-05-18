// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_waveletmodel.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {

// Wave-specialized gridwise GEMM for convolution backward weight.
//
// Extends the flat-GEMM wavelet model (gridwise_gemm_xdl_waveletmodel_cshuffle.hpp) to work
// with conv-to-GEMM descriptors from the device op. Uses a 2-way wave split:
//   - Load waves (threads TileMath..TileMath+TileLoad-1): RunRead + MoveSrcSliceWindow + RunWrite
//   - Math waves (threads 0..TileMath-1): LDS read + MFMA + CShuffle epilogue
//
// The conv-to-GEMM descriptor transforms (which generate heavy VALU ops) execute only on load
// waves, while math waves see simple LDS layouts — eliminating the VALU/MFMA slot conflict.

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
          index_t NumGemmKPrefetchStage,
          index_t TileLoadThreadGroupSize,
          index_t TileMathThreadGroupSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraMCustom,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraNCustom,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          typename ComputeTypeA = CDataType,
          typename ComputeTypeB = ComputeTypeA>
struct GridwiseGemm_xdl_waveletmodel_cshuffle_conv_v3
    : public GridwiseGemm_xdl_cshuffle_base<
          ALayout,
          BLayout,
          CLayout,
          ADataType,
          BDataType,
          AccDataType,
          CShuffleDataType,
          Tuple<>,
          CDataType,
          AElementwiseOperation,
          BElementwiseOperation,
          TileMathThreadGroupSize, // BlockSize = math group only (MFMA wave assignment)
          MPerBlock,
          NPerBlock,
          KPerBlock,
          AK1Value,
          BK1Value,
          MPerXdl,
          NPerXdl,
          MXdlPerWave,
          NXdlPerWave,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          ABlockTransferThreadClusterArrangeOrder,
          ABlockTransferSrcAccessOrder,
          ABlockTransferSrcVectorDim,
          ABlockTransferSrcScalarPerVector,
          ABlockTransferDstScalarPerVector_AK1,
          AThreadTransferSrcResetCoordinateAfterRun,
          ABlockLdsExtraMCustom,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          BBlockTransferThreadClusterArrangeOrder,
          BBlockTransferSrcAccessOrder,
          BBlockTransferSrcVectorDim,
          BBlockTransferSrcScalarPerVector,
          BBlockTransferDstScalarPerVector_BK1,
          BThreadTransferSrcResetCoordinateAfterRun,
          BBlockLdsExtraNCustom,
          CShuffleMXdlPerWavePerShuffle,
          CShuffleNXdlPerWavePerShuffle,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
          ComputeTypeA,
          ComputeTypeB,
          false> // ForceNaiveLayout
{
    using Base = GridwiseGemm_xdl_cshuffle_base<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        Tuple<>,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        TileMathThreadGroupSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1Value,
        BK1Value,
        MPerXdl,
        NPerXdl,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraMCustom,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraNCustom,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        ComputeTypeA,
        ComputeTypeB,
        false>;

    using Base::AK0Number;
    using Base::AK1Number;
    using Base::BK0Number;
    using Base::BK1Number;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    static constexpr auto BlockSize       = TileMathThreadGroupSize;
    static constexpr auto LaunchBlockSize = TileLoadThreadGroupSize + TileMathThreadGroupSize;

    // DirectLoad is not supported — wavelet model requires LDS as the sync boundary
    static constexpr bool DirectLoadEnabled = false;

    // =========================================================================
    // Wave specialization thread groups
    // =========================================================================

    // Math threads are 0..TileMath-1 (must be first — BlockwiseGemmXdlops_v1 uses
    // get_thread_local_1d_id() directly for wave index computation)
    struct TileMathThreadGroup
    {
        __device__ static constexpr index_t GetNumOfThread() { return TileMathThreadGroupSize; }

        __device__ static constexpr bool IsBelong()
        {
            return get_thread_local_1d_id() < TileMathThreadGroupSize;
        }

        __device__ static index_t GetThreadId() { return get_thread_local_1d_id(); }
    };

    // Load threads: TileMath..TileMath+TileLoad-1
    struct TileLoadThreadGroup
    {
        __device__ static constexpr index_t GetNumOfThread() { return TileLoadThreadGroupSize; }

        __device__ static constexpr bool IsBelong()
        {
            return get_thread_local_1d_id() >= TileMathThreadGroupSize;
        }

        __device__ static index_t GetThreadId()
        {
            return get_thread_local_1d_id() - TileMathThreadGroupSize;
        }
    };

    using CShuffleBlockTransferThreadGroup = ThisThreadBlock<TileMathThreadGroupSize>;

    // =========================================================================
    // Wave pipelines (from gridwise_gemm_waveletmodel.hpp)
    // =========================================================================

    using GridwiseGemmLoad = GridwiseGemmLoadWave<TileLoadThreadGroup, NumGemmKPrefetchStage>;
    using GridwiseGemmMath = GridwiseGemmMathWave<TileMathThreadGroup, NumGemmKPrefetchStage>;

    // =========================================================================
    // MFMA instruction selection
    // =========================================================================
    static constexpr auto lcm_AK1_BK1 = math::lcm(AK1Number, BK1Number);
    static constexpr bool is_single_rate_mfma =
        (((is_same<ComputeTypeA, half_t>::value || is_same<ComputeTypeA, bhalf_t>::value) &&
          lcm_AK1_BK1 <= 4) ||
         (is_same<ComputeTypeA, int8_t>::value && lcm_AK1_BK1 <= 8) ||
         ((is_same<ComputeTypeA, f8_t>::value || is_same<ComputeTypeA, bf8_t>::value) &&
          lcm_AK1_BK1 < 32))
            ? true
            : false;
    static constexpr auto is_scale_mfma = false;
    static constexpr index_t KPack =
        math::max(lcm_AK1_BK1,
                  MfmaSelector<ComputeTypeA,
                               MPerXdl,
                               NPerXdl,
                               ComputeTypeA,
                               is_single_rate_mfma,
                               is_scale_mfma>::selected_mfma.k_per_blk);

    // =========================================================================
    // Block-to-tile mapping (grouped conv)
    // =========================================================================
    using Block2CTileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;

    __host__ static auto CalculateGridSize(index_t M, index_t N, index_t KBatch, index_t Batch)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), KBatch, Batch);
    }

    // =========================================================================
    // Padding and block count helpers (for Problem struct)
    // =========================================================================
    __host__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * KPerBlock;
    }

    __host__ static auto CalculateAK0Padded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / AK1Value);
    }

    __host__ static auto CalculateBK0Padded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / BK1Value);
    }

    __host__ static auto CalculateKRead(index_t K, index_t K_Batch = 1)
    {
        constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
        auto K_t                = K_Batch * KReadVec;
        return (K + K_t - 1) / K_t * KReadVec;
    }

    __host__ static auto CalculateMBlock(index_t M)
    {
        return math::integer_divide_ceil(M, MPerBlock);
    }

    __host__ static auto CalculateNBlock(index_t N)
    {
        return math::integer_divide_ceil(N, NPerBlock);
    }

    // =========================================================================
    // Problem and Argument (passed to GPU kernel)
    // =========================================================================
    struct Problem
    {
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         index_t StrideA_,
                         index_t StrideB_,
                         index_t StrideC_,
                         index_t KBatch_)
            : M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
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
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "KRead:" << KRead << ", " << "KP:" << KPadded << ", " << "AK0:" << AK0
                      << ", " << "BK0:" << BK0 << ", " << "MBlock: " << MBlock << ", "
                      << "NBlock: " << NBlock << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
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
                          index_t k_batch_)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideC_, k_batch_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_}
        {
        }

        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        CDataType* p_c_grid;
    };

    // =========================================================================
    // LDS block descriptors (device-arch aware)
    // =========================================================================
    template <typename DeviceArch>
    __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch)
    {
        if constexpr(is_same_v<DeviceArch, gfx950_t>)
        {
            constexpr index_t ABlockLdsExtraM = 1;
            return make_naive_tensor_descriptor(
                make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1Number, AK1Number, I1));
        }
        else
        {
            return Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch{});
        }
    }

    template <typename DeviceArch>
    __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch)
    {
        if constexpr(is_same_v<DeviceArch, gfx950_t>)
        {
            constexpr index_t BBlockLdsExtraN = 1;
            return make_naive_tensor_descriptor(
                make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1Number, BK1Number, I1));
        }
        else
        {
            return Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch{});
        }
    }

    // =========================================================================
    // Compilation parameter validation
    // =========================================================================
    IS_VALID_COMPILATION_PARAMETER_IMPL(CDataType)

    // =========================================================================
    // =========================================================================
    // Shared memory sizing
    // =========================================================================
    template <typename DeviceArch>
    __device__ static constexpr index_t GetSharedMemoryNumberOfByte(DeviceArch)
    {
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch{});
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch{});

        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        // Data LDS (A + B)
        constexpr auto data_lds_bytes = a_block_space_size_aligned * sizeof(ADataType) +
                                        b_block_space_size_aligned * sizeof(BDataType);

        // C shuffle LDS (reuses same memory after GEMM loop — mutually exclusive)
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            Base::GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DeviceArch{});

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max(data_lds_bytes, c_block_size * sizeof(CShuffleDataType));
    }

    // =========================================================================
    // K-loop control
    // =========================================================================
    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;
        return GridwiseGemmMath::CalculateHasMainLoop(num_loop);
    }

    // =========================================================================
    // C grid descriptor for epilogue
    // =========================================================================
    template <typename CGridDesc>
    __host__ __device__ static constexpr auto MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const CGridDesc& c_grid_desc_m_n, index_t MBlock, index_t NBlock)
    {
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // =========================================================================
    // Run: wave-specialized GEMM kernel
    // =========================================================================
    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              bool SplitKOffsetHack = false>
    __device__ static void Run(const ADataType* p_a_grid,
                               const BDataType* p_b_grid,
                               CDataType* p_c_grid,
                               void* p_shared,
                               const Problem& problem,
                               const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_K1& b_grid_desc_bk0_n_bk1,
                               const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const index_t k_id    = 0,
                               const index_t k_batch = 1)
    {
        const long_index_t a_space_size_divisor = SplitKOffsetHack ? k_batch : 1;
        const long_index_t b_space_size_divisor = SplitKOffsetHack ? k_batch : 1;

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize() / a_space_size_divisor);
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize() / b_space_size_divisor);

        const AElementwiseOperation a_element_op{};
        const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{problem.M, problem.N, 4};

        const auto block_work_idx = block_2_ctile_map.CalculateBottomIndex(
            make_multi_index(static_cast<index_t>(blockIdx.x)));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

        // HACK: force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ADataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<BDataType*>(p_shared) +
                a_block_space_size_aligned * sizeof(ADataType) / sizeof(BDataType),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1Number, 0, 0);

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            (KPerBlock * problem.KBatch));

        // =====================================================================
        // Wave specialization
        // =====================================================================

        if(TileLoadThreadGroup::IsBelong())
        {
            auto a_blockwise_copy =
                ThreadGroupTensorSliceTransfer_v4r1<TileLoadThreadGroup,
                                                    AElementwiseOperation,
                                                    ck::tensor_operation::element_wise::PassThrough,
                                                    InMemoryDataOperationEnum::Set,
                                                    Sequence<AK0Number, MPerBlock, AK1Number>,
                                                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                    ABlockTransferThreadClusterArrangeOrder,
                                                    ADataType,
                                                    ADataType,
                                                    decltype(a_grid_desc_ak0_m_ak1),
                                                    decltype(a_block_desc_ak0_m_ak1),
                                                    ABlockTransferSrcAccessOrder,
                                                    Sequence<0, 1, 2>,
                                                    ABlockTransferSrcVectorDim,
                                                    2,
                                                    ABlockTransferSrcScalarPerVector,
                                                    ABlockTransferDstScalarPerVector_AK1,
                                                    1,
                                                    1,
                                                    AThreadTransferSrcResetCoordinateAfterRun,
                                                    true,
                                                    NumGemmKPrefetchStage>(
                    a_grid_desc_ak0_m_ak1,
                    make_multi_index(SplitKOffsetHack ? 0 : k_id, m_block_data_idx_on_grid, 0),
                    a_element_op,
                    a_block_desc_ak0_m_ak1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

            auto b_blockwise_copy =
                ThreadGroupTensorSliceTransfer_v4r1<TileLoadThreadGroup,
                                                    BElementwiseOperation,
                                                    ck::tensor_operation::element_wise::PassThrough,
                                                    InMemoryDataOperationEnum::Set,
                                                    Sequence<BK0Number, NPerBlock, BK1Number>,
                                                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                    BBlockTransferThreadClusterArrangeOrder,
                                                    BDataType,
                                                    BDataType,
                                                    decltype(b_grid_desc_bk0_n_bk1),
                                                    decltype(b_block_desc_bk0_n_bk1),
                                                    BBlockTransferSrcAccessOrder,
                                                    Sequence<0, 1, 2>,
                                                    BBlockTransferSrcVectorDim,
                                                    2,
                                                    BBlockTransferSrcScalarPerVector,
                                                    BBlockTransferDstScalarPerVector_BK1,
                                                    1,
                                                    1,
                                                    BThreadTransferSrcResetCoordinateAfterRun,
                                                    true,
                                                    NumGemmKPrefetchStage>(
                    b_grid_desc_bk0_n_bk1,
                    make_multi_index(SplitKOffsetHack ? 0 : k_id, n_block_data_idx_on_grid, 0),
                    b_element_op,
                    b_block_desc_bk0_n_bk1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

            GridwiseGemmLoad::template RunLoadWavePipeline<HasMainKBlockLoop>(
                a_grid_desc_ak0_m_ak1,
                a_block_desc_ak0_m_ak1,
                a_blockwise_copy,
                a_grid_buf,
                a_block_buf,
                a_block_slice_copy_step,
                b_grid_desc_bk0_n_bk1,
                b_block_desc_bk0_n_bk1,
                b_blockwise_copy,
                b_grid_buf,
                b_block_buf,
                b_block_slice_copy_step,
                num_k_block_main_loop);

            // Match epilogue LDS syncs: RunEpilogue calls block_sync_lds() twice per
            // SFC access iteration (once before VGPR→LDS, once before LDS→global).
            // For !TransposeC && !IsMxGemm, the SFC access count equals
            // (MXdlPerWave / CShuffleMXdlPerWavePerShuffle) * (NXdlPerWave /
            // CShuffleNXdlPerWavePerShuffle) because the M2/M4/N2 SFC dimensions have equal
            // total and per-access lengths. See Base::GetCThreadWiseSpaceFillingCurve and
            // RunEpilogue in gridwise_gemm_xdl_cshuffle_common.hpp.
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "CShuffle XdlPerWave must divide MXdlPerWave/NXdlPerWave");
            constexpr index_t num_access = (MXdlPerWave / CShuffleMXdlPerWavePerShuffle) *
                                           (NXdlPerWave / CShuffleNXdlPerWavePerShuffle);
            static_for<0, 2 * num_access, 1>{}([&](auto) { block_sync_lds(); });
        }
        else if(TileMathThreadGroup::IsBelong())
        {
            auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<
                TileMathThreadGroupSize,
                ADataType,
                BDataType,
                AccDataType,
                decltype(a_block_desc_ak0_m_ak1),
                decltype(b_block_desc_bk0_n_bk1),
                MPerXdl,
                NPerXdl,
                MXdlPerWave,
                NXdlPerWave,
                KPack,
                ComputeTypeA,
                ComputeTypeB>{};

            auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

            GridwiseGemmMath::template RunMathWavePipeline<HasMainKBlockLoop>(
                a_block_buf, b_block_buf, blockwise_gemm, c_thread_buf, num_k_block_main_loop);

            Base::template RunEpilogue<CGlobalMemoryDataOperation, false, false>(
                blockwise_gemm,
                c_grid_desc_mblock_mperblock_nblock_nperblock,
                c_thread_buf,
                block_m_id,
                block_n_id,
                p_shared,
                p_c_grid,
                c_element_op);
        }
    }
};

} // namespace ck
