// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_moe_blockscale_b_preshuffle_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1_gather.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"
#define DEBUG_LOG 0

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck {

// Currently we do not have a elegant way to put single lds buffer & double lds buffer pipe in same
// kernel function Blockers:
// 1. Two separted declaration of __shared__ pointer is the key to make sure data access operate on
// two lds chunks.
// 2. Occupied __shared__ won't release until whole shader end, a.k.a AB and C may not use same lds
// buffer when we declare __shared__ inside blkgemmpipe

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Even>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_moe_gemm(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        __shared__ char
            p_shared[GridwiseGemm::template GetSharedMemoryNumberOfByte<true>(get_device_arch())];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_sorted_token_ids,
            karg.p_sorted_expert_ids,
            karg.p_max_token_id,
            karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
            karg.p_b_grid + splitk_batch_offset.b_k_split_offset,
            karg.p_ds_grid,
            karg.p_c_grid,
            karg.p_a_scale_grid + splitk_batch_offset.ascale_k_split_offset,
            karg.p_b_scale_grid + splitk_batch_offset.bscale_k_split_offset,
            p_shared,
            karg,
            karg.a_element_op,
            karg.b_element_op,
            karg.c_element_op);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Even>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_moe_gemm_2lds(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        __shared__ char
            p_shared[GridwiseGemm::template GetSharedMemoryNumberOfByte<true>(get_device_arch())];
        __shared__ char
            p_shared1[GridwiseGemm::template GetSharedMemoryNumberOfByte<true>(get_device_arch())];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_sorted_token_ids,
            karg.p_sorted_expert_ids,
            karg.p_max_token_id,
            karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
            karg.p_b_grid + splitk_batch_offset.b_k_split_offset,
            karg.p_ds_grid,
            karg.p_c_grid,
            karg.p_a_scale_grid + splitk_batch_offset.ascale_k_split_offset,
            karg.p_b_scale_grid + splitk_batch_offset.bscale_k_split_offset,
            p_shared,
            p_shared1,
            karg,
            karg.a_element_op,
            karg.b_element_op,
            karg.c_element_op);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
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
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          index_t ActivationOperation                 = 0,
          bool NSwizzle                               = false,
          bool IsInputGemm                            = true,
          bool IsSplitK                               = false,
          bool MulRoutedWeight                        = true,
          typename IndexType                          = index_t,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          typename LDSTypeA                           = ADataType,
          typename LDSTypeB                           = BDataType,
          bool NonTemporalLoadB                       = false>
struct GridwiseMoeGemmBlockScale
    : public GridwiseGemm_xdl_cshuffle_base<
          ALayout,
          BLayout,
          CLayout,
          LDSTypeA,
          LDSTypeB,
          AccDataType,
          CShuffleDataType,
          DsDataType,
          CDataType,
          AElementwiseOperation,
          BElementwiseOperation,
          BlockSize,
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
          ABlockLdsExtraM,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          BBlockTransferThreadClusterArrangeOrder,
          BBlockTransferSrcAccessOrder,
          BBlockTransferSrcVectorDim,
          BBlockTransferSrcScalarPerVector,
          BBlockTransferDstScalarPerVector_BK1,
          BThreadTransferSrcResetCoordinateAfterRun,
          BBlockLdsExtraN,
          CShuffleMXdlPerWavePerShuffle,
          CShuffleNXdlPerWavePerShuffle,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          CDEShuffleBlockTransferScalarPerVectors,
          ComputeTypeA,
          ComputeTypeB,
          false>
{
    using Base = GridwiseGemm_xdl_cshuffle_base<
        ALayout,
        BLayout,
        CLayout,
        LDSTypeA,
        LDSTypeB,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        BlockSize,
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
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        ComputeTypeA,
        ComputeTypeB,
        false>;

    using Base::AK0Number;
    using Base::AK1Number;
    using Base::BK0Number;
    using Base::BK1Number;
    using Base::CShuffleBlockTransferScalarPerVector_NPerBlock;
    using Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1;
    using Base::GetSharedMemoryNumberOfByte;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;
    using Base::I4;
    using Base::I5;
    using Base::I6;
    using Base::I7;
    using ThisThreadBlock = typename Base::ThisThreadBlock;
    using Base::NumDTensor;
    using AScaleType = float;
    using BScaleType = float;

    static constexpr auto BlockSizeNumber = Number<BlockSize>{};

    using mfma_selector = MfmaSelector<ComputeTypeA, MPerXdl, NPerXdl, ComputeTypeB>;
    static constexpr index_t KPack =
        math::max(math::lcm(AK1Number, BK1Number), mfma_selector::selected_mfma.k_per_blk);
    static constexpr index_t KGroup = []() {
        if constexpr(is_same_v<remove_cvref_t<BDataType>, f8_t>)
            // On gfx950, we have a mfma that required 32 f8 elements as input,
            // splited into 2 groups of 16 f8 elements.
            // the 2 groups is not contiguous in the B preshuffed layout.
            // and we do not want it to be contiguous in the B preshuffled layout
            // because a memory instruction can only read 16 f8 elements at a time.
            return mfma_selector::selected_mfma.k_per_blk == 32 ? 2 : 1;
        else
            return 1;
    }();
    static constexpr index_t KLane =
        mfma_selector::GetKPerXdlops() / mfma_selector::GetK1PerXdlops();
    static constexpr index_t KRepeat = KPerBlock / KLane / (KPack / KGroup);
    static constexpr index_t NLane   = NPerXdl;
    static constexpr index_t NWave   = NPerBlock / NPerXdl / NXdlPerWave;
    // static constexpr index_t NumTokens = 1;
    static constexpr index_t SortedTileSize = MPerBlock;

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

    static constexpr index_t APackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<ADataType>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    static constexpr index_t BPackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    __host__ static auto CalculateGridSize(index_t M, index_t N, index_t K, index_t KBatch)
    {
        const index_t nblock = math::integer_divide_ceil(N, NPerBlock);
        const index_t mblock = math::integer_divide_ceil(M, MPerBlock);
        const index_t gridx  = NSwizzle ? nblock * mblock : nblock;
        const index_t gridy  = NSwizzle ? 1 : mblock;
        const index_t gridz  = KBatch == 1 ? 1 : math::integer_divide_ceil(K, KPerBlock * KBatch);

        return std::make_tuple(gridx, gridy, gridz);
    }

    __host__ __device__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateBN0Shuffled(index_t N)
    {
        return math::integer_divide_ceil(N, NLane);
    }
    __host__ __device__ static auto CalculateBK0Shuffled(index_t K)
    {
        return math::integer_divide_ceil(K, KLane * KPack / KGroup);
    }

    __host__ __device__ static auto CalculateKPadded(index_t K)
    {
        return math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
    }

    __host__ __device__ static auto CalculateAK0Padded(index_t K, index_t K_Batch = 1)
    {
        // auto K_t = K_Batch * KPerBlock;
        // return (K + K_t - 1) / K_t * (KPerBlock / AK1Value);
        return K_Batch == 1 ? K / AK1Value : K_Batch * KPerBlock / AK1Value;
    }

    __host__ __device__ static auto CalculateBK0Padded(index_t K, index_t K_Batch = 1)
    {
        // auto K_t = K_Batch * KPerBlock;
        // return (K + K_t - 1) / K_t * (KPerBlock / BK1Value);
        return K_Batch == 1 ? K / BK1Value : K_Batch * KPerBlock / BK1Value;
    }

    __host__ __device__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        // auto K_t = K_Batch * KPerBlock;
        // return (K + K_t - 1) / K_t * KPerBlock;
        return K_Batch == 1 ? K : K_Batch * KPerBlock;
    }

    __host__ __device__ static auto CalculateKRead(index_t K, index_t K_Batch = 1)
    {
        constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
        // auto K_t                = K_Batch * KReadVec;
        // return (K + K_t - 1) / K_t * KReadVec;
        return K_Batch == 1 ? math::integer_divide_ceil(K, KReadVec) * KReadVec
                            : K_Batch * KPerBlock;
    }

    __host__ __device__ static auto CalculateMBlock(index_t M)
    {
        return math::integer_divide_ceil(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNBlock(index_t N)
    {
        return math::integer_divide_ceil(N, NPerBlock);
    }

    template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
    __host__ __device__ static constexpr auto MakeGemmMmaTileDescriptor(const TileDesc_K0_MN_K1&)
    {
        constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
        constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

        return transform_tensor_descriptor(
            TileDesc_K0_MN_K1{},
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                       make_unmerge_transform(make_tuple(
                           Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
    }

    __host__ __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
        IndexType M, IndexType MPad, IndexType K, IndexType KPad, IndexType StrideA, IndexType AK0)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(M, MPad - M),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(MPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
            return a_grid_desc_ak0_m_ak1;
        }
    }

    __host__ __device__ static auto MakeBGridDescriptor_Preshuffled(index_t N0, index_t K0)
    {
        constexpr index_t MWave           = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t WaveSize        = BlockSize / (MWave * NWave);
        constexpr index_t NkSwizzleNumber = Number<WaveSize * KPack / KGroup>{};
        return make_naive_tensor_descriptor(
            make_tuple(N0 / NWave, NWave, K0, NkSwizzleNumber),
            make_tuple(NWave * K0 * NkSwizzleNumber, K0 * NkSwizzleNumber, NkSwizzleNumber, I1));
    }

    __host__ __device__ static auto MakeBGridDescriptor_BK0_N_BK1(
        index_t K, index_t KPad, index_t N, index_t NPad, index_t StrideB, index_t BK0)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(StrideB, I1));
            }
        }();

        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        static_assert(!(is_same_v<remove_cvref_t<ADataType>, pk_i4_t> &&
                        GemmSpec != GemmSpecialization::Default),
                      "pk_i4_t does not support padding");

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(N, NPad - N),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(NPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(N), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeAMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor<MXdlPerWave, MWaves, MPerXdl>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeBMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        return MakeGemmMmaTileDescriptor<NXdlPerWave, NWave, NPerXdl>(BBlockDesc_BK0_N_BK1{});
    }

    template <typename ELayout>
    __host__ __device__ static auto MakeCGridDescriptor_M_N(
        IndexType M, IndexType MPad, IndexType N, IndexType NPad, IndexType StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        // pad M and N
        return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                           make_tuple(make_right_pad_transform(M, MPad - M),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    template <typename DLayout>
    __host__ __device__ static auto
    MakeDGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, DLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I0));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, DLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I0, StrideC));
            }
        }();

        // pad M and N
        return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                           make_tuple(make_right_pad_transform(M, MPad - M),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    __host__ __device__ static auto MakeDsGridDescriptor_M_N(
        index_t M, index_t MPad, index_t N, index_t NPad, std::array<index_t, NumDTensor> StrideDs)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                return MakeDGridDescriptor_M_N<DLayout>(M, MPad, N, NPad, StrideDs[i]);
            },
            Number<NumDTensor>{});
    }

    template <typename DsGridDesc>
    __device__ static constexpr auto MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const DsGridDesc& ds_grid_desc_m_n, index_t MBlock, index_t NBlock)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ds_grid_desc_m_n[i], MBlock, NBlock);
            },
            Number<NumDTensor>{});
    }

    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N(0, 0, 0, 0, {}))>;

    struct Problem
    {
        __host__ __device__ Problem(index_t NumTokens_,
                                    index_t TopK_,
                                    index_t M_,
                                    index_t N_,
                                    index_t K_,
                                    index_t StrideA_,
                                    index_t StrideB_,
                                    std::array<index_t, NumDTensor> StrideDs_,
                                    index_t StrideC_,
                                    index_t KBatch_)
            : NumTokens{NumTokens_},
              TopK{TopK_},
              M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideDs{StrideDs_},
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
            std::cout << "problem {" << "NumTokens:" << NumTokens << ", " << "TopK:" << TopK << ", "
                      << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "KRead:" << KRead << ", " << "KP:" << KPadded << ", " << "AK0:" << AK0
                      << ", " << "BK0:" << BK0 << ", " << "MBlock: " << MBlock << ", "
                      << "NBlock: " << NBlock << "}" << std::endl;
        }

        index_t NumTokens;
        index_t TopK;
        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        std::array<index_t, NumDTensor> StrideDs;
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

    // Argument
    struct Argument : public tensor_operation::device::BaseArgument, public Problem
    {
        __host__ Argument(const index_t* p_sorted_token_ids_,
                          const index_t* p_sorted_expert_ids_,
                          const index_t* p_max_token_id_,
                          const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          std::array<const void*, NumDTensor> p_ds_grid_,
                          CDataType* p_c_grid_,
                          index_t NumTokens_,
                          index_t TopK_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          std::array<index_t, NumDTensor> StrideDs_,
                          index_t StrideC_,
                          const AScaleType* p_a_scale_grid_,
                          const BScaleType* p_b_scale_grid_,
                          index_t k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CElementwiseOperation c_element_op_)
            : Problem{NumTokens_,
                      TopK_,
                      M_,
                      N_,
                      K_,
                      StrideA_,
                      StrideB_,
                      StrideDs_,
                      StrideC_,
                      k_batch_},
              p_sorted_token_ids{p_sorted_token_ids_},
              p_sorted_expert_ids{p_sorted_expert_ids_},
              p_max_token_id{p_max_token_id_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_ds_grid{},
              p_c_grid{p_c_grid_},
              p_a_scale_grid{p_a_scale_grid_},
              p_b_scale_grid{p_b_scale_grid_},
              a_element_op{a_element_op_},
              b_element_op{b_element_op_},
              c_element_op{c_element_op_}
        {

            // populate pointer, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType_ = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid(i) = static_cast<const DDataType_*>(p_ds_grid_[i]);
            });
        }

        const index_t* p_sorted_token_ids;
        const index_t* p_sorted_expert_ids;
        const index_t* p_max_token_id;
        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        DsGridPointer p_ds_grid;
        CDataType* p_c_grid;

        const AScaleType* p_a_scale_grid;
        const BScaleType* p_b_scale_grid;

        const AElementwiseOperation a_element_op;
        const BElementwiseOperation b_element_op;
        const CElementwiseOperation c_element_op;
    };

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset()
            : a_k_split_offset(0),
              b_k_split_offset(0),
              ascale_k_split_offset(0),
              bscale_k_split_offset(0)
        {
        }

        __device__ SplitKBatchOffset(const Problem& karg, index_t k_id)
        {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset      = k_id * karg.KRead / APackedSize;
                ascale_k_split_offset = math::integer_divide_floor(a_k_split_offset, ScaleBlockK);
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset      = k_id * karg.KRead * karg.StrideA;
                ascale_k_split_offset = math::integer_divide_floor(a_k_split_offset, ScaleBlockK);
            }

            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset      = k_id * karg.KRead * karg.StrideB;
                bscale_k_split_offset = math::integer_divide_floor(b_k_split_offset, ScaleBlockK);
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset      = k_id * karg.KRead * NLane / BPackedSize;
                bscale_k_split_offset = k_id * karg.KRead / ScaleBlockK;
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t ascale_k_split_offset;
        index_t bscale_k_split_offset;
    };

    __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // K0 -> N0/NWave -> NWave -> KLane -> NLane -> KPack
        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<NXdlPerWave>{}, I1, Number<KRepeat>{}, Number<BK1Value>{}));
    }

    using BlockwiseGemmPipe = remove_cvref_t<
        decltype(BlockGemmBlockMoeScaleBPreshufflePipeline_Selector < BlkGemmPipelineVer,
                 BlkGemmPipeSched,
                 BlockSize,
                 ADataType,
                 BDataType,
                 ComputeTypeA,
                 AccDataType,
                 decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch())),
                 decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()),
                 decltype(MakeAMmaTileDescriptor_M0_M1_M2_K(
                     GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch()))),
                 decltype(MakeBMmaTileDescriptor_N0_N1_N2_K(
                     GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1())),
                 ABlockTransferSrcScalarPerVector,
                 BBlockTransferSrcScalarPerVector,
                 MPerBlock,
                 NPerBlock,
                 KPerBlock,
                 ScaleBlockM,
                 ScaleBlockN,
                 ScaleBlockK,
                 MPerXdl,
                 NPerXdl,
                 MXdlPerWave,
                 NXdlPerWave,
                 KPack,
                 IsInputGemm && !IsSplitK > ())>;

    IS_VALID_COMPILATION_PARAMETER_IMPL(CDataType)

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    __host__ static constexpr bool CheckValidity(const Argument& karg)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     !(is_same<tensor_layout::gemm::RowMajor, ALayout>::value))
        {
            if(!(karg.M % MPerBlock == 0))
            {
#if DEBUG_LOG
                std::cout << "Arg M value is not a multiple of MPerBlock! M: " << karg.M << " "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;

#endif // DEBUG_LOG
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
#if DEBUG_LOG
                std::cout << "Arg N value is not a multiple of NPerBlock! N: " << karg.N << " "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;

#endif // DEBUG_LOG
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
#if DEBUG_LOG
                std::cout << "Arg K value is not a multiple of K_Batch * K0PerBlock * K1! K: "
                          << karg.K << " " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
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
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg K (" << karg.K
                          << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                          << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg M (" << karg.M
                          << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                          << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg N (" << karg.N
                          << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                          << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg K (" << karg.K
                          << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                          << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg N (" << karg.N
                          << ") value is not a multiple of "
                             "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                          << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.M % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg M (" << karg.M
                          << ") value is not a multiple of "
                             "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                          << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        // check gridwise gemm pipeline
#if 0
        const auto num_k_loop = karg.AK0 / (KPerBlock / AK1Value);

        if(num_k_loop <= BlockwiseGemmPipe::PrefetchStages)
        {
            return false;
        }
#endif
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

    template <typename CGridDesc>
    __device__ static constexpr auto MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
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

    // return block_id to C matrix tile idx (m0, n0) mapping
    // if arch = gfx942
    // using Block2CTileMapDefault = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock,
    // NPerBlock>;

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(const index_t* p_sorted_token_ids,
                               const index_t* p_sorted_expert_ids,
                               const index_t* p_max_token_id,
                               const ADataType* p_a_grid,
                               const BDataType* p_b_grid,
                               DsGridPointer& p_ds_grid,
                               CDataType* p_c_grid,
                               const AScaleType* p_a_scale_grid,
                               const BScaleType* p_b_scale_grid,
                               void* p_shared,
                               const Problem& problem,
                               AElementwiseOperation a_element_op,
                               BElementwiseOperation b_element_op,
                               CElementwiseOperation c_element_op)
    {
#if defined(__gfx942__) || defined(__gfx950__)
        constexpr auto b_coherence_flag = NonTemporalLoadB
                                              ? AmdBufferCoherenceEnum::WAVE_NT1
                                              : AmdBufferCoherenceEnum::DefaultCoherence;
#else
        constexpr auto b_coherence_flag = AmdBufferCoherenceEnum::DefaultCoherence;
#endif
        ignore              = b_element_op;
        index_t BN0Shuffled = CalculateBN0Shuffled(problem.N * (IsInputGemm && IsSplitK ? 2 : 1));
        index_t BK0Shuffled = CalculateBK0Shuffled(problem.K);
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            IsInputGemm ? problem.NumTokens : problem.NumTokens * problem.TopK,
            problem.MPadded,
            problem.K,
            problem.KPadded,
            problem.StrideA,
            problem.AK0);
        const auto b_grid_desc_bpreshuffled =
            MakeBGridDescriptor_Preshuffled(BN0Shuffled, BK0Shuffled);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N<CLayout>(
            IsInputGemm ? problem.NumTokens * problem.TopK : problem.NumTokens,
            problem.MPadded,
            problem.N * (IsInputGemm && IsSplitK ? 2 : 1),
            problem.NPadded * (IsInputGemm && IsSplitK ? 2 : 1),
            problem.StrideC);

        const auto a_scale_grid_desc_am_ak = make_naive_tensor_descriptor(
            make_tuple(math::integer_divide_ceil(IsInputGemm ? problem.NumTokens
                                                             : problem.NumTokens * problem.TopK,
                                                 ScaleBlockM),
                       math::integer_divide_ceil(problem.K, ScaleBlockK)),
            make_tuple(math::integer_divide_ceil(problem.K, ScaleBlockK), 1));
        const auto b_scale_grid_desc_bn_ak = make_naive_tensor_descriptor(
            make_tuple(math::integer_divide_ceil(problem.N * (IsInputGemm && IsSplitK ? 2 : 1),
                                                 ScaleBlockN),
                       math::integer_divide_ceil(problem.K, ScaleBlockK)),
            make_tuple(math::integer_divide_ceil(problem.K, ScaleBlockK), 1));

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);
        const index_t max_token_id = __builtin_amdgcn_readfirstlane(p_max_token_id[0]);
        // static_assert(NSwizzle == false, "to do fix: need another pr in sorting merged");
        const index_t expert_block_id = NSwizzle ? blockIdx.x / problem.NBlock : blockIdx.y;
        if(expert_block_id * MPerBlock >= max_token_id)
            return;
        const index_t expert_id =
            __builtin_amdgcn_readfirstlane(p_sorted_expert_ids[expert_block_id]);
        const auto block_mn = [&]() -> std::pair<int, int> {
            if constexpr(NSwizzle)
            {
                const index_t ecnt_prefix  = p_max_token_id[1 + expert_id];
                const index_t prefix_block = ecnt_prefix * problem.NBlock;
                const index_t ecnt         = p_max_token_id[2 + expert_id] - ecnt_prefix;
                const index_t expert_swizzle =
                    ecnt > 0 ? ecnt : 1; // p_max_token_id[expert_id + 1]; // 2
                const index_t bid_new = blockIdx.x - prefix_block;
                const index_t nid     = __builtin_amdgcn_readfirstlane(
                    bid_new % 8 + bid_new / (8 * expert_swizzle) * 8);
                const index_t mid =
                    __builtin_amdgcn_readfirstlane(ecnt_prefix + bid_new / 8 % expert_swizzle);
                return {nid, mid};
            }
            else
            {
                return {blockIdx.x, blockIdx.y};
            }
        }();
        const index_t block_n_id = block_mn.first;
        const index_t block_m_id = block_mn.second;
        const index_t token0 =
            __builtin_amdgcn_readfirstlane(p_sorted_token_ids[block_m_id * MPerBlock] & 0xffffff);

        // constexpr auto M0 = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
        constexpr auto AMThreads  = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
        constexpr auto AK0Threads = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I0);
        constexpr auto AK1Threads = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I2);
        constexpr auto AKThreads  = AK0Threads * AK1Threads;
        constexpr auto AMRepeats  = MPerBlock / AMThreads;
        const index_t token_pos   = block_m_id * MPerBlock + threadIdx.x / AKThreads * AMRepeats;

        if(token_pos >= max_token_id || token0 >= problem.NumTokens)
            return;
        StaticallyIndexedArray<IndexType, AMRepeats> gather_offsets;
        static_for<0, AMRepeats, 1>{}([&](auto m0) {
            const index_t fused_token = p_sorted_token_ids[token_pos + m0];
            index_t token_offset      = fused_token & 0xffffff;
            if constexpr(!IsInputGemm)
            {
                token_offset = token_offset * problem.TopK + (fused_token >> 24);
            }
            gather_offsets(m0) = static_cast<IndexType>(token_offset) * problem.K;
        });
        const long_index_t expert_stride = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(problem.N) * problem.K * (IsInputGemm ? 2 : 1));
        const long_index_t expert_scale_stride = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(math::integer_divide_ceil(problem.N, ScaleBlockN)) *
            (IsInputGemm ? 2 : 1) * math::integer_divide_ceil(problem.K, ScaleBlockK));

        // N0, K0, Blocksize*KPack
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NXdlPerWave);

        // When SplitK is enabled, base pointers have been shifted by
        // SplitKBatchOffset in the kernel entry, but buffer descriptor element
        // spaces are still based on full K. Subtract the pointer shift from
        // each element space so the hardware buffer resource doesn't extend
        // beyond the actual tensor allocation.
        const auto splitk_offset = [&]() -> SplitKBatchOffset {
            if constexpr(IsSplitK)
            {
                return SplitKBatchOffset(problem, blockIdx.z);
            }
            else
            {
                return SplitKBatchOffset();
            }
        }();

        assert(a_grid_desc_ak0_m_ak1.GetElementSpaceSize() >= splitk_offset.a_k_split_offset);
        assert(b_grid_desc_bpreshuffled.GetElementSpaceSize() >= splitk_offset.b_k_split_offset);
        assert(a_scale_grid_desc_am_ak.GetElementSpaceSize() >=
               splitk_offset.ascale_k_split_offset);
        assert(b_scale_grid_desc_bn_ak.GetElementSpaceSize() >=
               splitk_offset.bscale_k_split_offset);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize() - splitk_offset.a_k_split_offset);
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global, b_coherence_flag>(
            p_b_grid + static_cast<long_index_t>(expert_id) * expert_stride / BPackedSize,
            b_grid_desc_bpreshuffled.GetElementSpaceSize() - splitk_offset.b_k_split_offset);

        const auto a_scale_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_scale_grid,
            a_scale_grid_desc_am_ak.GetElementSpaceSize() - splitk_offset.ascale_k_split_offset);
        const auto b_scale_grid_buf =
            make_dynamic_buffer<AddressSpaceEnum::Global, b_coherence_flag>(
                p_b_scale_grid + static_cast<long_index_t>(expert_id) * expert_scale_stride,
                b_scale_grid_desc_bn_ak.GetElementSpaceSize() -
                    splitk_offset.bscale_k_split_offset);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        // dummy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        // A matrix blockwise copy
        auto a_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1_gather<
            ThisThreadBlock,
            AElementwiseOperation,
            ck::tensor_operation::element_wise::PassThrough,
            InMemoryDataOperationEnum::Set,
            Sequence<AK0Number, MPerBlock, AK1Number>,
            ABlockTransferThreadClusterLengths_AK0_M_AK1,
            ABlockTransferThreadClusterArrangeOrder,
            ADataType,
            LDSTypeA,
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
            IndexType,
            1,
            BlockwiseGemmPipe::GlobalBufferNum>(a_grid_desc_ak0_m_ak1,
                                                make_multi_index(0, 0, 0),
                                                a_element_op,
                                                a_block_desc_ak0_m_ak1,
                                                make_multi_index(0, 0, 0),
                                                ck::tensor_operation::element_wise::PassThrough{},
                                                gather_offsets);

        // Thread-wise copy
        // K0 -> N0/NWave -> NWave -> KLane -> NLane -> KPack
        auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        auto b_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
            BDataType,
            BDataType,
            decltype(b_grid_desc_bpreshuffled),
            decltype(b_block_desc_bk0_n_bk1),
            Sequence<Number<NXdlPerWave>{}, I1, Number<KRepeat>{}, Number<BK1Value>{}>,
            Sequence<1, 2, 0, 3>,
            3,
            BBlockTransferSrcScalarPerVector,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_grid_desc_bpreshuffled,
                  make_multi_index(n_block_data_idx_on_grid,
                                   get_warp_local_1d_id() % NWave,
                                   0,
                                   KPack / KGroup * (get_thread_local_1d_id() % WarpSize)));

        // LDS allocation for A and B: be careful of alignment
        // Cast after lds
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeA*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, 0, KRepeat, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();
        decltype(c_thread_buf) c_thread_buf_up;

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            problem.KBatch == 1
                ? (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
                      KPerBlock
                : problem.KBatch);
        constexpr index_t ScaleSliceSizeM = MXdlPerWave;
        constexpr index_t ScaleSliceSizeN = math::integer_divide_ceil(NPerBlock, ScaleBlockN);
        constexpr index_t ScaleSliceSizeK = math::integer_divide_ceil(KPerBlock, ScaleBlockK);

        // ScaleSliceSizeK is last dimension in A/B scale for vector memory access
        // ScaleSliceSizeK is first dimension in C scale for packed math
        constexpr auto a_scale_thread_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<ScaleSliceSizeM>{}, Number<ScaleSliceSizeK>{}));

        constexpr index_t MWaves   = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWaves   = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr index_t WaveSize = BlockSize / (MWaves * NWaves);
        auto a_thread_offset       = get_thread_local_1d_id() % MPerXdl +
                               (get_thread_local_1d_id() / WaveSize) / NWaves * MPerXdl;

        constexpr auto b_scale_thread_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<ScaleSliceSizeN>{}, Number<ScaleSliceSizeK>{}));

        constexpr auto c_scale_thread_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<ScaleSliceSizeK>{}, Number<ScaleSliceSizeM>{}, Number<ScaleSliceSizeN>{}));

        // get each thread's offset in the scale tensor
        // A scale
        const index_t token_scale_pos = block_m_id * MPerBlock / ScaleBlockM;

        if(token_scale_pos >= max_token_id || token0 >= problem.NumTokens)
            return;
        StaticallyIndexedArray<index_t, MXdlPerWave> scale_gather_offsets;
        static_for<0, MXdlPerWave, 1>{}([&](auto m0) {
            const index_t fused_token =
                p_sorted_token_ids[token_scale_pos + m0 * MPerXdl * MWaves + a_thread_offset];
            index_t token_offset = fused_token & 0xffffff;
            if constexpr(!IsInputGemm)
            {
                token_offset = token_offset * problem.TopK + (fused_token >> 24);
            }
            scale_gather_offsets(m0) =
                token_offset * math::integer_divide_ceil(problem.K, ScaleBlockK);
        });

        auto a_scale_thread_copy =
            ThreadwiseTensorSliceTransfer_v2_gather<AScaleType,
                                                    AScaleType,
                                                    decltype(a_scale_grid_desc_am_ak),
                                                    decltype(a_scale_thread_desc),
                                                    Sequence<1, ScaleSliceSizeK>,
                                                    Sequence<0, 1>,
                                                    1,
                                                    ScaleSliceSizeK,
                                                    1,
                                                    false,
                                                    MXdlPerWave>(
                a_scale_grid_desc_am_ak, make_multi_index(0, 0), scale_gather_offsets);

        auto b_scale_thread_copy =
            ThreadwiseTensorSliceTransfer_v2<BScaleType,
                                             BScaleType,
                                             decltype(b_scale_grid_desc_bn_ak),
                                             decltype(b_scale_thread_desc),
                                             Sequence<ScaleSliceSizeN, ScaleSliceSizeK>,
                                             Sequence<0, 1>,
                                             1,
                                             ScaleSliceSizeK,
                                             1,
                                             false>(
                b_scale_grid_desc_bn_ak, make_multi_index(block_n_id * NPerBlock / ScaleBlockN, 0));

        // constexpr auto a_scale_thread_slice_copy_step = make_multi_index(0, 1);
        constexpr auto a_scale_thread_slice_copy_step =
            make_tuple(make_multi_index(0, 0), make_multi_index(0, ScaleSliceSizeK));
        constexpr auto b_scale_thread_slice_copy_step = make_multi_index(0, ScaleSliceSizeK);

        constexpr auto NumKBlockPerScale = math::integer_divide_ceil(ScaleBlockK, KPerBlock);
        if constexpr(IsInputGemm && !IsSplitK)
        {
            const BDataType* p_b_grid_up = p_b_grid + expert_stride / 2 / BPackedSize;
            const auto b_grid_buf_up =
                make_dynamic_buffer<AddressSpaceEnum::Global, b_coherence_flag>(
                    p_b_grid_up +
                        static_cast<long_index_t>(expert_id) * expert_stride / BPackedSize,
                    b_grid_desc_bpreshuffled.GetElementSpaceSize());
            auto b_blockwise_copy_up = ThreadwiseTensorSliceTransfer_v2<
                BDataType,
                BDataType,
                decltype(b_grid_desc_bpreshuffled),
                decltype(b_block_desc_bk0_n_bk1),
                Sequence<Number<NXdlPerWave>{}, I1, Number<KRepeat>{}, Number<BK1Value>{}>,
                Sequence<1, 2, 0, 3>,
                3,
                BBlockTransferSrcScalarPerVector,
                BThreadTransferSrcResetCoordinateAfterRun,
                true>(b_grid_desc_bpreshuffled,
                      make_multi_index(n_block_data_idx_on_grid,
                                       get_warp_local_1d_id() % NWave,
                                       0,
                                       KPack / KGroup * (get_thread_local_1d_id() % WarpSize)));
            const BScaleType* p_b_scale_grid_up =
                p_b_scale_grid + expert_scale_stride / 2 / BPackedSize;
            const auto b_scale_grid_buf_up =
                make_dynamic_buffer<AddressSpaceEnum::Global, b_coherence_flag>(
                    p_b_scale_grid_up + static_cast<long_index_t>(expert_id) * expert_scale_stride,
                    b_scale_grid_desc_bn_ak.GetElementSpaceSize());
            auto b_scale_thread_copy_up =
                ThreadwiseTensorSliceTransfer_v2<BScaleType,
                                                 BScaleType,
                                                 decltype(b_scale_grid_desc_bn_ak),
                                                 decltype(b_scale_thread_desc),
                                                 Sequence<ScaleSliceSizeN, ScaleSliceSizeK>,
                                                 Sequence<0, 1>,
                                                 1,
                                                 ScaleSliceSizeK,
                                                 1,
                                                 false>(
                    b_scale_grid_desc_bn_ak,
                    make_multi_index(block_n_id * NPerBlock / ScaleBlockN, 0));

            blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, NumKBlockPerScale, TailNum>(
                a_grid_desc_ak0_m_ak1,
                a_block_desc_ak0_m_ak1,
                a_blockwise_copy,
                a_grid_buf,
                a_block_buf,
                a_block_slice_copy_step,

                b_grid_desc_bpreshuffled,
                b_block_desc_bk0_n_bk1,
                b_blockwise_copy,
                b_blockwise_copy_up,
                b_grid_buf,
                b_grid_buf_up,
                b_block_buf,
                b_block_slice_copy_step,

                c_scale_thread_desc,
                c_thread_buf,
                c_thread_buf_up,

                a_scale_grid_desc_am_ak,
                a_scale_thread_desc,
                a_scale_thread_copy,
                a_scale_grid_buf,
                a_scale_thread_slice_copy_step,

                b_scale_grid_desc_bn_ak,
                b_scale_thread_desc,
                b_scale_thread_copy,
                b_scale_thread_copy_up,
                b_scale_grid_buf,
                b_scale_grid_buf_up,
                b_scale_thread_slice_copy_step,

                num_k_block_main_loop);
        }
        else
        {
            blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, NumKBlockPerScale, TailNum>(
                a_grid_desc_ak0_m_ak1,
                a_block_desc_ak0_m_ak1,
                a_blockwise_copy,
                a_grid_buf,
                a_block_buf,
                a_block_slice_copy_step,

                b_grid_desc_bpreshuffled,
                b_block_desc_bk0_n_bk1,
                b_blockwise_copy,
                b_grid_buf,
                b_block_buf,
                b_block_slice_copy_step,

                c_scale_thread_desc,
                c_thread_buf,

                a_scale_grid_desc_am_ak,
                a_scale_thread_desc,
                a_scale_thread_copy,
                a_scale_grid_buf,
                a_scale_thread_slice_copy_step,

                b_scale_grid_desc_bn_ak,
                b_scale_thread_desc,
                b_scale_thread_copy,
                b_scale_grid_buf,
                b_scale_thread_slice_copy_step,

                num_k_block_main_loop);
        }

        // shuffle C and write out
        // TODO: hacky, fix it!
        // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
            BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I0);
        constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I1);
        constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
        constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
        constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
        constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
        constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
        constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

        static_assert(N0 * N1 * N2 * N3 * N4 == NPerBlock);
        static_assert(M0 * M1 * M2 == MPerBlock);
        static_assert(N4 == 4 || N4 == 8);
        const index_t m1 = get_warp_local_1d_id() / NWave;
        const index_t m2 = threadIdx.x % get_warp_size() % M2;

        float topk_weight;
        static_for<0, MXdlPerWave, 1>{}([&](auto m0) { // MXDLPerWave
            static_for<0, NXdlPerWave, 1>{}([&](auto n0) {
                if constexpr(MulRoutedWeight)
                {
                    const index_t m_pos = block_m_id * MPerBlock + m0 * M1 * M2 + m1 * M2 + m2;
                    topk_weight         = p_ds_grid[I0][m_pos];
                }
                static_for<0, N2, 1>{}([&](auto n2) {     // num_groups_per_blk
                    static_for<0, N4, 1>{}([&](auto n4) { // inst_group_size
                        constexpr index_t c_offset =
                            BlockwiseGemmPipe::GetCThreadDesc().CalculateOffset(
                                make_tuple(m0, n0, n2 * N4 + n4));
                        constexpr auto cidx = Number<c_offset>{};
                        if constexpr(IsInputGemm && !IsSplitK) // gu fusion, elementwise
                        {
                            if constexpr(ActivationOperation == Activation::silu_and_mul)
                            {
                                float gate = c_thread_buf[cidx];
                                float up   = c_thread_buf_up[cidx];
                                if constexpr(MulRoutedWeight)
                                {
                                    gate = gate * topk_weight;
                                    up   = up * topk_weight;
                                }
                                if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
                                {
                                    gate *= 16;
                                    up *= 16;
                                }
                                tensor_operation::element_wise::Silu{}(gate, gate);
                                c_thread_buf(cidx) = gate * up;
                            }
                            else if constexpr(ActivationOperation == Activation::swiglustep_and_mul)
                            {
                                float gate = c_thread_buf[cidx];
                                float up   = c_thread_buf_up[cidx];
                                if constexpr(MulRoutedWeight)
                                {
                                    gate = gate * topk_weight;
                                    up   = up * topk_weight;
                                }
                                if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
                                {
                                    gate *= 16;
                                    up *= 16;
                                }
                                tensor_operation::element_wise::Silu{}(gate, gate);
                                gate               = gate < 7.0f ? gate : 7.0f;
                                up                 = up < 7.0f ? (up > -7.0f ? up : -7.0f) : 7.0f;
                                c_thread_buf(cidx) = gate * up;
                            }
                            else if(ActivationOperation == Activation::gelu_and_mul)
                            {
                                float gate = c_thread_buf[cidx];
                                float up   = c_thread_buf_up[cidx];
                                if constexpr(MulRoutedWeight)
                                {
                                    gate = gate * topk_weight;
                                    up   = up * topk_weight;
                                }
                                if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
                                {
                                    gate *= 16;
                                    up *= 16;
                                }
                                tensor_operation::element_wise::Gelu{}(gate, gate);
                                c_thread_buf(cidx) = gate * up;
                            }
                        }
                        else
                        {
                            if constexpr(MulRoutedWeight)
                            {
                                c_thread_buf(cidx) = c_thread_buf[cidx] * topk_weight;
                            }
                        }
                    });
                });
            });
        });

        auto problemN               = problem.N * (IsInputGemm && IsSplitK ? 2 : 1);
        auto problemNPadded         = problem.NPadded * (IsInputGemm && IsSplitK ? 2 : 1);
        const auto ds_grid_desc_m_n = MakeDsGridDescriptor_M_N(
            problem.M, problem.MPadded, problemN, problemNPadded, problem.StrideDs);
        const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                ds_grid_desc_m_n, problem.MBlock, problem.NBlock);
        Base::template RunMoeEpilogue<CGlobalMemoryDataOperation, true, IsInputGemm, IndexType>(
            blockwise_gemm_pipeline,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            c_thread_buf,
            block_m_id,
            block_n_id,
            p_shared,
            p_sorted_token_ids,
            p_c_grid,
            p_ds_grid,
            c_element_op,
            problem.TopK,
            problemN);
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run_2Lds(const index_t* p_sorted_token_ids,
                                    const index_t* p_sorted_expert_ids,
                                    const index_t* p_max_token_id,
                                    const ADataType* p_a_grid,
                                    const BDataType* p_b_grid,
                                    DsGridPointer& p_ds_grid,
                                    CDataType* p_c_grid,
                                    const AScaleType* p_a_scale_grid,
                                    const BScaleType* p_b_scale_grid,
                                    void* p_shared,
                                    void* p_shared1,
                                    const Problem& problem,
                                    AElementwiseOperation a_element_op,
                                    BElementwiseOperation b_element_op,
                                    CElementwiseOperation c_element_op)
    {
#if defined(__gfx942__) || defined(__gfx950__)
        constexpr auto b_coherence_flag = NonTemporalLoadB
                                              ? AmdBufferCoherenceEnum::WAVE_NT1
                                              : AmdBufferCoherenceEnum::DefaultCoherence;
#else
        constexpr auto b_coherence_flag = AmdBufferCoherenceEnum::DefaultCoherence;
#endif
        ignore                           = b_element_op;
        index_t BN0Shuffled              = CalculateBN0Shuffled(problem.N);
        index_t BK0Shuffled              = CalculateBK0Shuffled(problem.K);
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            IsInputGemm ? problem.NumTokens : problem.NumTokens * problem.TopK,
            problem.MPadded,
            problem.K,
            problem.KPadded,
            problem.StrideA,
            problem.AK0);
        const auto b_grid_desc_bpreshuffled =
            MakeBGridDescriptor_Preshuffled(BN0Shuffled, BK0Shuffled);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N<CLayout>(
            IsInputGemm ? problem.NumTokens * problem.TopK : problem.NumTokens,
            problem.MPadded,
            problem.N * (IsInputGemm && IsSplitK ? 2 : 1),
            problem.NPadded * (IsInputGemm && IsSplitK ? 2 : 1),
            problem.StrideC);

        const auto a_scale_grid_desc_am_ak = make_naive_tensor_descriptor(
            make_tuple(math::integer_divide_ceil(IsInputGemm ? problem.NumTokens
                                                             : problem.NumTokens * problem.TopK,
                                                 ScaleBlockM),
                       math::integer_divide_ceil(problem.K, ScaleBlockK)),
            make_tuple(math::integer_divide_ceil(problem.K, ScaleBlockK), 1));
        const auto b_scale_grid_desc_bn_ak = make_naive_tensor_descriptor(
            make_tuple(math::integer_divide_ceil(problem.N, ScaleBlockN),
                       math::integer_divide_ceil(problem.K, ScaleBlockK)),
            make_tuple(math::integer_divide_ceil(problem.K, ScaleBlockK), 1));
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);
        const index_t max_token_id    = __builtin_amdgcn_readfirstlane(p_max_token_id[0]);
        const index_t expert_block_id = NSwizzle ? blockIdx.x / problem.NBlock : blockIdx.y;
        if(expert_block_id * MPerBlock >= max_token_id)
            return;
        const index_t expert_id =
            __builtin_amdgcn_readfirstlane(p_sorted_expert_ids[expert_block_id]);
        const auto block_mn = [&]() -> std::pair<int, int> {
            if constexpr(NSwizzle)
            {
                const index_t ecnt_prefix    = p_max_token_id[1 + expert_id];
                const index_t prefix_block   = ecnt_prefix * problem.NBlock;
                const index_t ecnt           = p_max_token_id[2 + expert_id] - ecnt_prefix;
                const index_t expert_swizzle = ecnt > 0 ? ecnt : 1;
                const index_t bid_new        = blockIdx.x - prefix_block;
                const index_t nid            = __builtin_amdgcn_readfirstlane(
                    bid_new % 8 + bid_new / (8 * expert_swizzle) * 8);
                const index_t mid =
                    __builtin_amdgcn_readfirstlane(ecnt_prefix + bid_new / 8 % expert_swizzle);
                return {nid, mid};
            }
            else
            {
                return {blockIdx.x, blockIdx.y};
            }
        }();
        const index_t block_n_id = block_mn.first;
        const index_t block_m_id = block_mn.second;

        const index_t token0 =
            __builtin_amdgcn_readfirstlane(p_sorted_token_ids[block_m_id * MPerBlock] & 0xffffff);

        // constexpr auto M0 = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
        constexpr auto AMThreads  = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
        constexpr auto AK0Threads = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I0);
        constexpr auto AK1Threads = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I2);
        constexpr auto AKThreads  = AK0Threads * AK1Threads;
        constexpr auto AMRepeats  = MPerBlock / AMThreads;
        const index_t token_pos   = block_m_id * MPerBlock + threadIdx.x / AKThreads * AMRepeats;

        if(token_pos >= max_token_id || expert_block_id * MPerBlock >= max_token_id ||
           token0 >= problem.NumTokens)
            return;
        StaticallyIndexedArray<IndexType, AMRepeats>
            gather_offsets; //= p_sorted_token_ids[token_pos];
        static_for<0, AMRepeats, 1>{}([&](auto m0) {
            const index_t fused_token = p_sorted_token_ids[token_pos + m0];
            index_t token_offset      = fused_token & 0xffffff;
            if constexpr(!IsInputGemm)
            {
                token_offset = token_offset * problem.TopK + (fused_token >> 24);
            }
            gather_offsets(m0) = static_cast<IndexType>(token_offset) * problem.K;
        });
        const long_index_t expert_stride = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(problem.N) * problem.K * (IsInputGemm ? 2 : 1));
        const long_index_t expert_scale_stride = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(math::integer_divide_ceil(problem.N, ScaleBlockN)) *
            (IsInputGemm ? 2 : 1) * math::integer_divide_ceil(problem.K, ScaleBlockK));
        // N0, K0, Blocksize*KPack
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NXdlPerWave);

        // Same fix as Run(): reduce buffer element spaces by split offset
        const auto splitk_offset = [&]() -> SplitKBatchOffset {
            if constexpr(IsSplitK)
            {
                return SplitKBatchOffset(problem, blockIdx.z);
            }
            else
            {
                return SplitKBatchOffset();
            }
        }();

        assert(a_grid_desc_ak0_m_ak1.GetElementSpaceSize() >= splitk_offset.a_k_split_offset);
        assert(b_grid_desc_bpreshuffled.GetElementSpaceSize() >= splitk_offset.b_k_split_offset);
        assert(a_scale_grid_desc_am_ak.GetElementSpaceSize() >=
               splitk_offset.ascale_k_split_offset);
        assert(b_scale_grid_desc_bn_ak.GetElementSpaceSize() >=
               splitk_offset.bscale_k_split_offset);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize() - splitk_offset.a_k_split_offset);
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global, b_coherence_flag>(
            p_b_grid + static_cast<long_index_t>(expert_id) * expert_stride / BPackedSize,
            b_grid_desc_bpreshuffled.GetElementSpaceSize() - splitk_offset.b_k_split_offset);

        const auto a_scale_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_scale_grid,
            a_scale_grid_desc_am_ak.GetElementSpaceSize() - splitk_offset.ascale_k_split_offset);
        const auto b_scale_grid_buf =
            make_dynamic_buffer<AddressSpaceEnum::Global, b_coherence_flag>(
                p_b_scale_grid + static_cast<long_index_t>(expert_id) * expert_scale_stride,
                b_scale_grid_desc_bn_ak.GetElementSpaceSize() -
                    splitk_offset.bscale_k_split_offset);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        // dummy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        // A matrix blockwise copy
        auto a_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1_gather<
            ThisThreadBlock,
            AElementwiseOperation,
            ck::tensor_operation::element_wise::PassThrough,
            InMemoryDataOperationEnum::Set,
            Sequence<AK0Number, MPerBlock, AK1Number>,
            ABlockTransferThreadClusterLengths_AK0_M_AK1,
            ABlockTransferThreadClusterArrangeOrder,
            ADataType,
            LDSTypeA,
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
            IndexType,
            1,
            BlockwiseGemmPipe::GlobalBufferNum>(a_grid_desc_ak0_m_ak1,
                                                make_multi_index(0, 0, 0),
                                                a_element_op,
                                                a_block_desc_ak0_m_ak1,
                                                make_multi_index(0, 0, 0),
                                                ck::tensor_operation::element_wise::PassThrough{},
                                                gather_offsets);

        // Thread-wise copy
        // K0 -> N0/NWave -> NWave -> KLane -> NLane -> KPack
        auto b_block_buf_ping = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());
        auto b_block_buf_pong = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());
        auto b_block_bufs = make_tuple(b_block_buf_ping, b_block_buf_pong);

        auto b_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
            BDataType,
            BDataType,
            decltype(b_grid_desc_bpreshuffled),
            decltype(b_block_desc_bk0_n_bk1),
            Sequence<Number<NXdlPerWave>{}, I1, Number<KRepeat>{}, Number<BK1Value>{}>,
            Sequence<1, 2, 0, 3>,
            3,
            BBlockTransferSrcScalarPerVector,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_grid_desc_bpreshuffled,
                  make_multi_index(n_block_data_idx_on_grid,
                                   get_warp_local_1d_id() % NWave,
                                   0,
                                   KPack / KGroup * (get_thread_local_1d_id() % WarpSize)));

        // LDS allocation for A and B: be careful of alignment
        // Cast after lds
        auto a_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ADataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());
        auto a_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ADataType*>(p_shared1), a_block_desc_ak0_m_ak1.GetElementSpaceSize());
        auto a_block_bufs = make_tuple(a_block_buf_ping, a_block_buf_pong);

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, 0, KRepeat, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();
        decltype(c_thread_buf) c_thread_buf_up;

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            problem.KBatch == 1
                ? (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
                      KPerBlock
                : problem.KBatch);

        // scale
        constexpr index_t ScaleSliceSizeM = MXdlPerWave;
        constexpr index_t ScaleSliceSizeN = math::integer_divide_ceil(NPerBlock, ScaleBlockN);
        constexpr index_t ScaleSliceSizeK = math::integer_divide_ceil(KPerBlock, ScaleBlockK);

        // ScaleSliceSizeK is last dimension in A/B scale for vector memory access
        // ScaleSliceSizeK is first dimension in C scale for packed math
        constexpr auto a_scale_thread_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<ScaleSliceSizeM>{}, Number<ScaleSliceSizeK>{}));

        constexpr index_t MWaves   = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWaves   = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr index_t WaveSize = BlockSize / (MWaves * NWaves);
        auto a_thread_offset       = get_thread_local_1d_id() % MPerXdl +
                               (get_thread_local_1d_id() / WaveSize) / NWaves * MPerXdl;

        constexpr auto b_scale_thread_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<ScaleSliceSizeN>{}, Number<ScaleSliceSizeK>{}));

        constexpr auto c_scale_thread_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<ScaleSliceSizeK>{}, Number<ScaleSliceSizeM>{}, Number<ScaleSliceSizeN>{}));

        // get each thread's offset in the scale tensor
        // A scale
        const index_t token_scale_pos = block_m_id * MPerBlock / ScaleBlockM;

        if(token_scale_pos >= max_token_id || token0 >= problem.NumTokens)
            return;
        StaticallyIndexedArray<index_t, MXdlPerWave> scale_gather_offsets;
        static_for<0, MXdlPerWave, 1>{}([&](auto m0) {
            const index_t fused_token =
                p_sorted_token_ids[token_scale_pos + m0 * MPerXdl * MWaves + a_thread_offset];
            index_t token_offset = fused_token & 0xffffff;
            if constexpr(!IsInputGemm)
            {
                token_offset = token_offset * problem.TopK + (fused_token >> 24);
            }
            scale_gather_offsets(m0) = static_cast<IndexType>(token_offset) *
                                       math::integer_divide_ceil(problem.K, ScaleBlockK);
        });

        auto a_scale_thread_copy =
            ThreadwiseTensorSliceTransfer_v2_gather<AScaleType,
                                                    AScaleType,
                                                    decltype(a_scale_grid_desc_am_ak),
                                                    decltype(a_scale_thread_desc),
                                                    Sequence<1, ScaleSliceSizeK>,
                                                    Sequence<0, 1>,
                                                    1,
                                                    ScaleSliceSizeK,
                                                    1,
                                                    false,
                                                    MXdlPerWave>(
                a_scale_grid_desc_am_ak, make_multi_index(0, 0), scale_gather_offsets);

        auto b_scale_thread_copy =
            ThreadwiseTensorSliceTransfer_v2<BScaleType,
                                             BScaleType,
                                             decltype(b_scale_grid_desc_bn_ak),
                                             decltype(b_scale_thread_desc),
                                             Sequence<ScaleSliceSizeN, ScaleSliceSizeK>,
                                             Sequence<0, 1>,
                                             1,
                                             ScaleSliceSizeK,
                                             1,
                                             false>(
                b_scale_grid_desc_bn_ak, make_multi_index(block_n_id * NPerBlock / ScaleBlockN, 0));

        // constexpr auto a_scale_thread_slice_copy_step = make_multi_index(0, 1);
        constexpr auto a_scale_thread_slice_copy_step =
            make_tuple(make_multi_index(0, 0), make_multi_index(0, ScaleSliceSizeK));
        constexpr auto b_scale_thread_slice_copy_step = make_multi_index(0, ScaleSliceSizeK);

        constexpr auto NumKBlockPerScale = math::integer_divide_ceil(ScaleBlockK, KPerBlock);
        if constexpr(IsInputGemm && !IsSplitK)
        {
            const BDataType* p_b_grid_up = p_b_grid + expert_stride / 2 / BPackedSize;
            const auto b_grid_buf_up =
                make_dynamic_buffer<AddressSpaceEnum::Global, b_coherence_flag>(
                    p_b_grid_up +
                        static_cast<long_index_t>(expert_id) * expert_stride / BPackedSize,
                    b_grid_desc_bpreshuffled.GetElementSpaceSize());
            auto b_blockwise_copy_up = ThreadwiseTensorSliceTransfer_v2<
                BDataType,
                BDataType,
                decltype(b_grid_desc_bpreshuffled),
                decltype(b_block_desc_bk0_n_bk1),
                Sequence<Number<NXdlPerWave>{}, I1, Number<KRepeat>{}, Number<BK1Value>{}>,
                Sequence<1, 2, 0, 3>,
                3,
                BBlockTransferSrcScalarPerVector,
                BThreadTransferSrcResetCoordinateAfterRun,
                true>(b_grid_desc_bpreshuffled,
                      make_multi_index(n_block_data_idx_on_grid,
                                       get_warp_local_1d_id() % NWave,
                                       0,
                                       KPack / KGroup * (get_thread_local_1d_id() % WarpSize)));
            const BScaleType* p_b_scale_grid_up =
                p_b_scale_grid + expert_scale_stride / 2 / BPackedSize;
            const auto b_scale_grid_buf_up =
                make_dynamic_buffer<AddressSpaceEnum::Global, b_coherence_flag>(
                    p_b_scale_grid_up +
                        static_cast<long_index_t>(expert_id) * expert_scale_stride / BPackedSize,
                    b_scale_grid_desc_bn_ak.GetElementSpaceSize());
            auto b_scale_thread_copy_up =
                ThreadwiseTensorSliceTransfer_v2<BScaleType,
                                                 BScaleType,
                                                 decltype(b_scale_grid_desc_bn_ak),
                                                 decltype(b_scale_thread_desc),
                                                 Sequence<ScaleSliceSizeN, ScaleSliceSizeK>,
                                                 Sequence<0, 1>,
                                                 1,
                                                 ScaleSliceSizeK,
                                                 1,
                                                 false>(
                    b_scale_grid_desc_bn_ak,
                    make_multi_index(block_n_id * NPerBlock / ScaleBlockN, 0));

            blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, NumKBlockPerScale, TailNum>(
                a_grid_desc_ak0_m_ak1,
                a_block_desc_ak0_m_ak1,
                a_blockwise_copy,
                a_grid_buf,
                a_block_bufs,
                a_block_slice_copy_step,
                b_grid_desc_bpreshuffled,
                b_block_desc_bk0_n_bk1,
                b_blockwise_copy,
                b_blockwise_copy_up,
                b_grid_buf,
                b_grid_buf_up,
                b_block_bufs,
                b_block_slice_copy_step,
                c_scale_thread_desc,
                c_thread_buf,
                c_thread_buf_up,
                a_scale_grid_desc_am_ak,
                a_scale_thread_desc,
                a_scale_thread_copy,
                a_scale_grid_buf,
                a_scale_thread_slice_copy_step,
                b_scale_grid_desc_bn_ak,
                b_scale_thread_desc,
                b_scale_thread_copy,
                b_scale_thread_copy_up,
                b_scale_grid_buf,
                b_scale_grid_buf_up,
                b_scale_thread_slice_copy_step,
                num_k_block_main_loop);
        }
        else
        {
            blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, NumKBlockPerScale, TailNum>(
                a_grid_desc_ak0_m_ak1,
                a_block_desc_ak0_m_ak1,
                a_blockwise_copy,
                a_grid_buf,
                a_block_bufs,
                a_block_slice_copy_step,
                b_grid_desc_bpreshuffled,
                b_block_desc_bk0_n_bk1,
                b_blockwise_copy,
                b_grid_buf,
                b_block_bufs,
                b_block_slice_copy_step,
                c_scale_thread_desc,
                c_thread_buf,
                a_scale_grid_desc_am_ak,
                a_scale_thread_desc,
                a_scale_thread_copy,
                a_scale_grid_buf,
                a_scale_thread_slice_copy_step,
                b_scale_grid_desc_bn_ak,
                b_scale_thread_desc,
                b_scale_thread_copy,
                b_scale_grid_buf,
                b_scale_thread_slice_copy_step,
                num_k_block_main_loop);
        }

        // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
            BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I0);
        constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I1);
        constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
        constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
        constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
        constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
        constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
        constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

        static_assert(N0 * N1 * N2 * N3 * N4 == NPerBlock);
        static_assert(M0 * M1 * M2 == MPerBlock);
        static_assert(N4 == 4 || N4 == 8);
        const index_t m1 = get_warp_local_1d_id() / NWave;
        const index_t m2 = threadIdx.x % get_warp_size() % M2;

        float topk_weight;
        static_for<0, MXdlPerWave, 1>{}([&](auto m0) { // MXDLPerWave
            static_for<0, NXdlPerWave, 1>{}([&](auto n0) {
                if constexpr(MulRoutedWeight)
                {
                    const index_t m_pos = block_m_id * MPerBlock + m0 * M1 * M2 + m1 * M2 + m2;
                    topk_weight         = p_ds_grid[I0][m_pos];
                }
                static_for<0, N2, 1>{}([&](auto n2) {     // num_groups_per_blk
                    static_for<0, N4, 1>{}([&](auto n4) { // inst_group_size
                        constexpr index_t c_offset =
                            BlockwiseGemmPipe::GetCThreadDesc().CalculateOffset(
                                make_tuple(m0, n0, n2 * N4 + n4));
                        constexpr auto cidx = Number<c_offset>{};
                        if constexpr(IsInputGemm && !IsSplitK) // gu fusion, elementwise
                        {
                            if constexpr(ActivationOperation == Activation::silu_and_mul)
                            {
                                float gate = c_thread_buf[cidx];
                                float up   = c_thread_buf_up[cidx];
                                if constexpr(MulRoutedWeight)
                                {
                                    gate = gate * topk_weight;
                                    up   = up * topk_weight;
                                }
                                if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
                                {
                                    gate *= 16;
                                    up *= 16;
                                }
                                tensor_operation::element_wise::Silu{}(gate, gate);
                                c_thread_buf(cidx) = gate * up;
                            }
                            else if constexpr(ActivationOperation == Activation::swiglustep_and_mul)
                            {
                                float gate = c_thread_buf[cidx];
                                float up   = c_thread_buf_up[cidx];
                                if constexpr(MulRoutedWeight)
                                {
                                    gate = gate * topk_weight;
                                    up   = up * topk_weight;
                                }
                                if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
                                {
                                    gate *= 16;
                                    up *= 16;
                                }
                                tensor_operation::element_wise::Silu{}(gate, gate);
                                gate               = gate < 7.0f ? gate : 7.0f;
                                up                 = up < 7.0f ? (up > -7.0f ? up : -7.0f) : 7.0f;
                                c_thread_buf(cidx) = gate * up;
                            }
                            else if(ActivationOperation == Activation::gelu_and_mul)
                            {
                                float gate = c_thread_buf[cidx];
                                float up   = c_thread_buf_up[cidx];
                                if constexpr(MulRoutedWeight)
                                {
                                    gate = gate * topk_weight;
                                    up   = up * topk_weight;
                                }
                                if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
                                {
                                    gate *= 16;
                                    up *= 16;
                                }
                                tensor_operation::element_wise::Gelu{}(gate, gate);
                                c_thread_buf(cidx) = gate * up;
                            }
                        }
                        else
                        {
                            if constexpr(MulRoutedWeight)
                            {
                                c_thread_buf(cidx) = c_thread_buf[cidx] * topk_weight;
                            }
                        }
                    });
                });
            });
        });

        // shuffle C and write out
        auto problemN               = problem.N * (IsInputGemm && IsSplitK ? 2 : 1);
        auto problemNPadded         = problem.NPadded * (IsInputGemm && IsSplitK ? 2 : 1);
        const auto ds_grid_desc_m_n = MakeDsGridDescriptor_M_N(
            problem.M, problem.MPadded, problemN, problemNPadded, problem.StrideDs);

        const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                ds_grid_desc_m_n, problem.MBlock, problem.NBlock);
        Base::template RunMoeEpilogue<CGlobalMemoryDataOperation, true, IsInputGemm, IndexType>(
            blockwise_gemm_pipeline,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            c_thread_buf,
            block_m_id,
            block_n_id,
            p_shared,
            p_sorted_token_ids,
            p_c_grid,
            p_ds_grid,
            c_element_op,
            problem.TopK,
            problemN);
    }
};

} // namespace ck
#pragma clang diagnostic pop
