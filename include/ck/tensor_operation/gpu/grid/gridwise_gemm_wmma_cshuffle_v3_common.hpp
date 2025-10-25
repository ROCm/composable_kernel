// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <iostream>
#include <ostream>
#endif

#include "ck/utility/env.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

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
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared, splitk_batch_offset, karg);

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
          bool ForceThreadTileTransfer = false> // only needed for convolution (limitation)
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

    static constexpr index_t KPack = math::max(
        math::lcm(AK1Number, BK1Number),
        WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>::selected_wmma
            .k_per_wmma);

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

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

    // Limitations of the current implementation:
    //  - no multiAB
    //  - GemmSpecialization Default
    //  - pipeline v1 because v3 is buggy (fixed in batched gemm gemm implementation)
    // AK1Value == 8 is not really a limitation but a requirement for the method so
    // it will stay
#ifdef __gfx12__
    static constexpr bool IsAWaveTransferApplicable =
        !ForceThreadTileTransfer && NumATensor == 1 && APackedSize == 1 &&
        GemmSpec == tensor_operation::device::GemmSpecialization::Default &&
        BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 && AK1Value == 8;

    static constexpr bool IsBWaveTransferApplicable =
        !ForceThreadTileTransfer && NumBTensor == 1 && BPackedSize == 1 &&
        GemmSpec == tensor_operation::device::GemmSpecialization::Default &&
        BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 && BK1Value == 8;
#else
    static constexpr bool IsAWaveTransferApplicable = false;
    static constexpr bool IsBWaveTransferApplicable = false;
#endif

    static constexpr index_t WaveSize =
        WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>::selected_wmma
            .wave_size;
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
        IsBWaveTransferApplicable,
        ABTransferWaveTiles<BLayout,
                            tensor_layout::gemm::ColumnMajor,
                            LDSTypeB,
                            BlockSize,
                            NPerBlock,
                            KPerBlock,
                            NPerWmma,
                            KPack,
                            BK1Value,
                            WaveSize>,
        ABTransferThreadTiles<BLayout,
                              tensor_layout::gemm::ColumnMajor,
                              LDSTypeB,
                              BlockSize,
                              NPerBlock,
                              KPerBlock,
                              NPerWmma,
                              BK1Value,
                              UseBlockPaddingB,
                              PermuteB,
                              BBlockTransferThreadClusterLengths_BK0_N_BK1,
                              BBlockTransferThreadClusterArrangeOrder,
                              BBlockTransferSrcAccessOrder,
                              BBlockTransferSrcVectorDim,
                              BBlockTransferSrcScalarPerVector,
                              BBlockTransferDstScalarPerVector_BK1,
                              BThreadTransferSrcResetCoordinateAfterRun>>::type;

    static_assert(!(is_same_v<remove_cvref_t<LDSTypeB>, pk_i4_t> &&
                    GemmSpec != tensor_operation::device::GemmSpecialization::Default),
                  "pk_i4_t does not support padding");

    static_assert(!PermuteA, "PermuteA is not supported");

    // return block_id to C matrix tile idx (m0, n0) mapping
    // if arch = gfx942
    using Block2CTileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;
    // using Block2CTileMap = BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>;

    __host__ static auto CalculateGridSize(index_t M, index_t N, index_t KBatch)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), 1, KBatch);
    }

    __host__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ static auto CalculateKPadded(index_t K)
    {
        return math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
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

    __host__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * KPerBlock;
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

    __host__ __device__ static auto
    MakeAsGridDescriptor_AK0_M_AK1(const index_t M,
                                   const index_t MPad,
                                   const index_t K,
                                   const index_t KPad,
                                   const std::array<index_t, NumATensor>& StrideAs,
                                   const index_t AK0)
    {
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;
        constexpr bool padM      = GemmSpec == GemmSpecialization::MKPadding ||
                              GemmSpec == GemmSpecialization::MNKPadding ||
                              GemmSpec == GemmSpecialization::MPadding ||
                              GemmSpec == GemmSpecialization::MNPadding;
        constexpr bool padK = GemmSpec == GemmSpecialization::MKPadding ||
                              GemmSpec == GemmSpecialization::MNKPadding ||
                              GemmSpec == GemmSpecialization::KPadding ||
                              GemmSpec == GemmSpecialization::NKPadding;
        return generate_tuple(
            [&](auto i) {
                const auto base_desc = MakeAGridDescriptor_M_K(M, K, StrideAs[i]);

                return ATransfer::template MakeGridDescriptor<padM, padK>(
                    base_desc, M, MPad, K, KPad, StrideAs[i], AK0);
            },
            Number<NumATensor>{});
    }

    __host__ __device__ static auto
    MakeBsGridDescriptor_BK0_N_BK1(const index_t K,
                                   const index_t KPad,
                                   const index_t N,
                                   const index_t NPad,
                                   const std::array<index_t, NumBTensor>& StrideBs,
                                   const index_t BK0)
    {
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;
        constexpr bool padN      = GemmSpec == GemmSpecialization::NKPadding ||
                              GemmSpec == GemmSpecialization::MNKPadding ||
                              GemmSpec == GemmSpecialization::NPadding ||
                              GemmSpec == GemmSpecialization::MNPadding;
        constexpr bool padK = GemmSpec == GemmSpecialization::NKPadding ||
                              GemmSpec == GemmSpecialization::MNKPadding ||
                              GemmSpec == GemmSpecialization::KPadding ||
                              GemmSpec == GemmSpecialization::MKPadding;
        return generate_tuple(
            [&](auto i) {
                const auto base_desc = MakeBGridDescriptor_N_K(N, K, StrideBs[i]);
                return BTransfer::template MakeGridDescriptor<padN, padK>(
                    base_desc, N, NPad, K, KPad, StrideBs[i], BK0);
            },
            Number<NumBTensor>{});
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
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

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

    static constexpr index_t NumDTensor = DsDataType::Size();

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
    __device__ static constexpr auto MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const DsGridDesc& ds_grid_desc_m_n, index_t MBlock, index_t NBlock)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ds_grid_desc_m_n[i], MBlock, NBlock);
            },
            Number<NumDTensor>{});
    }

    __host__ __device__ static constexpr auto
    // *Caution Here repeat is shuffle repeat
    GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
    {
        constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWmma);
        constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMRepeatPerShuffle * MWaves * MPerWmma>{},
                           I1,
                           Number<CShuffleNRepeatPerShuffle * NWaves * NPerWmma>{}));

        return c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat;
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
                                                           KPack>())>;

    template <typename DEGridDesc>
    __device__ static constexpr auto MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const DEGridDesc& de_grid_desc_m_n, index_t MBlock, index_t NBlock)
    {
        const auto de_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            de_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return de_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Argument>
    __host__ static constexpr bool CheckValidity(const Argument& karg)
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
                    std::cout << "Arg K value is not a multiple of K_Batch * K0PerBlock * K1! K: "
                              << karg.K << " " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
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

    __host__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockHasHotloop(num_loop);
    }

    __host__ static constexpr TailNumber CalculateKBlockLoopTailNum(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockLoopTailNum(num_loop);
    }

    __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = ATransfer::GetBlockDescriptor();
        constexpr auto b_block_desc_bk0_n_bk1 = BTransfer::GetBlockDescriptor();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

        constexpr auto c_block_size =
            c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat
                .GetElementSpaceSize();

        return math::max((a_block_space_size_aligned * sizeof(LDSTypeA) / APackedSize +
                          b_block_space_size_aligned * sizeof(LDSTypeB) / BPackedSize),
                         c_block_size * sizeof(CShuffleDataType));
    }

    template <index_t numElements, typename Type>
    __device__ __forceinline__ static auto get_first_element_workaround(Type& array)
    {
        if constexpr(numElements > 1)
        {
            return array;
        }
        else
        {
            return array[I0];
        }
    }

    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename BScaleStruct,
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
                               BScaleStruct& b_scale_struct)
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

        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});
        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

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
                as_grid_desc_ak0_m_ak1, a_block_desc_ak0_m_ak1, a_element_op, block_m_id);

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BTransfer::template GetBlockTransfer<BGridDesc_BK0_N_K1,
                                                 decltype(b_block_desc_bk0_n_bk1),
                                                 BsDataType,
                                                 BElementwiseOperation,
                                                 BlockwiseGemmPipe::GlobalBufferNum>(
                bs_grid_desc_bk0_n_bk1, b_block_desc_bk0_n_bk1, b_element_op, block_n_id);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        // Cast after lds
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeA*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            reinterpret_cast<LDSTypeB*>(static_cast<char*>(p_shared) + a_block_space_size_aligned *
                                                                           sizeof(LDSTypeA) /
                                                                           APackedSize),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = ATransfer::GetBlockStep();
        constexpr auto b_block_slice_copy_step = BTransfer::GetBlockStep();

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            ATransfer::GetKDimension(as_grid_desc_ak0_m_ak1[I0]) / KPerBlock);

        blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(
            get_first_element_workaround<NumATensor>(as_grid_desc_ak0_m_ak1),
            a_block_desc_ak0_m_ak1,
            a_blockwise_copy,
            get_first_element_workaround<NumATensor>(as_grid_buf),
            a_block_buf,
            a_block_slice_copy_step,
            get_first_element_workaround<NumBTensor>(bs_grid_desc_bk0_n_bk1),
            b_block_desc_bk0_n_bk1,
            b_blockwise_copy,
            get_first_element_workaround<NumBTensor>(bs_grid_buf),
            b_block_buf,
            b_block_slice_copy_step,
            c_thread_buf,
            b_scale_struct,
            num_k_block_main_loop,
            num_k_block_per_scale);

        // shuffle C and write out
        {
            // C mapping in single thread.
            constexpr auto c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
                blockwise_gemm_pipeline
                    .GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

            // C mapping in single block
            constexpr auto
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp =
                    blockwise_gemm_pipeline
                        .GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

            constexpr auto MWave =
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                    .GetLength(I1);
            constexpr auto MSubGroup =
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                    .GetLength(I2);
            constexpr auto NWave =
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                    .GetLength(I4);
            constexpr auto NThreadPerSubGroup =
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                    .GetLength(I5);
            constexpr auto MAccVgprs =
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                    .GetLength(I6);

            // LDS descriptor, shuffle and write out in MRepeat x NRepeat times
            constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
                GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<CShuffleDataType*>(p_shared),
                c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat
                    .GetElementSpaceSize());

            constexpr auto
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
                    transform_tensor_descriptor(
                        c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                        make_tuple(
                            make_freeze_transform(I0),
                            make_unmerge_transform(make_tuple(
                                Number<CShuffleMRepeatPerShuffle>{}, // MRepeat per shuffle repeat
                                MWave,                               // MWave
                                MSubGroup, // MSubGroup * MAccVgprs = MPerWmma
                                MAccVgprs)),
                            make_freeze_transform(I0),
                            make_unmerge_transform(make_tuple(
                                Number<CShuffleNRepeatPerShuffle>{}, // NRepeat per shuffle repeat
                                NWave,                               // NWave
                                NThreadPerSubGroup))), // NThreadPerSubGroup = NPerWmma
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                        make_tuple(Sequence<>{},
                                   Sequence<0, 1, 2, 6>{},
                                   Sequence<>{},
                                   Sequence<3, 4, 5>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm_pipeline.CalculateCThreadOriginDataIndex(I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor =
                make_single_stage_tensor_adaptor(make_tuple(make_merge_transform(make_tuple(
                                                     MRepeat, MWave, MSubGroup, MAccVgprs))),
                                                 make_tuple(Sequence<0, 1, 2, 3>{}),
                                                 make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor
                    .CalculateBottomIndex(make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor =
                make_single_stage_tensor_adaptor(make_tuple(make_merge_transform(make_tuple(
                                                     NRepeat, NWave, NThreadPerSubGroup))),
                                                 make_tuple(Sequence<0, 1, 2>{}),
                                                 make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor
                    .CalculateBottomIndex(make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds = ThreadwiseTensorSliceTransfer_v1r3<
                AccDataType,
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
                generate_tie([&](auto i) -> const auto& // return type should be reference
                             { return ds_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                             Number<NumDTensor>{}));

            // tuple of reference to C/Ds tensor buffers
            const auto c_ds_buf_refs = concat_tuple_of_reference(
                tie(c_shuffle_block_buf),
                generate_tie([&](auto i) -> const auto& // return type should be reference
                             { return ds_grid_buf[i]; },
                             Number<NumDTensor>{}));

            // tuple of starting index of C/Ds blockwise copy
            const auto idx_c_ds_block_begin = container_concat(
                make_tuple(make_multi_index(0, 0, 0, 0)),
                generate_tuple([&](auto) { return make_multi_index(block_m_id, 0, block_n_id, 0); },
                               Number<NumDTensor>{}));

            // blockwise copy which loads C from LDS, D from global, applies elementwise
            // operation and stores result E to global
            auto cde_shuffle_block_copy_lds_and_global = ThreadGroupTensorSliceTransfer_v7r3<
                ThisThreadBlock, // ThreadGroup
                decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType{})),
                Tuple<EDataType>,
                decltype(c_ds_desc_refs),
                decltype(tie(e_grid_desc_mblock_mperblock_nblock_nperblock)),
                CDEElementwiseOperation,                                    // ElementwiseOperation,
                Sequence<static_cast<index_t>(EGlobalMemoryDataOperation)>, // DstInMemOps,
                Sequence<1,
                         CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                         1,
                         CShuffleNRepeatPerShuffle * NWave * NPerWmma>, // BlockSliceLengths,
                CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>,                    // ThreadClusterArrangeOrder,
                Sequence<0, 1, 2, 3>,                    // SrcDimAccessOrder,
                Sequence<0, 1, 2, 3>,                    // DstDimAccessOrder,
                3,                                       // SrcVectorDim,
                3,                                       // DstVectorDim,
                CDEShuffleBlockTransferScalarPerVectors, // SrcScalarPerVectors
                EShuffleBlockTransferScalarPerVector,    // DstScalarPerVector
                sequence_merge_t<
                    Sequence<true>,
                    uniform_sequence_gen_t<NumDTensor,
                                           false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
                Sequence<false>>                    // ThreadTransferDstResetCoordinateAfterRunFlags
                {c_ds_desc_refs,
                 idx_c_ds_block_begin,
                 tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                 make_tuple(make_multi_index(block_m_id, 0, block_n_id, 0)),
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
                c_thread_copy_vgpr_to_lds.Run(
                    c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                    sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                    c_thread_buf,
                    c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                    c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block loads its C data from LDS, D from global, applies elementwise
                // operation and stores result E to global
                cde_shuffle_block_copy_lds_and_global.Run(
                    c_ds_desc_refs,
                    c_ds_buf_refs,
                    tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                    tie(e_grid_buf));

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto cde_global_step = sfc_cde_global.GetForwardStep(access_id);
                    // move on Ds
                    static_for<0, NumDTensor, 1>{}([&](auto i) {
                        cde_shuffle_block_copy_lds_and_global.MoveSrcSliceWindow(
                            c_ds_desc_refs, i + I1, cde_global_step);
                    });

                    // move on E
                    cde_shuffle_block_copy_lds_and_global.MoveDstSliceWindow(
                        tie(e_grid_desc_mblock_mperblock_nblock_nperblock), cde_global_step);
                }
            });
        }
    }
};

} // namespace ck
