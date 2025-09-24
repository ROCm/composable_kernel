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
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3.hpp"
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
          bool PermuteB>
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

    template <index_t MNRepeat, index_t MNWaves, index_t MNPerWmma, typename BlockDesc>
    __host__ __device__ static constexpr auto MakeWmmaTileDescriptor(const BlockDesc&)
    {
        // K0_MN_K1 -> K0_MNRepeat_MNWaves_KRow_MNPerWmma_K1
        constexpr auto K0 = BlockDesc{}.GetLength(I0);
        constexpr auto K1 = BlockDesc{}.GetLength(I2);
#ifdef __gfx12__
        constexpr auto KRow = I2;
#else
        constexpr auto KRow = I1;
#endif
        return transform_tensor_descriptor(
            BlockDesc{},
            make_tuple(make_unmerge_transform(make_tuple(Number<K0 / KRow>{}, KRow)),
                       make_unmerge_transform(
                           make_tuple(Number<MNRepeat>{}, Number<MNWaves>{}, Number<MNPerWmma>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 3>{}, Sequence<1, 2, 4>{}, Sequence<5>{}));
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

    __host__ __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
        index_t M, index_t MPad, index_t K, index_t KPad, index_t StrideA, index_t AK0)
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
            static_assert(!PermuteA, "PermuteA is not supported");

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

    __host__ __device__ static auto
    MakeAsGridDescriptor_AK0_M_AK1(const index_t M,
                                   const index_t MPad,
                                   const index_t K,
                                   const index_t KPad,
                                   const std::array<index_t, NumATensor>& StrideAs,
                                   const index_t AK0)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeAGridDescriptor_AK0_M_AK1(M, MPad, K, KPad, StrideAs[i], AK0);
            },
            Number<NumATensor>{});
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

        static_assert(!(is_same_v<remove_cvref_t<LDSTypeB>, pk_i4_t> &&
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
            if constexpr(!PermuteB)
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
            else
            {
                // Pre-shuffled Weight
                // BGlobal[K / KPerBlock, N, KPerBlock / K1, K1] -> BTile[K / K1, N, K1]
                constexpr index_t BK01 = KPerBlock / BK1Value;
                const index_t BK0_     = StrideB / BK1Value;
                const index_t BK00     = BK0_ / BK01;

                const auto b_grid_desc_bk00_n_bk01_bk1_permute =
                    make_naive_tensor_descriptor_packed(make_tuple(BK00, N, BK01, BK1Value));

                const auto b_grid_desc_bk0_n_bk1_permute = transform_tensor_descriptor(
                    b_grid_desc_bk00_n_bk01_bk1_permute,
                    make_tuple(make_merge_transform(make_tuple(BK00, BK01)),
                               make_pass_through_transform(make_tuple(N)),
                               make_pass_through_transform(BK1Value)),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                return b_grid_desc_bk0_n_bk1_permute;
            }
        }
    }

    __host__ __device__ static auto
    MakeBsGridDescriptor_BK0_N_BK1(const index_t K,
                                   const index_t KPad,
                                   const index_t N,
                                   const index_t NPad,
                                   const std::array<index_t, NumBTensor>& StrideBs,
                                   const index_t BK0)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeBGridDescriptor_BK0_N_BK1(K, KPad, N, NPad, StrideBs[i], BK0);
            },
            Number<NumBTensor>{});
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto MakeAWmmaTileDescriptor(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWmma);

        return MakeWmmaTileDescriptor<MRepeat, MWaves, MPerWmma>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto MakeBWmmaTileDescriptor(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

        return MakeWmmaTileDescriptor<NRepeat, NWaves, NPerWmma>(BBlockDesc_BK0_N_BK1{});
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

    __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        if constexpr(ABlockLdsExtraM || BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
        {
            // bank conflict when writting the data into LDS, but don't worry, we have whole entire
            // loop to hide it in v4. it may give you some benefit from less valu in compute address
            return make_naive_tensor_descriptor(
                make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                make_tuple(Number<MPerBlock>{} * AK1Number, AK1Number, I1));
        }
        // xor tensor transformation request more unnecessary vgpr usage, would cause register spill
        // in some cases.
        else if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            constexpr index_t LdsSize       = 32 * 4 / KPerBlock / sizeof(LDSTypeA) / APackedSize;
            constexpr auto MLdsLayer        = LdsSize < 1 ? 1 : LdsSize;
            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    AK0Number * Number<MLdsLayer>{}, Number<MPerBlock / MLdsLayer>{}, AK1Number),
                make_tuple(AK1Number, Number<KPerBlock * MLdsLayer>{}, I1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(make_tuple(
                               Number<MPerBlock / MLdsLayer>{}, Number<AK0Number * MLdsLayer>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            constexpr auto a_lds_block_desc_ak0_mldslayer_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(AK0Number, Number<MLdsLayer>{})),
                           make_pass_through_transform(Number<MPerBlock / MLdsLayer>{}),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}));

            constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_ak0_mldslayer_m_ak1,
                make_tuple(make_pass_through_transform(AK0Number),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<MPerBlock / MLdsLayer>{}, Number<MLdsLayer>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return a_lds_block_desc_ak0_m_ak1;
        }
        else // ColumnMajor A
        {
            // kfold and mpair dimension is not always required.
            // more dimension in merge_transform increase the difficulty of generating immarg offset
            // for compiler.
            constexpr auto M0 = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
            constexpr auto M1 = MPerBlock / M0;

            constexpr auto KThreadWrite     = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I0);
            constexpr auto K0PerThreadWrite = AK0Number / KThreadWrite;
            constexpr auto KThreadRead      = 64 / MPerWmma;
            constexpr auto K0PerThreadRead  = AK0Number / KThreadRead;

            constexpr auto kfold = (AK1Number * M0 * sizeof(LDSTypeA) > 128)
                                       ? 1
                                       : 128 / (AK1Number * M0 * sizeof(LDSTypeA));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=mpair<=n0
            constexpr auto mpair = (AK1Number * MPerWmma * sizeof(LDSTypeA) > 128)
                                       ? 1
                                       : ((128 / (AK1Number * MPerWmma * sizeof(LDSTypeA))) > M0
                                              ? M0
                                              : 128 / (AK1Number * MPerWmma * sizeof(LDSTypeA)));

            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * M1>{},
                           Number<kfold * M0 / mpair>{},
                           Number<mpair>{},
                           AK1Number));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * M1>{}, Number<kfold * M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<M1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<M0 / mpair>{}, Number<mpair>{}, Number<M1>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return a_lds_block_desc_ak0_m_ak1;
        }
    }

    __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        if constexpr(BBlockLdsExtraN || BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
        {
            // bank conflict when writting the data into LDS, but don't worry, we have whole entire
            // loop to hide it in v4. it may give you some benefit from less valu in compute address
            return make_naive_tensor_descriptor(
                make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1Number, BK1Number, I1));
        }
        else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
        {
            // NLdsLayer * K0 as logical Bank
            constexpr index_t LdsSize       = 32 * 4 / KPerBlock / sizeof(LDSTypeB) / BPackedSize;
            constexpr index_t NLdsLayer     = LdsSize < 1 ? 1 : LdsSize;
            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    BK0Number * Number<NLdsLayer>{}, Number<NPerBlock / NLdsLayer>{}, BK1Number),
                make_tuple(BK1Number, Number<KPerBlock * NLdsLayer>{}, I1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(make_tuple(
                               Number<NPerBlock / NLdsLayer>{}, Number<BK0Number * NLdsLayer>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(BK0Number, Number<NLdsLayer>{})),
                           make_pass_through_transform(Number<NPerBlock / NLdsLayer>{}),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}));

            constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_bk0_nldslayer_n_bk1,
                make_tuple(make_pass_through_transform(BK0Number),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<NPerBlock / NLdsLayer>{}, Number<NLdsLayer>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return b_lds_block_desc_bk0_n_bk1;
        }
        else // RowMajor B
        {
            constexpr auto N0 = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I1);
            constexpr auto N1 = NPerBlock / N0;

            constexpr auto KThreadWrite     = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I0);
            constexpr auto K0PerThreadWrite = BK0Number / KThreadWrite;
            constexpr auto KThreadRead      = 64 / NPerWmma;
            constexpr auto K0PerThreadRead  = BK0Number / KThreadRead;

            constexpr auto kfold = (BK1Number * N0 * sizeof(LDSTypeB) > 128)
                                       ? 1
                                       : 128 / (BK1Number * N0 * sizeof(LDSTypeB));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=npair<=n0
            constexpr auto npair = (BK1Number * NPerWmma * sizeof(LDSTypeB) > 128)
                                       ? 1
                                       : ((128 / (BK1Number * NPerWmma * sizeof(LDSTypeB))) > N0
                                              ? N0
                                              : 128 / (BK1Number * NPerWmma * sizeof(LDSTypeB)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * N1>{},
                           Number<kfold * N0 / npair>{},
                           Number<npair>{},
                           BK1Number));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * N1>{}, Number<kfold * N0 / npair>{})),
                    make_pass_through_transform(Number<npair>{}),
                    make_pass_through_transform(BK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<N1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<N0 / npair>{})),
                    make_pass_through_transform(Number<npair>{}),
                    make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<N0 / npair>{}, Number<npair>{}, Number<N1>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return b_lds_block_desc_bk0_n_bk1;
        }
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

    using BlockwiseGemmPipe = remove_cvref_t<
        decltype(BlockGemmPipeline_Selector<
                 BlkGemmPipelineVer,
                 BlkGemmPipeSched,
                 BlockSize,
                 LDSTypeA,
                 LDSTypeB,
                 ComputeTypeA,
                 ComputeTypeB,
                 AccDataType,
                 decltype(MakeAWmmaTileDescriptor(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1())),
                 decltype(MakeBWmmaTileDescriptor(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1())),
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
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

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

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A matrix blockwise copy
        // workaround because v7r2 is not as general as v4r1
        auto get_a_blockwise_transfer = [&]() {
            if constexpr(NumATensor > 1)
            {
                const auto idx_as_block_begin = generate_tuple(
                    [&](auto) { return make_multi_index(0, m_block_data_idx_on_grid, 0); },
                    Number<NumATensor>{});

                return ThreadGroupTensorSliceTransfer_v7r2<
                    ThisThreadBlock,
                    AsDataType,
                    Tuple<LDSTypeA>,
                    AGridDesc_AK0_M_K1,
                    decltype(tie(a_block_desc_ak0_m_ak1)),
                    AElementwiseOperation,
                    Sequence<static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
                    Sequence<AK0Number, MPerBlock, AK1Number>,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ABlockTransferSrcAccessOrder,
                    Sequence<1, 0, 2>,
                    ABlockTransferSrcVectorDim,
                    2,
                    ABlockTransferSrcScalarPerVector,
                    ABlockTransferDstScalarPerVector_AK1,
                    uniform_sequence_gen_t<NumATensor, AThreadTransferSrcResetCoordinateAfterRun>,
                    Sequence<true>,
                    BlockwiseGemmPipe::GlobalBufferNum>{as_grid_desc_ak0_m_ak1,
                                                        idx_as_block_begin,
                                                        tie(a_block_desc_ak0_m_ak1),
                                                        make_tuple(make_multi_index(0, 0, 0)),
                                                        a_element_op};
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    AElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<AK0Number, MPerBlock, AK1Number>,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    remove_cvref_t<tuple_element_t<0, AsDataType>>,
                    remove_cvref_t<tuple_element_t<0, AsDataType>>,
                    decltype(as_grid_desc_ak0_m_ak1[I0]),
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
                    BlockwiseGemmPipe::GlobalBufferNum>(
                    as_grid_desc_ak0_m_ak1[I0],
                    make_multi_index(0, m_block_data_idx_on_grid, 0),
                    a_element_op,
                    a_block_desc_ak0_m_ak1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});
            }
        };

        auto a_blockwise_copy = get_a_blockwise_transfer();

        // B matrix blockwise copy
        // workaround because v7r2 is not as general as v4r1
        auto get_b_blockwise_transfer = [&]() {
            if constexpr(NumBTensor > 1)
            {
                const auto idx_bs_block_begin = generate_tuple(
                    [&](auto) { return make_multi_index(0, n_block_data_idx_on_grid, 0); },
                    Number<NumBTensor>{});

                return ThreadGroupTensorSliceTransfer_v7r2<
                    ThisThreadBlock,
                    BsDataType,
                    Tuple<LDSTypeB>,
                    BGridDesc_BK0_N_K1,
                    decltype(tie(b_block_desc_bk0_n_bk1)),
                    BElementwiseOperation,
                    Sequence<static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
                    Sequence<BK0Number, NPerBlock, BK1Number>,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BBlockTransferSrcAccessOrder,
                    Sequence<1, 0, 2>,
                    BBlockTransferSrcVectorDim,
                    2,
                    BBlockTransferSrcScalarPerVector,
                    BBlockTransferDstScalarPerVector_BK1,
                    uniform_sequence_gen_t<NumBTensor, BThreadTransferSrcResetCoordinateAfterRun>,
                    Sequence<true>,
                    BlockwiseGemmPipe::GlobalBufferNum>{bs_grid_desc_bk0_n_bk1,
                                                        idx_bs_block_begin,
                                                        tie(b_block_desc_bk0_n_bk1),
                                                        make_tuple(make_multi_index(0, 0, 0)),
                                                        b_element_op};
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    BElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<BK0Number, NPerBlock, BK1Number>,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    remove_cvref_t<tuple_element_t<0, BsDataType>>,
                    remove_cvref_t<tuple_element_t<0, BsDataType>>,
                    decltype(bs_grid_desc_bk0_n_bk1[I0]),
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
                    BlockwiseGemmPipe::GlobalBufferNum>(
                    bs_grid_desc_bk0_n_bk1[I0],
                    make_multi_index(0, n_block_data_idx_on_grid, 0),
                    b_element_op,
                    b_block_desc_bk0_n_bk1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});
            }
        };

        auto b_blockwise_copy = get_b_blockwise_transfer();

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

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1Number, 0, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (as_grid_desc_ak0_m_ak1[I0].GetLength(I0) * as_grid_desc_ak0_m_ak1[I0].GetLength(I2)) /
            KPerBlock);

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
