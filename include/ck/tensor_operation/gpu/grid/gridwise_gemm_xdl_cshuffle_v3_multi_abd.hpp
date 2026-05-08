// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_selector.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

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
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_gemm_xdl_cshuffle_v3(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_as_grid,
            karg.p_bs_grid,
            karg.p_ds_grid,
            karg.p_c_grid,
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
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_gemm_xdl_cshuffle_v3_2lds(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        // Pass two lds pointer is the key to tell compiler that ds_read/write
        // operate on different lds chunk at same time without order dependecy
        __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];
        __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_as_grid,
            karg.p_bs_grid,
            karg.p_ds_grid,
            karg.p_c_grid,
            p_shared_0,
            p_shared_1,
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
          typename CLayout,
          typename AsDataType,
          typename BsDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
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
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v4,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA>
struct GridwiseGemm_xdl_cshuffle_v3
    : public GridwiseGemm_xdl_cshuffle_base<
          ALayout,
          BLayout,
          CLayout,
          ComputeTypeA,
          ComputeTypeB,
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
          Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
          ComputeTypeA,
          ComputeTypeB,
          false> // ForceNaiveLayout
{
    using Base = GridwiseGemm_xdl_cshuffle_base<
        ALayout,
        BLayout,
        CLayout,
        ComputeTypeA,
        ComputeTypeB,
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
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        ComputeTypeA,
        ComputeTypeB,
        false>; // ForceNaiveLayout

    using Base::AK0Number;
    using Base::AK1Number;
    using Base::BK0Number;
    using Base::BK1Number;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using ThisThreadBlock = typename Base::ThisThreadBlock;
    using Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1;
    using Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1;
    using Base::NumDTensor;

    using LDSTypeA = ComputeTypeA;
    using LDSTypeB = ComputeTypeB;

    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();

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

    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    using AsGridPointer = decltype(MakeAsGridPointer());
    using BsGridPointer = decltype(MakeBsGridPointer());
    using DsGridPointer = decltype(MakeDsGridPointer());

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
                               ComputeTypeB,
                               is_single_rate_mfma,
                               is_scale_mfma>::selected_mfma.k_per_blk);

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

    __host__ __device__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
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

    __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
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

    __device__ static auto MakeBGridDescriptor_BK0_N_BK1(
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
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

        return MakeGemmMmaTileDescriptor<NXdlPerWave, NWaves, NPerXdl>(BBlockDesc_BK0_N_BK1{});
    }

    __host__ __device__ static auto
    MakeCGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

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
    }

    __host__ __device__ static auto MakeDsGridDescriptor_M_N(
        index_t M, index_t MPad, index_t N, index_t NPad, std::array<index_t, NumDTensor> StrideDs)
    {
        return generate_tuple(
            [&](auto i) { return MakeCGridDescriptor_M_N(M, MPad, N, NPad, StrideDs[i]); },
            Number<NumDTensor>{});
    }

    struct Problem
    {
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         std::array<index_t, NumATensor> StrideAs_,
                         std::array<index_t, NumBTensor> StrideBs_,
                         std::array<index_t, NumDTensor> StrideDs_,
                         index_t StrideC_,
                         index_t KBatch_)
            : M{M_},
              N{N_},
              K{K_},
              StrideAs{StrideAs_},
              StrideBs{StrideBs_},
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
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "MP:" << MPadded << ", " << "NP:" << NPadded << ", " << "KRead:" << KRead
                      << ", " << "KP:" << KPadded << ", " << "AK0:" << AK0 << ", " << "BK0:" << BK0
                      << ", " << "MBlock: " << MBlock << ", " << "NBlock: " << NBlock << "}"
                      << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;

        std::array<index_t, NumATensor> StrideAs;
        std::array<index_t, NumBTensor> StrideBs;
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
        __host__ Argument(std::array<const void*, NumATensor> p_as_grid_,
                          std::array<const void*, NumBTensor> p_bs_grid_,
                          std::array<const void*, NumDTensor> p_ds_grid_,
                          void* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          std::array<index_t, NumATensor> StrideAs_,
                          std::array<index_t, NumBTensor> StrideBs_,
                          std::array<index_t, NumDTensor> StrideDs_,
                          index_t StrideC_,
                          index_t k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CElementwiseOperation c_element_op_)
            : Problem{M_, N_, K_, StrideAs_, StrideBs_, StrideDs_, StrideC_, k_batch_},
              p_as_grid{},
              p_bs_grid{},
              p_ds_grid{},
              p_c_grid{static_cast<CDataType*>(p_c_grid_)},
              a_element_op{a_element_op_},
              b_element_op{b_element_op_},
              c_element_op{c_element_op_}

        {
            // populate pointer, desc for As
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;

                // A pointer
                p_as_grid(i) = static_cast<const ADataType_*>(p_as_grid_[i]);
            });

            // populate pointer, desc for Bs
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;

                // B pointer
                p_bs_grid(i) = static_cast<const BDataType_*>(p_bs_grid_[i]);
            });

            // populate pointer, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType_ = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid(i) = static_cast<const DDataType_*>(p_ds_grid_[i]);
            });
        }

        AsGridPointer p_as_grid;
        BsGridPointer p_bs_grid;
        DsGridPointer p_ds_grid;
        CDataType* p_c_grid;

        const AElementwiseOperation a_element_op;
        const BElementwiseOperation b_element_op;
        const CElementwiseOperation c_element_op;
    };

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(Argument& karg)
        {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = blockIdx.z * karg.KRead;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = blockIdx.z * karg.KRead * karg.M;
            }

            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = blockIdx.z * karg.KRead * karg.N;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = blockIdx.z * karg.KRead;
            }

            if(blockIdx.z < static_cast<uint32_t>(karg.KBatch - 1))
            {
                karg.K = karg.KRead;
            }
            else
            {
                karg.K = karg.K - karg.KRead * (karg.KBatch - 1);
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
    };

    using BlockwiseGemmPipe = remove_cvref_t<
        decltype(BlockGemmPipeline_Selector<
                 BlkGemmPipelineVer,
                 BlkGemmPipeSched,
                 BlockSize,
                 LDSTypeA,
                 LDSTypeB,
                 ComputeTypeA,
                 AccDataType,
                 decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch())),
                 decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch())),
                 decltype(MakeAMmaTileDescriptor_M0_M1_M2_K(
                     GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch()))),
                 decltype(MakeBMmaTileDescriptor_N0_N1_N2_K(
                     GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch()))),
                 ABlockTransferSrcScalarPerVector,
                 BBlockTransferSrcScalarPerVector,
                 MPerBlock,
                 NPerBlock,
                 KPerBlock,
                 MPerXdl,
                 NPerXdl,
                 MXdlPerWave,
                 NXdlPerWave,
                 KPack>())>;

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
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
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
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
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

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of "
                                 "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                              << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.M % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of "
                                 "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                              << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
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

    // return block_id to C matrix tile idx (m0, n0) mapping
    // if arch = gfx942
    using Block2CTileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;
    // using Block2CTileMap = BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>;

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(AsGridPointer& p_as_grid,
                               BsGridPointer& p_bs_grid,
                               DsGridPointer& p_ds_grid,
                               CDataType* p_c_grid,
                               void* p_shared,
                               const Problem& problem,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CElementwiseOperation& c_element_op)
    {
        // std::array<index_t, NumATensor> StrideAs = {problem.StrideA};
        // std::array<index_t, NumBTensor> StrideBs = {problem.StrideB};

        // AsGridPointer p_as_grid;
        // BsGridPointer p_bs_grid;
        // DsGridPointer p_ds_grid;

        const auto as_grid_desc_ak0_m_ak1 = MakeAsGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideAs, problem.AK0);
        const auto bs_grid_desc_bk0_n_bk1 = MakeBsGridDescriptor_BK0_N_BK1(
            problem.K, problem.KPadded, problem.N, problem.NPadded, problem.StrideBs, problem.BK0);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);

        const auto ds_grid_desc_m_n = MakeDsGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideDs);

        const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                ds_grid_desc_m_n, problem.MBlock, problem.NBlock);

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

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

#if 0
        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
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
                                                BlockwiseGemmPipe::GlobalBufferNum>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});
#else
        const auto idx_as_block_begin =
            generate_tuple([&](auto) { return make_multi_index(0, m_block_data_idx_on_grid, 0); },
                           Number<NumATensor>{});

        auto a_blockwise_copy = ThreadGroupTensorSliceTransfer_v7r2<
            ThisThreadBlock,
            AsDataType,
            Tuple<LDSTypeA>,
            decltype(as_grid_desc_ak0_m_ak1),
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
#endif

#if 0
        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
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
                                                BlockwiseGemmPipe::GlobalBufferNum>(
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});
#else
        const auto idx_bs_block_begin =
            generate_tuple([&](auto) { return make_multi_index(0, n_block_data_idx_on_grid, 0); },
                           Number<NumBTensor>{});

        auto b_blockwise_copy = ThreadGroupTensorSliceTransfer_v7r2<
            ThisThreadBlock,
            BsDataType,
            Tuple<LDSTypeB>,
            decltype(bs_grid_desc_bk0_n_bk1),
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

#endif

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        // Cast after lds
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeA*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeB*>(p_shared) +
                a_block_space_size_aligned * sizeof(LDSTypeA) / sizeof(LDSTypeB),
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

        blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(as_grid_desc_ak0_m_ak1,
                                                                         a_block_desc_ak0_m_ak1,
                                                                         a_blockwise_copy,
                                                                         as_grid_buf,
                                                                         a_block_buf,
                                                                         a_block_slice_copy_step,
                                                                         bs_grid_desc_bk0_n_bk1,
                                                                         b_block_desc_bk0_n_bk1,
                                                                         b_blockwise_copy,
                                                                         bs_grid_buf,
                                                                         b_block_buf,
                                                                         b_block_slice_copy_step,
                                                                         c_thread_buf,
                                                                         num_k_block_main_loop);

        // shuffle C and write out
        Base::template RunMultiDEpilogue<CGlobalMemoryDataOperation, false, false, false>(
            blockwise_gemm_pipeline,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            c_thread_buf,
            block_m_id,
            block_n_id,
            p_shared,
            p_ds_grid,
            p_c_grid,
            c_element_op);
    }

#if 1
    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run_2Lds(AsGridPointer& p_as_grid,
                                    BsGridPointer& p_bs_grid,
                                    DsGridPointer& p_ds_grid,
                                    CDataType* p_c_grid,
                                    void* p_shared_0,
                                    void* p_shared_1,
                                    const Problem& problem,
                                    const AElementwiseOperation& a_element_op,
                                    const BElementwiseOperation& b_element_op,
                                    const CElementwiseOperation& c_element_op)
    {
        const auto as_grid_desc_ak0_m_ak1 = MakeAsGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideAs, problem.AK0);
        const auto bs_grid_desc_bk0_n_bk1 = MakeBsGridDescriptor_BK0_N_BK1(
            problem.K, problem.KPadded, problem.N, problem.NPadded, problem.StrideBs, problem.BK0);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);

        const auto ds_grid_desc_m_n = MakeDsGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideDs);

        const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                ds_grid_desc_m_n, problem.MBlock, problem.NBlock);

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

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

#if 0
        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
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
                                                BlockwiseGemmPipe::GlobalBufferNum>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});
#else
        const auto idx_as_block_begin =
            generate_tuple([&](auto) { return make_multi_index(0, m_block_data_idx_on_grid, 0); },
                           Number<NumATensor>{});

        auto a_blockwise_copy = ThreadGroupTensorSliceTransfer_v7r2<
            ThisThreadBlock,
            AsDataType,
            Tuple<LDSTypeA>,
            decltype(as_grid_desc_ak0_m_ak1),
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
            uniform_sequence_gen_t<NumATensor, false>,
            Sequence<true>,
            BlockwiseGemmPipe::GlobalBufferNum>{as_grid_desc_ak0_m_ak1,
                                                idx_as_block_begin,
                                                tie(a_block_desc_ak0_m_ak1),
                                                make_tuple(make_multi_index(0, 0, 0)),
                                                a_element_op};

#endif

#if 0
        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
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
                                                BlockwiseGemmPipe::GlobalBufferNum>(
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});
#else
        const auto idx_bs_block_begin =
            generate_tuple([&](auto) { return make_multi_index(0, n_block_data_idx_on_grid, 0); },
                           Number<NumBTensor>{});

        auto b_blockwise_copy = ThreadGroupTensorSliceTransfer_v7r2<
            ThisThreadBlock,
            BsDataType,
            Tuple<LDSTypeB>,
            decltype(bs_grid_desc_bk0_n_bk1),
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
            uniform_sequence_gen_t<NumBTensor, false>,
            Sequence<true>,
            BlockwiseGemmPipe::GlobalBufferNum>{bs_grid_desc_bk0_n_bk1,
                                                idx_bs_block_begin,
                                                tie(b_block_desc_bk0_n_bk1),
                                                make_tuple(make_multi_index(0, 0, 0)),
                                                b_element_op};
#endif

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeA*>(p_shared_0), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeB*>(p_shared_0) +
                a_block_space_size_aligned * sizeof(LDSTypeA) / sizeof(LDSTypeB),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        auto a_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeA*>(p_shared_1), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeB*>(p_shared_1) +
                a_block_space_size_aligned * sizeof(LDSTypeA) / sizeof(LDSTypeB),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        auto a_block_bufs = make_tuple(a_block_buf_ping, a_block_buf_pong);
        auto b_block_bufs = make_tuple(b_block_buf_ping, b_block_buf_pong);

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1Number, 0, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (as_grid_desc_ak0_m_ak1[I0].GetLength(I0) * as_grid_desc_ak0_m_ak1[I0].GetLength(I2)) /
            KPerBlock);

        blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(as_grid_desc_ak0_m_ak1,
                                                                         a_block_desc_ak0_m_ak1,
                                                                         a_blockwise_copy,
                                                                         as_grid_buf,
                                                                         a_block_bufs,
                                                                         a_block_slice_copy_step,
                                                                         bs_grid_desc_bk0_n_bk1,
                                                                         b_block_desc_bk0_n_bk1,
                                                                         b_blockwise_copy,
                                                                         bs_grid_buf,
                                                                         b_block_bufs,
                                                                         b_block_slice_copy_step,
                                                                         c_thread_buf,
                                                                         num_k_block_main_loop);

        // shuffle C and write out
        Base::template RunMultiDEpilogue<CGlobalMemoryDataOperation, false, false, false>(
            blockwise_gemm_pipeline,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            c_thread_buf,
            block_m_id,
            block_n_id,
            p_shared_0,
            p_ds_grid,
            p_c_grid,
            c_element_op);
    }
#endif
};

} // namespace ck
#pragma clang diagnostic pop
