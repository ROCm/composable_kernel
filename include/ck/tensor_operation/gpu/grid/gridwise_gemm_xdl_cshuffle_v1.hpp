// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck {

template <typename GridwiseGemm, bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_xdl_cshuffle_v1(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx11__) || \
    defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        GridwiseGemm::template Run<HasMainKBlockLoop>(
            karg.p_a_grid, karg.p_b_grid, karg.p_c_grid, p_shared, karg);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename GridwiseGemm,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_xdl_cshuffle_v1(const FloatA* __restrict__ p_a_grid,
                                const FloatB* __restrict__ p_b_grid,
                                FloatC* __restrict__ p_c_grid,
                                typename GridwiseGemm::Problem problem)
{
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx11__) || \
    defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        GridwiseGemm::template Run<HasMainKBlockLoop>(
            p_a_grid, p_b_grid, p_c_grid, p_shared, problem);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = problem;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename FloatA,
          typename FloatB,
          typename FloatGemmAcc,
          typename FloatCShuffle,
          typename FloatC,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t NumGemmKPrefetchStage,
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
          LoopScheduler LoopSched,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          typename ComputeTypeA       = FloatC,
          typename ComputeTypeB       = ComputeTypeA>
struct GridwiseGemm_k0mk1_k0nk1_mn_xdl_cshuffle_v1
    : public GridwiseGemm_xdl_cshuffle_base<
          ALayout,
          BLayout,
          CLayout,
          ComputeTypeA,
          ComputeTypeB,
          FloatGemmAcc,
          FloatCShuffle,
          Tuple<>,
          FloatC,
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
          true> // ForceNaiveLayout
{
    using Base = GridwiseGemm_xdl_cshuffle_base<
        ALayout,
        BLayout,
        CLayout,
        ComputeTypeA,
        ComputeTypeB,
        FloatGemmAcc,
        FloatCShuffle,
        Tuple<>,
        FloatC,
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
        true>; // ForceNaiveLayout

    using Base::AK0Number;
    using Base::AK1Number;
    using Base::BK0Number;
    using Base::BK1Number;
    using Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1;
    using Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1;
    using Base::GetSharedMemoryNumberOfByte;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using ThisThreadBlock = typename Base::ThisThreadBlock;

    __host__ static auto CalculateGridSize(index_t M, index_t N)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), 1, 1);
    }

    __host__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_divide_ceil(M, MPerBlock) * MPerBlock;
    }

    __host__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_divide_ceil(N, NPerBlock) * NPerBlock;
    }

    __host__ static auto CalculateKPadded(index_t K)
    {
        return math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
    }

    __host__ static auto CalculateAK0(index_t K)
    {
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding ||
                     GemmSpec == GemmSpecialization::KPadding ||
                     GemmSpec == GemmSpecialization::NKPadding)
        {
            return CalculateKPadded(K) / AK1Value;
        }
        else
        {
            return K / AK1Value;
        }
    }

    __host__ static auto CalculateBK0(index_t K)
    {
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding ||
                     GemmSpec == GemmSpecialization::KPadding ||
                     GemmSpec == GemmSpecialization::MKPadding)
        {
            return CalculateKPadded(K) / BK1Value;
        }
        else
        {
            return K / BK1Value;
        }
    }

    __host__ static auto CalculateMBlock(index_t M)
    {
        return math::integer_divide_floor(M, MPerBlock);
    }

    __host__ static auto CalculateNBlock(index_t N)
    {
        return math::integer_divide_floor(N, NPerBlock);
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

    struct Problem
    {
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         index_t StrideA_,
                         index_t StrideB_,
                         index_t StrideC_)
            : M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
              MPadded{CalculateMPadded(M_)},
              NPadded{CalculateNPadded(N_)},
              KPadded{CalculateKPadded(K_)},
              AK0{CalculateAK0(K_)},
              BK0{CalculateBK0(K_)},
              MBlock{CalculateMBlock(M_)},
              NBlock{CalculateNBlock(N_)}
        {
        }

        __host__ void Print() const
        {
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "KP:" << KPadded << ", " << "AK0:" << AK0 << ", " << "BK0:" << BK0 << ", "
                      << "MBlock: " << MBlock << ", " << "NBlock: " << NBlock << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t MPadded;
        index_t NPadded;
        index_t KPadded;
        index_t AK0;
        index_t BK0;
        index_t MBlock;
        index_t NBlock;
    };

    // Argument
    struct Argument : public tensor_operation::device::BaseArgument, public Problem
    {
        __host__ Argument(const FloatA* p_a_grid_,
                          const FloatB* p_b_grid_,
                          FloatC* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideC_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_}
        {
        }

        const FloatA* p_a_grid;
        const FloatB* p_b_grid;
        FloatC* p_c_grid;
    };

    // FIXME: pass GridwiseGemmPipe as a template arguement into GridwiseGemm
    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
        return ck::tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            FloatC,
            CGlobalMemoryDataOperation>();
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    __host__ static constexpr bool CheckValidity(const Problem& problem)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(problem.M % MPerBlock == 0))
            {
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(problem.N % NPerBlock == 0))
            {
                return false;
            }
        }

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding)
        {
            if(!(CalculateKPadded(problem.K) % AK1Value == 0) ||
               !(CalculateKPadded(problem.K) % BK1Value == 0))
            {
                return false;
            }
        }
        else
        {
            if(!(problem.K % AK1Value == 0) || !(problem.K % BK1Value == 0))
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(problem.K % ABlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }
        else
        {
            if(problem.M % ABlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(problem.N % BBlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }
        else
        {
            if(problem.K % BBlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(problem.N % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }
        }
        else
        {
            if(problem.M % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = (CalculateAK0(problem.K) * AK1Value) / KPerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
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
    using Block2CTileMap = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;

    template <bool HasMainKBlockLoop>
    __device__ static void Run(const FloatA* __restrict__ p_a_grid,
                               const FloatB* __restrict__ p_b_grid,
                               FloatC* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
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

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        const AElementwiseOperation a_element_op{};
        const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{problem.M, problem.N};

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0Number, MPerBlock, AK1Number>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                FloatA,
                                                ComputeTypeA,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
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
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0Number, NPerBlock, BK1Number>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                FloatB,
                                                ComputeTypeB,
                                                decltype(b_grid_desc_bk0_n_bk1),
                                                decltype(b_block_desc_bk0_n_bk1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
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
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check
        constexpr auto lcm_AK1_BK1 = math::lcm(AK1Number, BK1Number);
        constexpr bool is_single_rate_mfma =
            (((is_same<ComputeTypeA, half_t>::value || is_same<ComputeTypeA, bhalf_t>::value) &&
              lcm_AK1_BK1 <= 4) ||
             (is_same<ComputeTypeA, int8_t>::value && lcm_AK1_BK1 <= 8) ||
             ((is_same<ComputeTypeA, f8_t>::value || is_same<ComputeTypeA, bf8_t>::value) &&
              lcm_AK1_BK1 < 32))
                ? true
                : false;
        constexpr auto is_scale_mfma = false;
        constexpr index_t KPack      = math::max(lcm_AK1_BK1,
                                            MfmaSelector<ComputeTypeA,
                                                              MPerXdl,
                                                              NPerXdl,
                                                              ComputeTypeB,
                                                              is_single_rate_mfma,
                                                              is_scale_mfma>::selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            ComputeTypeA,
            ComputeTypeB,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            LoopSched>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ComputeTypeA*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ComputeTypeB*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1Number, 0, 0);

        // gridwise GEMM pipeline
        static_assert(std::is_default_constructible_v<GridwiseGemmPipe>);
        const auto gridwise_gemm_pipeline = GridwiseGemmPipe{};

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_grid_desc_ak0_m_ak1,
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
                                                               blockwise_gemm,
                                                               c_thread_buf,
                                                               num_k_block_main_loop);

        // shuffle C and write out
        Base::template RunEpilogue<CGlobalMemoryDataOperation, false, false>(
            blockwise_gemm,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            c_thread_buf,
            block_work_idx[I0],
            block_work_idx[I1],
            p_shared,
            p_c_grid,
            c_element_op);
    }
};

} // namespace ck

#pragma clang diagnostic pop
