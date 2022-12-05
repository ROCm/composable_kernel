// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_welford_second_half_layernorm2d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "device_base.hpp"

namespace ck {

template <typename GridwiseGemmWelford,
          typename ABDataType,
          typename DsPointer,
          typename EDataType,
          typename MeanDataType,
          typename VarDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename MeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock,
          typename Block2ETileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_multiple_d_welford_first_half_xdl_cshuffle(
            const ABDataType* __restrict__ p_a_grid,
            const ABDataType* __restrict__ p_b_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            MeanDataType* __restrict__ p_welford_mean_grid,
            VarDataType* __restrict__ p_welford_var_grid,
            int32_t* __restrict__ p_welford_count_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock,
            const MeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock
                mean_var_count_grid_desc_mblock_mperblock_nblock,
            const Block2ETileMap block_2_etile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemmWelford::GetSharedMemoryNumberOfByte()];

    GridwiseGemmWelford::template Run<HasMainKBlockLoop>(
        p_a_grid,
        p_b_grid,
        p_ds_grid,
        p_e_grid,
        p_welford_mean_grid,
        p_welford_var_grid,
        p_welford_count_grid,
        p_shared,
        a_element_op,
        b_element_op,
        cde_element_op,
        a_grid_desc_ak0_m_ak1,
        b_grid_desc_bk0_n_bk1,
        ds_grid_desc_mblock_mperblock_nblock_nperblock,
        e_grid_desc_mblock_mperblock_nblock_nperblock,
        mean_var_count_grid_desc_mblock_mperblock_nblock,
        block_2_etile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = p_welford_mean_grid;
    ignore = p_welford_var_grid;
    ignore = p_welford_count_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = mean_var_count_grid_desc_mblock_mperblock_nblock;
    ignore = block_2_etile_map;
#endif
}

template <typename GridwiseWelfordLayernorm,
          typename EDataType,
          typename HDataType,
          typename MeanDataType,
          typename VarDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename EHGridDesc_M_N,
          typename MeanVarCountGridDesc_M_NBlock,
          typename GammaBetaGridDesc_N,
          typename HElementwiseOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_welford_layernorm2d_second_half(
            const EDataType* __restrict__ p_e_grid,
            const MeanDataType* __restrict__ p_in_welford_mean_grid,
            const VarDataType* __restrict__ p_in_welford_var_grid,
            const int32_t* __restrict__ p_in_welford_count_grid,
            const GammaDataType* __restrict__ p_gamma_grid,
            const BetaDataType* __restrict__ p_beta_grid,
            HDataType* __restrict__ p_h_grid,
            const EHGridDesc_M_N e_grid_desc_m_n,
            const EHGridDesc_M_N h_grid_desc_m_n,
            const MeanVarCountGridDesc_M_NBlock mean_var_count_grid_desc_m_nblock,
            const GammaBetaGridDesc_N gamma_grid_desc_n,
            const GammaBetaGridDesc_N beta_grid_desc_n,
            index_t numMeanVarCountBlockTileIteration_N,
            index_t numNormBlockTileIteration_N,
            ComputeDataType epsilon,
            HElementwiseOperation h_element_op)
{
    GridwiseWelfordLayernorm::Run(p_e_grid,
                                  p_in_welford_mean_grid,
                                  p_in_welford_var_grid,
                                  p_in_welford_count_grid,
                                  p_gamma_grid,
                                  p_beta_grid,
                                  p_h_grid,
                                  e_grid_desc_m_n,
                                  h_grid_desc_m_n,
                                  mean_var_count_grid_desc_m_nblock,
                                  gamma_grid_desc_n,
                                  beta_grid_desc_n,
                                  numMeanVarCountBlockTileIteration_N,
                                  numNormBlockTileIteration_N,
                                  epsilon,
                                  h_element_op);
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A[M, K]
//   input : B[N, K]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   output : H[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
//   H = layernorm(E)
// Assume:
//   D0, D1, ... and E have the same layout
//   Calculate mean & variance along N dimension in layernorm(E)
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename HLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename HDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename HElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename PostShuffleThreadClusterSize_M_N,
          index_t PostShuffleScalarPerVector,
          typename LayernormThreadClusterSize_M_N,
          typename LayernormThreadSliceSize_M_N,
          index_t LayernormESrcHDstVectorDim,
          index_t LayernormESrcVectorSize,
          index_t LayernormHDstVectorSize,
          index_t LayernormGammaSrcVectorSize,
          index_t LayernormBetaSrcVectorSize,
          index_t LayernormMeanVarSrcDstVectorSize,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceGemmMultipleDLayernorm_Xdl_CShuffle : public BaseOperator
{
    using DeviceOp     = DeviceGemmMultipleDLayernorm_Xdl_CShuffle;
    using ELayout      = HLayout;
    using EDataType    = HDataType;
    using MeanDataType = CShuffleDataType;
    using VarDataType  = CShuffleDataType;

    static constexpr index_t NumDTensor = DsDataType::Size();

    using LayernormBlockTileSize_M_N =
        Sequence<LayernormThreadClusterSize_M_N::At(0) * LayernormThreadSliceSize_M_N::At(0),
                 LayernormThreadClusterSize_M_N::At(1) * LayernormThreadSliceSize_M_N::At(1)>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    static auto MakeAGridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    static auto MakeBGridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

    template <typename LayOut>
    static auto MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t Stride)
    {
        const auto grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, LayOut>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw), make_tuple(Stride, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, LayOut>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw), make_tuple(I1, Stride));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(grid_desc_mraw_nraw);
    }

    static auto MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                                         const std::array<index_t, NumDTensor>& NRaws,
                                         const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    static auto MakeMeanVarCountGridDescriptor_M_NBlock(index_t M, index_t NBlock)
    {
        const auto grid_desc_m_n = make_naive_tensor_descriptor_packed(make_tuple(M, NBlock));

        // TODO - padding according to MNperBlock of Gemm and Layernorm
        return grid_desc_m_n;
    }

    static auto MakeDescriptor_M(index_t MRaw)
    {
        const auto grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(MRaw));

        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(grid_desc_mraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad N
            return grid_desc_mraw;
        }
    };

    static auto MakeDescriptor_N(index_t NRaw)
    {
        const auto grid_desc_nraw = make_naive_tensor_descriptor_packed(make_tuple(NRaw));

        const auto N    = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad N
            return transform_tensor_descriptor(grid_desc_nraw,
                                               make_tuple(make_right_pad_transform(NRaw, NPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad N
            return grid_desc_nraw;
        }
    };

    using AGridDesc_M_K  = decltype(MakeAGridDescriptor_M_K(1, 1, 1));
    using BGridDesc_N_K  = decltype(MakeBGridDescriptor_N_K(1, 1, 1));
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}, {}))>;
    using MeanVarCountGridDesc_M_NBlock = decltype(MakeMeanVarCountGridDescriptor_M_NBlock(1, 1));
    using GammaBetaGridDesc_N           = decltype(MakeDescriptor_N(1));
    using EHGridDesc_M_N                = decltype(MakeEGridDescriptor_M_N<HLayout>(1, 1, 1));

    using GridwiseGemmWelford = GridwiseGemmMultipleDWelfordFirstHalf_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        MeanDataType,
        VarDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_M_K,
        BGridDesc_N_K,
        DsGridDesc_M_N,
        EHGridDesc_M_N,
        MeanVarCountGridDesc_M_NBlock,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        PostShuffleThreadClusterSize_M_N,
        PostShuffleScalarPerVector,
        LoopSched>;

    using Block2ETileMap = typename GridwiseGemmWelford::DefaultBlock2ETileMap;

    using GridwiseWelfordLayernorm =
        GridwiseWelfordSecondHalfLayernorm2d<EDataType,
                                             HDataType,
                                             MeanDataType,
                                             VarDataType,
                                             GammaDataType,
                                             BetaDataType,
                                             AccDataType,
                                             EHGridDesc_M_N,
                                             MeanVarCountGridDesc_M_NBlock,
                                             GammaBetaGridDesc_N,
                                             HElementwiseOperation,
                                             BlockSize,
                                             LayernormThreadClusterSize_M_N::At(I0),
                                             LayernormThreadClusterSize_M_N::At(I1),
                                             LayernormThreadSliceSize_M_N::At(I0),
                                             LayernormThreadSliceSize_M_N::At(I1),
                                             LayernormESrcHDstVectorDim,
                                             LayernormESrcVectorSize,
                                             LayernormHDstVectorSize,
                                             LayernormGammaSrcVectorSize,
                                             LayernormBetaSrcVectorSize,
                                             LayernormMeanVarSrcDstVectorSize>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 const void* p_gamma_grid,
                 const void* p_beta_grid,
                 void* p_h_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 std::array<index_t, NumDTensor> StrideDs,
                 index_t StrideH,
                 AccDataType epsilon,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op,
                 HElementwiseOperation h_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_ds_grid_{},
              p_e_grid_{nullptr},
              p_welford_mean_grid_{nullptr},
              p_welford_var_grid_{nullptr},
              p_welford_count_grid_{nullptr},
              p_gamma_grid_{static_cast<const GammaDataType*>(p_gamma_grid)},
              p_beta_grid_{static_cast<const BetaDataType*>(p_beta_grid)},
              p_h_grid_{static_cast<HDataType*>(p_h_grid)},
              a_grid_desc_m_k_{DeviceOp::MakeAGridDescriptor_M_K(MRaw, KRaw, StrideA)},
              b_grid_desc_n_k_{DeviceOp::MakeBGridDescriptor_N_K(KRaw, NRaw, StrideB)},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{DeviceOp::MakeEGridDescriptor_M_N<ELayout>(MRaw, NRaw, StrideH)},
              mean_var_count_grid_desc_m_nblock_{},
              gamma_grid_desc_n_{DeviceOp::MakeDescriptor_N(NRaw)},
              beta_grid_desc_n_{DeviceOp::MakeDescriptor_N(NRaw)},
              h_grid_desc_m_n_{DeviceOp::MakeEGridDescriptor_M_N<HLayout>(MRaw, NRaw, StrideH)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemmWelford::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemmWelford::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              block_2_etile_map_{GridwiseGemmWelford::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              h_element_op_{h_element_op},
              gemm_nblock_{math::integer_divide_ceil(NRaw, NPerBlock)},
              epsilon_{epsilon}
        {
            mean_var_count_grid_desc_m_nblock_ =
                DeviceOp::MakeMeanVarCountGridDescriptor_M_NBlock(MRaw, gemm_nblock_);

            // TODO - hipFree
            hip_check_error(hipMalloc(&p_e_grid_, sizeof(EDataType) * MRaw * NRaw));

            int gemm_welford_size = MRaw * gemm_nblock_;
            hip_check_error(
                hipMalloc(&p_welford_mean_grid_, sizeof(MeanDataType) * gemm_welford_size));
            hip_check_error(
                hipMalloc(&p_welford_var_grid_, sizeof(VarDataType) * gemm_welford_size));
            hip_check_error(hipMalloc(&p_welford_count_grid_, sizeof(int32_t) * gemm_welford_size));

            // populate pointer, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                // D desc
                ds_grid_desc_m_n_(i) =
                    DeviceOp::MakeEGridDescriptor_M_N<DLayout>(MRaw, NRaw, StrideDs[i]);
            });

            // populate desc for Ds/E/F/G
            if(GridwiseGemmWelford::CheckValidity(a_grid_desc_m_k_,
                                                  b_grid_desc_n_k_,
                                                  ds_grid_desc_m_n_,
                                                  e_grid_desc_m_n_,
                                                  block_2_etile_map_))
            {
                ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemmWelford::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        ds_grid_desc_m_n_);

                e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemmWelford::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_);

                mean_var_count_grid_desc_mblock_mperblock_nblock_ =
                    GridwiseGemmWelford::MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(
                        mean_var_count_grid_desc_m_nblock_);
            }
        }

        void Print() const
        {
            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
            std::cout << "H[M, N]: " << h_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemmWelford::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;
        MeanDataType* p_welford_mean_grid_;
        VarDataType* p_welford_var_grid_;
        int32_t* p_welford_count_grid_;
        const GammaDataType* p_gamma_grid_;
        const BetaDataType* p_beta_grid_;
        HDataType* p_h_grid_;

        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EHGridDesc_M_N e_grid_desc_m_n_;
        MeanVarCountGridDesc_M_NBlock mean_var_count_grid_desc_m_nblock_;
        GammaBetaGridDesc_N gamma_grid_desc_n_;
        GammaBetaGridDesc_N beta_grid_desc_n_;
        EHGridDesc_M_N h_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        typename GridwiseGemmWelford::DefaultAGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        typename GridwiseGemmWelford::DefaultBGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        typename GridwiseGemmWelford::DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemmWelford::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemmWelford::MeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock
            mean_var_count_grid_desc_mblock_mperblock_nblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
        HElementwiseOperation h_element_op_;

        int gemm_nblock_;
        AccDataType epsilon_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0;

            if(!GridwiseGemmWelford::CheckValidity(arg.a_grid_desc_m_k_,
                                                   arg.b_grid_desc_n_k_,
                                                   arg.ds_grid_desc_m_n_,
                                                   arg.e_grid_desc_m_n_,
                                                   arg.block_2_etile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemmWelford has invalid setting");
            }

            index_t grid_size = arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_);

            const auto M = arg.h_grid_desc_m_n_.GetLength(I0);
            const auto N = arg.h_grid_desc_m_n_.GetLength(I1);
            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel_gemm_welford =
                    kernel_gemm_multiple_d_welford_first_half_xdl_cshuffle<
                        GridwiseGemmWelford,
                        ADataType, // TODO: distiguish A/B datatype
                        typename GridwiseGemmWelford::DsGridPointer,
                        EDataType,
                        MeanDataType,
                        VarDataType,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation,
                        typename GridwiseGemmWelford::DefaultAGridDesc_AK0_M_AK1,
                        typename GridwiseGemmWelford::DefaultBGridDesc_BK0_N_BK1,
                        typename GridwiseGemmWelford::
                            DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        typename GridwiseGemmWelford::
                            EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        typename GridwiseGemmWelford::
                            MeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock,
                        typename GridwiseGemmWelford::DefaultBlock2ETileMap,
                        has_main_loop>;

                const auto kernel_welford_layernorm =
                    kernel_welford_layernorm2d_second_half<GridwiseWelfordLayernorm,
                                                           EDataType,
                                                           HDataType,
                                                           MeanDataType,
                                                           VarDataType,
                                                           GammaDataType,
                                                           BetaDataType,
                                                           AccDataType,
                                                           EHGridDesc_M_N,
                                                           MeanVarCountGridDesc_M_NBlock,
                                                           GammaBetaGridDesc_N,
                                                           HElementwiseOperation>;

                avg_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel_gemm_welford,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_ds_grid_,
                                           arg.p_e_grid_,
                                           arg.p_welford_mean_grid_,
                                           arg.p_welford_var_grid_,
                                           arg.p_welford_count_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.cde_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.mean_var_count_grid_desc_mblock_mperblock_nblock_,
                                           arg.block_2_etile_map_);

                grid_size = math::integer_divide_ceil(M, LayernormBlockTileSize_M_N::At(0));

                index_t numMeanVarCountBlockTileIteration_N = math::integer_divide_ceil(
                    arg.gemm_nblock_, LayernormThreadClusterSize_M_N::At(I1));

                index_t numNormBlockTileIteration_N =
                    math::integer_divide_ceil(N, LayernormBlockTileSize_M_N::At(I1));

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_welford_layernorm,
                                                   dim3(grid_size),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_e_grid_,
                                                   arg.p_welford_mean_grid_,
                                                   arg.p_welford_var_grid_,
                                                   arg.p_welford_count_grid_,
                                                   arg.p_gamma_grid_,
                                                   arg.p_beta_grid_,
                                                   arg.p_h_grid_,
                                                   arg.e_grid_desc_m_n_,
                                                   arg.h_grid_desc_m_n_,
                                                   arg.mean_var_count_grid_desc_m_nblock_,
                                                   arg.gamma_grid_desc_n_,
                                                   arg.beta_grid_desc_n_,
                                                   numMeanVarCountBlockTileIteration_N,
                                                   numNormBlockTileIteration_N,
                                                   arg.epsilon_,
                                                   arg.h_element_op_);

                return avg_time;
            };

            if(GridwiseGemmWelford::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument&)
    {
        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a"))
        {
            return false;
        }

        // TODO

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             const void* p_gamma,
                             const void* p_beta,
                             void* p_h,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideH,
                             AccDataType epsilon,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op,
                             HElementwiseOperation h_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_gamma,
                        p_beta,
                        p_h,
                        MRaw,
                        NRaw,
                        KRaw,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideH,
                        epsilon,
                        a_element_op,
                        b_element_op,
                        cde_element_op,
                        h_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      std::array<const void*, NumDTensor> p_ds,
                                                      const void* p_gamma,
                                                      const void* p_beta,
                                                      void* p_h,
                                                      index_t MRaw,
                                                      index_t NRaw,
                                                      index_t KRaw,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      std::array<index_t, NumDTensor> StrideDs,
                                                      index_t StrideH,
                                                      AccDataType epsilon,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CDEElementwiseOperation cde_element_op,
                                                      HElementwiseOperation h_element_op)
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_gamma,
                                          p_beta,
                                          p_h,
                                          MRaw,
                                          NRaw,
                                          KRaw,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideH,
                                          epsilon,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op,
                                          h_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGemmMultipleDLayernorm_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">";
        // clang-format on

        return str.str();
    }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
