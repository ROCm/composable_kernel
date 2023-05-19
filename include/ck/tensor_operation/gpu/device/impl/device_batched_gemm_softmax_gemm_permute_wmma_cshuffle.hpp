// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_wmma_cshuffle.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Computes C = A  * B0 * B1
//         MN = MK * KL * LN
//              ^^^^^^ (Acc0)
//              ^^^^^^^^^^^ (Acc1)
template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimL,
          index_t NumDimK,
          index_t NumDimN,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename Acc0BiasDataType,
          typename Acc0DataType,
          typename Acc1BiasDataType,
          typename Acc1DataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          TensorSpecialization ASpec,
          TensorSpecialization B0Spec,
          TensorSpecialization B1Spec,
          TensorSpecialization CSpec,
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t LPerBlock,
          ck::index_t KPerBlock,
          ck::index_t K1,
          ck::index_t NPerBlock,
          ck::index_t LTilePerBlock,
          ck::index_t L1,
          ck::index_t MPerWmma,
          ck::index_t LPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t LRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename B0BlockTransferThreadClusterLengths_K0_L_K1,
          typename B0BlockTransferThreadClusterArrangeOrder,
          typename B0BlockTransferSrcAccessOrder,
          ck::index_t B0BlockTransferSrcVectorDim,
          ck::index_t B0BlockTransferSrcScalarPerVector,
          ck::index_t B0BlockTransferDstScalarPerVector_K1,
          bool B0BlockLdsAddExtraL,
          typename B1BlockTransferThreadClusterLengths_L0_N_L1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          ck::index_t B1BlockTransferSrcVectorDim,
          ck::index_t B1BlockTransferSrcScalarPerVector,
          ck::index_t B1BlockTransferDstScalarPerVector_L1,
          bool B1BlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          MaskingSpecialization MaskingSpec,
          ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::v1>
struct DeviceBatchedGemmSoftmaxGemmPermute_Wmma_CShuffle
    : public DeviceBatchedGemmSoftmaxGemmPermute<NumDimG,
                                                 NumDimM,
                                                 NumDimL,
                                                 NumDimK,
                                                 NumDimN,
                                                 ADataType,
                                                 B0DataType,
                                                 B1DataType,
                                                 CDataType,
                                                 Acc0BiasDataType,
                                                 Acc1BiasDataType,
                                                 AElementwiseOperation,
                                                 B0ElementwiseOperation,
                                                 AccElementwiseOperation,
                                                 B1ElementwiseOperation,
                                                 CElementwiseOperation,
                                                 MaskingSpec>
{
    static_assert(NumDimG > 0 && NumDimM > 0 && NumDimL > 0 && NumDimK > 0 && NumDimN > 0,
                  "Number of dimension must be greater than 0");

    static constexpr index_t NumAcc0Bias = Acc0BiasDataType::Size();
    static constexpr index_t NumAcc1Bias = Acc1BiasDataType::Size();

    // TODO ANT: implement bias combination
    static_assert(NumAcc0Bias == 0 && NumAcc0Bias == 0, "Bias addition is unimplemented");

    static constexpr index_t NumDimGemm0M = NumDimM;
    static constexpr index_t NumDimGemm0N = NumDimL;
    static constexpr index_t NumDimGemm0K = NumDimK;
    static constexpr index_t NumDimGemm1M = NumDimM;
    static constexpr index_t NumDimGemm1N = NumDimN;
    static constexpr index_t NumDimGemm1K = NumDimL;

    using DeviceOp = DeviceBatchedGemmSoftmaxGemmPermute_Wmma_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    static constexpr auto WmmaK = 16;

    static constexpr auto MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto LWaves = LPerBlock / (LRepeat * LPerWmma);
    static constexpr auto NWaves = NPerBlock / (NRepeat * NPerWmma);

    static constexpr auto AEnableLds_auto  = LWaves == 1 ? false : true;
    static constexpr auto B0EnableLds_auto = MWaves == 1 ? false : true;
    static constexpr auto B1EnableLds_auto = MWaves == 1 ? false : true;

    static constexpr auto AEnableLds_manu  = false;
    static constexpr auto B0EnableLds_manu = true;
    static constexpr auto B1EnableLds_manu = true;

    static constexpr auto AEnableLds  = AEnableLds_auto || AEnableLds_manu || (NumPrefetch > 1);
    static constexpr auto B0EnableLds = B0EnableLds_auto || B0EnableLds_manu || (NumPrefetch > 1);
    static constexpr auto B1EnableLds = B1EnableLds_auto || B1EnableLds_manu || (NumPrefetch > 1);

    using Transform = TransformBatchedContractionContractionToBatchedGemmGemm<
        Sequence<NumDimG, NumDimM, NumDimL, NumDimK, NumDimN>,
        Sequence<MPerBlock, LPerBlock, KPerBlock, NPerBlock>,
        GemmSpec,
        ASpec,
        B0Spec,
        B1Spec,
        CSpec>;

    static auto MakeAGridDescriptor(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                    const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        if constexpr(AEnableLds)
        {
            return Transform::MakeAGridDescriptor_AK0_M_AK1(
                Transform::MakeAGridDescriptor_M_K(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec),
                Number<K1>{});
        }
        else
        {
            return Transform::
                MakeAGridDescriptor_AKWmma_MBlockRepeat_MWaves_AK0PerWmma_AKRow_MPerWmma_AK1(
                    Transform::MakeAGridDescriptor_M_K(a_gs_ms_ks_lengths_vec,
                                                       a_gs_ms_ks_strides_vec),
                    Number<WmmaK>{},
                    Number<MRepeat>{},
                    Number<MWaves>{},
                    Number<MPerWmma>{},
                    Number<K1>{});
        }
    }

    static auto MakeB0GridDescriptor(const std::vector<index_t>& b0_gs_ls_ks_lengths_vec,
                                     const std::vector<index_t>& b0_gs_ls_ks_strides_vec)
    {
        if constexpr(B0EnableLds)
        {
            return Transform::MakeB0GridDescriptor_BK0_N_BK1(
                Transform::MakeB0GridDescriptor_N_K(b0_gs_ls_ks_lengths_vec,
                                                    b0_gs_ls_ks_strides_vec),
                Number<K1>{});
        }
        else
        {
            return Transform::
                MakeB0GridDescriptor_BKWmma_LBlockRepeat_LWaves_BK0PerWmma_BKRow_LPerWmma_BK1(
                    Transform::MakeB0GridDescriptor_N_K(b0_gs_ls_ks_lengths_vec,
                                                        b0_gs_ls_ks_strides_vec),
                    Number<WmmaK>{},
                    Number<LRepeat>{},
                    Number<LWaves>{},
                    Number<LPerWmma>{},
                    Number<K1>{});
        }
    }

    static auto MakeB1GridDescriptor(const std::vector<index_t>& b1_gs_ns_ls_lengths_vec,
                                     const std::vector<index_t>& b1_gs_ns_ls_strides_vec)
    {
        if constexpr(B1EnableLds)
        {
            return Transform::MakeB1GridDescriptor_BK0_N_BK1(
                Transform::MakeB1GridDescriptor_N_K(b1_gs_ns_ls_lengths_vec,
                                                    b1_gs_ns_ls_strides_vec),
                Number<L1>{});
        }
        else
        {
            return Transform::
                MakeB1GridDescriptor_BLWmma_NBlockRepeat_NWaves__BL0PerWmma_BLRow_NPerWmma_BL1(
                    Transform::MakeB1GridDescriptor_N_K(b1_gs_ns_ls_lengths_vec,
                                                        b1_gs_ns_ls_strides_vec),
                    Number<WmmaK>{},
                    Number<NRepeat>{},
                    Number<NWaves>{},
                    Number<NPerWmma>{},
                    Number<L1>{});
        }
    }

    using AGridDesc        = decltype(MakeAGridDescriptor({}, {}));
    using B0GridDesc       = decltype(MakeB0GridDescriptor({}, {}));
    using B1GridDesc       = decltype(MakeB1GridDescriptor({}, {}));
    using CGridDesc_M_N    = decltype(Transform::MakeCGridDescriptor_M_N({}, {}));
    using AGridDesc_G_M_K  = decltype(Transform::MakeAGridDescriptor_G_M_K({}, {}));
    using B0GridDesc_G_L_K = decltype(Transform::MakeB0GridDescriptor_G_N_K({}, {}));
    using B1GridDesc_G_N_L = decltype(Transform::MakeB1GridDescriptor_G_N_K({}, {}));
    using CGridDesc_G_M_N  = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));

    constexpr static auto make_MaskOutPredicate()
    {
        if constexpr(MaskingSpec == MaskingSpecialization::MaskDisabled)
        {
            return MaskDisabledPredicate{};
        }
        else if constexpr(MaskingSpec == MaskingSpecialization::MaskOutUpperTriangle)
        {
            return MaskOutUpperTrianglePredicate{};
        }
    }
    using C0MatrixMask = C0MatrixMask_impl<decltype(make_MaskOutPredicate())>;

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(const AGridDesc_G_M_K& a_grid_desc_g_m_k,
                                     const B0GridDesc_G_L_K& b0_grid_desc_g_l_k,
                                     const B1GridDesc_G_N_L& b1_grid_desc_g_n_l,
                                     const CGridDesc_G_M_N& c_grid_desc_g_m_n)
            : a_grid_desc_g_m_k_(a_grid_desc_g_m_k),
              b0_grid_desc_g_l_k_(b0_grid_desc_g_l_k),
              b1_grid_desc_g_n_l_(b1_grid_desc_g_n_l),
              c_grid_desc_g_m_n_(c_grid_desc_g_m_n)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return a_grid_desc_g_m_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetB0BasePtr(index_t g_idx) const
        {
            return b0_grid_desc_g_l_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return b1_grid_desc_g_n_l_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return c_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        private:
        AGridDesc_G_M_K a_grid_desc_g_m_k_;
        B0GridDesc_G_L_K b0_grid_desc_g_l_k_;
        B1GridDesc_G_N_L b1_grid_desc_g_n_l_;
        CGridDesc_G_M_N c_grid_desc_g_m_n_;
    };

    // GridwiseOp
    using GridwiseOp = GridwiseBatchedGemmSoftmaxGemm_Wmma<
        // DataType Family
        ADataType,
        B0DataType,
        Acc0DataType,
        B1DataType,
        Acc1DataType,
        CShuffleDataType,
        CDataType,
        // ElementwiseOp Family
        AElementwiseOperation,
        B0ElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        // InMemory Data Descriptor
        AGridDesc,
        B0GridDesc,
        B1GridDesc,
        CGridDesc_M_N,
        // Tiling Family
        MPerBlock,
        LPerBlock,
        KPerBlock,
        K1,
        NPerBlock,
        LTilePerBlock,
        L1,
        MPerWmma,
        LPerWmma,
        NPerWmma,
        MRepeat,
        LRepeat,
        NRepeat,
        // ThreadCluster Family
        BlockSize,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        true,
        AEnableLds,
        ABlockLdsAddExtraM,
        B0BlockTransferThreadClusterLengths_K0_L_K1,
        B0BlockTransferThreadClusterArrangeOrder,
        B0BlockTransferSrcAccessOrder,
        B0BlockTransferSrcVectorDim,
        B0BlockTransferSrcScalarPerVector,
        B0BlockTransferDstScalarPerVector_K1,
        true,
        B0EnableLds,
        B0BlockLdsAddExtraL,
        B1BlockTransferThreadClusterLengths_L0_N_L1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_L1,
        false,
        B1EnableLds,
        B1BlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        Transform::matrix_padder.PadN,
        MaskingSpec == MaskingSpecialization::MaskOutUpperTriangle,
        NumPrefetch,
        LoopSched,
        PipelineVer>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(
            const ADataType* p_a_grid,
            const B0DataType* p_b0_grid,
            const B1DataType* p_b1_grid,
            CDataType* p_c_grid,
            const std::array<void*, NumAcc0Bias> p_acc0_biases,
            const std::array<void*, NumAcc1Bias> p_acc1_biases,
            const std::vector<index_t>& a_gs_ms_ks_lengths,
            const std::vector<index_t>& a_gs_ms_ks_strides,
            const std::vector<index_t>& b0_gs_ls_ks_lengths,
            const std::vector<index_t>& b0_gs_ls_ks_strides,
            const std::vector<index_t>& b1_gs_ns_ls_lengths,
            const std::vector<index_t>& b1_gs_ns_ls_strides,
            const std::vector<index_t>& c_gs_ms_ns_lengths,
            const std::vector<index_t>& c_gs_ms_ns_strides,
            const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ls_lengths,
            const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ls_strides,
            const std::array<std::vector<ck::index_t>, NumAcc1Bias> acc1_biases_gs_ms_ns_lengths,
            const std::array<std::vector<ck::index_t>, NumAcc1Bias> acc1_biases_gs_ms_ns_strides,
            const index_t M01,
            const index_t N01,
            AElementwiseOperation a_element_op,
            B0ElementwiseOperation b0_element_op,
            AccElementwiseOperation acc_element_op,
            B1ElementwiseOperation b1_element_op,
            CElementwiseOperation c_element_op)
            : p_a_grid_{p_a_grid},
              p_b0_grid_{p_b0_grid},
              p_b1_grid_{p_b1_grid},
              p_c_grid_{p_c_grid},
              a_grid_desc{DeviceOp::MakeAGridDescriptor(a_gs_ms_ks_lengths, a_gs_ms_ks_strides)},
              b0_grid_desc{
                  DeviceOp::MakeB0GridDescriptor(b0_gs_ls_ks_lengths, b0_gs_ls_ks_strides)},
              b1_grid_desc{
                  DeviceOp::MakeB1GridDescriptor(b1_gs_ns_ls_lengths, b1_gs_ns_ls_strides)},
              c_grid_desc_m_n_{
                  Transform::MakeCGridDescriptor_M_N(c_gs_ms_ns_lengths, c_gs_ms_ns_strides)},
              a_grid_desc_g_m_k_{
                  Transform::MakeAGridDescriptor_G_M_K(a_gs_ms_ks_lengths, a_gs_ms_ks_strides)},
              b0_grid_desc_g_l_k_{
                  Transform::MakeB0GridDescriptor_G_N_K(b0_gs_ls_ks_lengths, b0_gs_ls_ks_strides)},
              b1_grid_desc_g_n_l_{
                  Transform::MakeB1GridDescriptor_G_N_K(b1_gs_ns_ls_lengths, b1_gs_ns_ls_strides)},
              c_grid_desc_g_m_n_{
                  Transform::MakeCGridDescriptor_G_M_N(c_gs_ms_ns_lengths, c_gs_ms_ns_strides)},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_ctile_map_{GridwiseOp::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01)},
              a_element_op_{a_element_op},
              b0_element_op_{b0_element_op},
              acc_element_op_{acc_element_op},
              b1_element_op_{b1_element_op},
              c_element_op_{c_element_op},
              c0_matrix_mask_{b0_grid_desc_g_l_k_.GetLength(I1)},
              raw_lengths_mz_lz_kz_nz_{a_gs_ms_ks_lengths[NumDimG + NumDimM - 1],
                                       b0_gs_ls_ks_lengths[NumDimG + NumDimL - 1],
                                       b0_gs_ls_ks_lengths[NumDimG + NumDimL + NumDimK - 1],
                                       b1_gs_ns_ls_lengths[NumDimG + NumDimN - 1]},
              a_mz_kz_strides_{a_gs_ms_ks_strides[NumDimG + NumDimM - 1],
                               a_gs_ms_ks_strides[NumDimG + NumDimM + NumDimK - 1]},
              b0_lz_kz_strides_{b0_gs_ls_ks_strides[NumDimG + NumDimL - 1],
                                b0_gs_ls_ks_strides[NumDimG + NumDimL + NumDimK - 1]},
              b1_nz_lz_strides_{b1_gs_ns_ls_strides[NumDimG + NumDimN - 1],
                                b1_gs_ns_ls_strides[NumDimG + NumDimN + NumDimL - 1]},
              c_mz_nz_strides_{c_gs_ms_ns_strides[NumDimG + NumDimM - 1],
                               c_gs_ms_ns_strides[NumDimG + NumDimM + NumDimN - 1]},
              batch_count_{c_grid_desc_g_m_n_.GetLength(I0)},
              compute_ptr_offset_of_batch_{
                  a_grid_desc_g_m_k_, b0_grid_desc_g_l_k_, b1_grid_desc_g_n_l_, c_grid_desc_g_m_n_}
        {
            // TODO ANT: implement bias addition
            ignore = p_acc0_biases;
            ignore = p_acc1_biases;
            ignore = acc0_biases_gs_ms_ls_lengths;
            ignore = acc0_biases_gs_ms_ls_strides;
            ignore = acc1_biases_gs_ms_ns_lengths;
            ignore = acc1_biases_gs_ms_ns_strides;

            if(GridwiseOp::CheckValidity(
                   a_grid_desc, b0_grid_desc, b1_grid_desc, c_grid_desc_m_n_, block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseOp::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);
            }
        }

        // Pointers
        const ADataType* p_a_grid_;
        const B0DataType* p_b0_grid_;
        const B1DataType* p_b1_grid_;
        CDataType* p_c_grid_;

        // Tensor Descriptors
        AGridDesc a_grid_desc;
        B0GridDesc b0_grid_desc;
        B1GridDesc b1_grid_desc;
        CGridDesc_M_N c_grid_desc_m_n_;

        AGridDesc_G_M_K a_grid_desc_g_m_k_;
        B0GridDesc_G_L_K b0_grid_desc_g_l_k_;
        B1GridDesc_G_N_L b1_grid_desc_g_n_l_;
        CGridDesc_G_M_N c_grid_desc_g_m_n_;

        typename GridwiseOp::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;

        // Block to Tile mapping
        typename GridwiseOp::DefaultBlock2CTileMap block_2_ctile_map_;

        // ElementwiseOp
        AElementwiseOperation a_element_op_;
        B0ElementwiseOperation b0_element_op_;
        AccElementwiseOperation acc_element_op_;
        B1ElementwiseOperation b1_element_op_;
        CElementwiseOperation c_element_op_;

        // check C0 masking and padding
        C0MatrixMask c0_matrix_mask_;

        // Strides for the last M/N/K dimensions of A/B0/B1/C
        //   for sanity check of vector load/store
        std::vector<index_t> raw_lengths_mz_lz_kz_nz_;
        std::vector<index_t> a_mz_kz_strides_;
        std::vector<index_t> b0_lz_kz_strides_;
        std::vector<index_t> b1_nz_lz_strides_;
        std::vector<index_t> c_mz_nz_strides_;

        index_t batch_count_;
        // Batch Offset
        ComputeBasePtrOfStridedBatch compute_ptr_offset_of_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) * arg.batch_count_;

            const auto K = [&]() {
                if constexpr(AEnableLds)
                {
                    return arg.a_grid_desc.GetLength(I0) * arg.a_grid_desc.GetLength(I2);
                }
                else
                {
                    return arg.a_grid_desc.GetLength(I0) * arg.a_grid_desc.GetLength(I3) *
                           arg.a_grid_desc.GetLength(I4) * arg.a_grid_desc.GetLength(I6);
                }
            }();

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                const auto kernel = kernel_batched_gemm_softmax_gemm_wmma_cshuffle<
                    GridwiseOp,
                    ADataType,
                    B0DataType,
                    B1DataType,
                    CDataType,
                    DeviceOp::AGridDesc,
                    DeviceOp::B0GridDesc,
                    DeviceOp::B1GridDesc,
                    typename GridwiseOp::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    AElementwiseOperation,
                    B0ElementwiseOperation,
                    AccElementwiseOperation,
                    B1ElementwiseOperation,
                    CElementwiseOperation,
                    ComputeBasePtrOfStridedBatch,
                    C0MatrixMask,
                    typename GridwiseOp::DefaultBlock2CTileMap,
                    has_main_k_block_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b0_grid_,
                                              arg.p_b1_grid_,
                                              arg.p_c_grid_,
                                              arg.a_grid_desc,
                                              arg.b0_grid_desc,
                                              arg.b1_grid_desc,
                                              arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.a_element_op_,
                                              arg.b0_element_op_,
                                              arg.acc_element_op_,
                                              arg.b1_element_op_,
                                              arg.c_element_op_,
                                              arg.batch_count_,
                                              arg.compute_ptr_offset_of_batch_,
                                              arg.c0_matrix_mask_,
                                              arg.block_2_ctile_map_);
            };

            if(GridwiseOp::CalculateHasMainKBlockLoop(K))
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

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
           ck::get_device_name() == "gfx1102")
        {
            if constexpr(!(is_same_v<Acc0DataType, float> || is_same_v<Acc0DataType, int32_t>))
            {
                printf("DeviceOp: Acc0 Type err");
                return false;
            }

            if constexpr(!(is_same_v<Acc1DataType, float> || is_same_v<Acc1DataType, int32_t>))
            {
                printf("DeviceOp: Acc1 Type err");
                return false;
            }
        }
        else
        {
            printf("DeviceOp: Arch err");
            return false;
        }

        if(!GridwiseOp::CheckValidity(arg.a_grid_desc,
                                      arg.b0_grid_desc,
                                      arg.b1_grid_desc,
                                      arg.c_grid_desc_m_n_,
                                      arg.block_2_ctile_map_))
        {
            return false;
        }

        // Check if C permute dimension matches GEMM + GEMM shape
        const index_t c_g = arg.c_grid_desc_g_m_n_.GetLength(I0); // unpadded

        if(!(c_g == arg.batch_count_))
        {
            printf("DeviceOp: BatchCount err");
            return false;
        }

        // Note: we need raw lengths since threadwise copy can not handle vector load when part of
        // vector is out of bounds
        // Note: need lowest dim in Ms/Ns/Ks/Os, not merged M/N/K/O
        const auto MzRaw = arg.raw_lengths_mz_lz_kz_nz_[0];
        const auto LzRaw = arg.raw_lengths_mz_lz_kz_nz_[1];
        const auto KzRaw = arg.raw_lengths_mz_lz_kz_nz_[2];
        const auto NzRaw = arg.raw_lengths_mz_lz_kz_nz_[3];

        // Check scalar per vector requirement
        const auto a_extent_lowest  = ABlockTransferSrcVectorDim == 2 ? KzRaw : MzRaw;
        const auto b0_extent_lowest = B0BlockTransferSrcVectorDim == 2 ? KzRaw : LzRaw;
        const auto b1_extent_lowest = B1BlockTransferSrcVectorDim == 2 ? LzRaw : NzRaw;
        const auto c_extent_lowest  = NzRaw;

        if(!(a_extent_lowest % ABlockTransferSrcScalarPerVector == 0 &&
             b0_extent_lowest % B0BlockTransferSrcScalarPerVector == 0 &&
             b1_extent_lowest % B1BlockTransferSrcScalarPerVector == 0 &&
             c_extent_lowest % CShuffleBlockTransferScalarPerVector_NPerBlock == 0))
        {
            printf("DeviceOp: Data Transfer Vector scalar err");
            return false;
        }

        // Check vector load/store requirement
        const auto a_stride_lowest =
            ABlockTransferSrcVectorDim == 2 ? arg.a_mz_kz_strides_[1] : arg.a_mz_kz_strides_[0];
        const auto b0_stride_lowest =
            B0BlockTransferSrcVectorDim == 2 ? arg.b0_lz_kz_strides_[1] : arg.b0_lz_kz_strides_[0];
        const auto b1_stride_lowest =
            B1BlockTransferSrcVectorDim == 2 ? arg.b1_nz_lz_strides_[1] : arg.b1_nz_lz_strides_[0];
        const auto c_stride_lowest = arg.c_mz_nz_strides_[1];

        if(!(a_stride_lowest == 1 || b0_stride_lowest == 1 || b1_stride_lowest == 1 ||
             c_stride_lowest == 1))
        {
            printf("DeviceOp: Data Vectorize transfer err");
            return false;
        }

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const ADataType* p_a,
        const B0DataType* p_b0,
        const B1DataType* p_b1,
        CDataType* p_c,
        const std::array<void*, NumAcc0Bias> p_acc0_biases,
        const std::array<void*, NumAcc1Bias> p_acc1_biases,
        const std::vector<index_t>& a_gs_ms_ks_lengths,
        const std::vector<index_t>& a_gs_ms_ks_strides,
        const std::vector<index_t>& b0_gs_ls_ks_lengths,
        const std::vector<index_t>& b0_gs_ls_ks_strides,
        const std::vector<index_t>& b1_gs_ns_ls_lengths,
        const std::vector<index_t>& b1_gs_ns_ls_strides,
        const std::vector<index_t>& c_gs_ms_ns_lengths,
        const std::vector<index_t>& c_gs_ms_ns_strides,
        const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ls_lengths,
        const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ls_strides,
        const std::array<std::vector<ck::index_t>, NumAcc1Bias> acc1_biases_gs_ms_ns_lengths,
        const std::array<std::vector<ck::index_t>, NumAcc1Bias> acc1_biases_gs_ms_ns_strides,
        AElementwiseOperation a_element_op,
        B0ElementwiseOperation b0_element_op,
        AccElementwiseOperation acc_element_op,
        B1ElementwiseOperation b1_element_op,
        CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_b0,
                        p_b1,
                        p_c,
                        p_acc0_biases,
                        p_acc1_biases,
                        a_gs_ms_ks_lengths,
                        a_gs_ms_ks_strides,
                        b0_gs_ls_ks_lengths,
                        b0_gs_ls_ks_strides,
                        b1_gs_ns_ls_lengths,
                        b1_gs_ns_ls_strides,
                        c_gs_ms_ns_lengths,
                        c_gs_ms_ns_strides,
                        acc0_biases_gs_ms_ls_lengths,
                        acc0_biases_gs_ms_ls_strides,
                        acc1_biases_gs_ms_ns_lengths,
                        acc1_biases_gs_ms_ns_strides,
                        1,
                        1,
                        a_element_op,
                        b0_element_op,
                        acc_element_op,
                        b1_element_op,
                        c_element_op};
    }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b0,
        const void* p_b1,
        void* p_c,
        const std::array<void*, NumAcc0Bias> p_acc0_biases,
        const std::array<void*, NumAcc1Bias> p_acc1_biases,
        const std::vector<index_t>& a_gs_ms_ks_lengths,
        const std::vector<index_t>& a_gs_ms_ks_strides,
        const std::vector<index_t>& b0_gs_ls_ks_lengths,
        const std::vector<index_t>& b0_gs_ls_ks_strides,
        const std::vector<index_t>& b1_gs_ns_ls_lengths,
        const std::vector<index_t>& b1_gs_ns_ls_strides,
        const std::vector<index_t>& c_gs_ms_ns_lengths,
        const std::vector<index_t>& c_gs_ms_ns_strides,
        const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ls_lengths,
        const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ls_strides,
        const std::array<std::vector<ck::index_t>, NumAcc1Bias> acc1_biases_gs_ms_ns_lengths,
        const std::array<std::vector<ck::index_t>, NumAcc1Bias> acc1_biases_gs_ms_ns_strides,
        AElementwiseOperation a_element_op,
        B0ElementwiseOperation b0_element_op,
        AccElementwiseOperation acc_element_op,
        B1ElementwiseOperation b1_element_op,
        CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const B0DataType*>(p_b0),
                                          static_cast<const B1DataType*>(p_b1),
                                          static_cast<CDataType*>(p_c),
                                          p_acc0_biases,
                                          p_acc1_biases,
                                          a_gs_ms_ks_lengths,
                                          a_gs_ms_ks_strides,
                                          b0_gs_ls_ks_lengths,
                                          b0_gs_ls_ks_strides,
                                          b1_gs_ns_ls_lengths,
                                          b1_gs_ns_ls_strides,
                                          c_gs_ms_ns_lengths,
                                          c_gs_ms_ns_strides,
                                          acc0_biases_gs_ms_ls_lengths,
                                          acc0_biases_gs_ms_ls_strides,
                                          acc1_biases_gs_ms_ns_lengths,
                                          acc1_biases_gs_ms_ns_strides,
                                          1,
                                          1,
                                          a_element_op,
                                          b0_element_op,
                                          acc_element_op,
                                          b1_element_op,
                                          c_element_op);
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{{PipelineVersion::v1, "v1"},
                                                                       {PipelineVersion::v2, "v2"}};

        // clang-format off
        str << "DeviceBatchedGemmSoftmaxGemmPermute_Wmma_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << LPerBlock << ", "
            << KPerBlock << ", "
            << K1 << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << LTilePerBlock << ", "
            << L1
            << getGemmSpecializationString(GemmSpec) << ", "
            << "ASpec" << getTensorSpecializationString(ASpec) << ", "
            << "B0Spec" << getTensorSpecializationString(B0Spec) << ", "
            << "B1Spec" << getTensorSpecializationString(B1Spec) << ", "
            << "CSpec" << getTensorSpecializationString(CSpec) << ", "
            << getMaskingSpecializationString(MaskingSpec)
            << ">"
            << " AEnableLds: "
            << AEnableLds << ", "
            << "B0EnableLds: "
            << B0EnableLds << ", "
            << "B1EnableLds: "
            << B1EnableLds << ", "
            << "NumPrefetch: "
            << NumPrefetch << ", "
            << "LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
