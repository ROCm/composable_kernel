// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename A0B0B1DataType,
          typename C1DataType,
          typename A0ElementwiseOperation,
          typename B0ElementwiseOperation,
          typename C0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename C1ElementwiseOperation,
          typename A0GridDesc_AK0_M_AK1,
          typename B0GridDesc_BK0_N_BK1,
          typename B1GridDesc_BK0_N_BK1,
          typename C1GridDescriptor_MBlock_Gemm0MPerBlock_NBlock_Gemm0NPerBlock,
          typename Block2C1TileMap,
          typename ComputeBasePtrOfStridedBatch,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_gemm_xdl_cshuffle_v1(
            const A0B0B1DataType* __restrict__ p_a0_grid,
            const A0B0B1DataType* __restrict__ p_b0_grid,
            const A0B0B1DataType* __restrict__ p_b1_grid,
            C1DataType* __restrict__ p_c1_grid,
            const A0ElementwiseOperation a0_element_op,
            const B0ElementwiseOperation b0_element_op,
            const C0ElementwiseOperation c0_element_op,
            const B1ElementwiseOperation b1_element_op,
            const C1ElementwiseOperation c1_element_op,
            const A0GridDesc_AK0_M_AK1 a0_grid_desc_ak0_m_ak1,
            const B0GridDesc_BK0_N_BK1 b0_grid_desc_bk0_n_bk1,
            const B1GridDesc_BK0_N_BK1 b1_grid_desc_bk0_n_bk1,
            const C1GridDescriptor_MBlock_Gemm0MPerBlock_NBlock_Gemm0NPerBlock
                c1_grid_desc_mblock_Gemm0MPerBlock_nblock_Gemm0NPerBlock,
            const Block2C1TileMap block_2_c1tile_map,
            const index_t batch_count,
            const ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetABasePtr(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetBBasePtr(g_idx)));
    const long_index_t b1_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetB1BasePtr(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetCBasePtr(g_idx)));

    GridwiseGemm::template Run<HasMainKBlockLoop>(
        p_a0_grid + a_batch_offset,
        p_b0_grid + b_batch_offset,
        p_b1_grid + b1_batch_offset,
        p_c1_grid + c_batch_offset,
        p_shared,
        a0_element_op,
        b0_element_op,
        c0_element_op,
        b1_element_op,
        c1_element_op,
        a0_grid_desc_ak0_m_ak1,
        b0_grid_desc_bk0_n_bk1,
        b1_grid_desc_bk0_n_bk1,
        c1_grid_desc_mblock_Gemm0MPerBlock_nblock_Gemm0NPerBlock,
        block_2_c1tile_map);
#else
    ignore = p_a0_grid;
    ignore = p_b0_grid;
    ignore = p_b1_grid;
    ignore = p_c1_grid;
    ignore = a0_element_op;
    ignore = b0_element_op;
    ignore = c0_element_op;
    ignore = b1_element_op;
    ignore = c1_element_op;
    ignore = a0_grid_desc_ak0_m_ak1;
    ignore = b0_grid_desc_bk0_n_bk1;
    ignore = b1_grid_desc_bk0_n_bk1;
    ignore = c1_grid_desc_mblock_Gemm0MPerBlock_nblock_Gemm0NPerBlock;
    ignore = block_2_c1tile_map;
    ignore = batch_count;
    ignore = compute_base_ptr_of_batch;
#endif
}

// Convolution Forward:
//   input : input image A0[G, N, C0, Hi0, Wi0]
//   input : weight B0[G, K0, C0, Y0, X0]
//   input : residual D00[G, N, K0, Ho0, Wo0], D01[G, N, K0, Ho0, Wo0], ...
//   input : weight B1[G, K1, C1, 1, 1], (1x1 only)
//   input : residual D10[G, N, K1, Ho1, Wo1], D11[G, N, K1, Ho1, Wo1], ...
//   output : output image E1[G, N, K1, Ho1, Wo1]
//   C0 = a0_op(A0) * b0_op(B0)
//   E0 = cde0_op(C0, D00, D01, ...)
//   C1 = E0 * b1_op(B1)
//   E1 = cde1_op(C1, D10, D11, ...)
template <index_t NDimSpatial,
          typename A0Layout,
          typename B0Layout,
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename E1Layout,
          typename A0DataType,
          typename B0DataType,
          typename D0sDataType,
          typename B1DataType,
          typename D1sDataType,
          typename E1DataType,
          typename A0ElementwiseOperation,
          typename B0ElementwiseOperation,
          typename CDE0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation,
          bool Gemm0PadM,
          bool Gemm0PadN,
          bool Gemm0PadK,
          bool Gemm1PadN,
          bool Gemm1PadK,
          index_t NumGemm0KPrefetchStage,
          index_t BlockSize,
          index_t Gemm0MPerBlock,
          index_t Gemm0NPerBlock,
          index_t Gemm0KPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t A0K1,
          index_t B0K1,
          index_t B1K1,
          index_t Gemm0MPerXdl,
          index_t Gemm0NPerXdl,
          index_t Gemm0MXdlPerWave,
          index_t Gemm0NXdlPerWave,
          index_t Gemm1NXdlPerWave,
          typename A0BlockTransferThreadClusterLengths_AK0_M_AK1,
          typename A0BlockTransferThreadClusterArrangeOrder,
          typename A0BlockTransferSrcAccessOrder,
          index_t A0BlockTransferSrcVectorDim,
          index_t A0BlockTransferSrcScalarPerVector,
          index_t A0BlockTransferDstScalarPerVector_AK1,
          bool A0BlockLdsExtraM,
          typename B0BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B0BlockTransferThreadClusterArrangeOrder,
          typename B0BlockTransferSrcAccessOrder,
          index_t B0BlockTransferSrcVectorDim,
          index_t B0BlockTransferSrcScalarPerVector,
          index_t B0BlockTransferDstScalarPerVector_BK1,
          bool B0BlockLdsExtraN,
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1BlockLdsExtraN,
          index_t C1ShuffleMXdlPerWavePerShuffle,
          index_t C1ShuffleGemm0NXdlPerWavePerShuffle,
          typename C1ShuffleBlockTransferClusterLengths_MBlock_Gemm0MPerBlock_NBlock_Gemm0NPerBlock,
          index_t C1ShuffleBlockTransferScalarPerVector_Gemm0NPerBlock,
          LoopScheduler LoopSched = LoopScheduler::Default>
struct DeviceGroupedConvConvFwd_Xdl : public DeviceBatchedGemmGemm<NDimSpatial,
                                                                   A0Layout,
                                                                   B0Layout,
                                                                   D0sLayout,
                                                                   B1Layout,
                                                                   D1sLayout,
                                                                   E1Layout,
                                                                   A0DataType,
                                                                   B0DataType,
                                                                   D0sDataType,
                                                                   B1DataType,
                                                                   D1sDataType,
                                                                   E1DataType,
                                                                   A0ElementwiseOperation,
                                                                   B0ElementwiseOperation,
                                                                   CDE0ElementwiseOperation,
                                                                   B1ElementwiseOperation,
                                                                   CDE1ElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmGemm_Xdl_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto gemm0_padder =
        MatrixPadder_v2<Gemm0PadM, Gemm0PadN, Gemm0PadK, index_t, index_t, index_t>{
            Gemm0MPerBlock, Gemm0NPerBlock, Gemm0KPerBlock};

    static constexpr auto gemm1_padder =
        MatrixPadder_v2<Gemm0PadM, Gemm1PadN, Gemm1PadK, index_t, index_t, index_t>{
            Gemm0KPerBlock, Gemm1NPerBlock, Gemm1KPerBlock};

    // for Gemm0
    static auto MakeA0GridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA0)
    {
        const auto a0_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, A0Layout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA0, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, A0Layout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA0));
            }
        }();

        return gemm0_padder.PadADescriptor_M_K(a0_grid_desc_mraw_kraw);
    }

    // for Gemm0
    static auto MakeB0GridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b0_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, B0Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, B0Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return gemm0_padder.PadBDescriptor_N_K(b0_grid_desc_nraw_kraw);
    }

    // for Gemm1
    static auto MakeB1GridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b1_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, B1Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, B1Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return gemm1_padder.PadBDescriptor_N_K(b1_grid_desc_nraw_kraw);
    }

    // for Gemm1
    static auto MakeC1GridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideC1)
    {
        const auto c1_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, C1Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideC1, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, C1Layout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideC1));
            }
        }();

        return gemm1_padder.PadCDescriptor_M_N(c1_grid_desc_mraw_nraw);
    }

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(index_t BatchStrideA0,
                                     index_t BatchStrideB0,
                                     index_t BatchStrideB1,
                                     index_t BatchStrideC1)
            : BatchStrideA0_(BatchStrideA0),
              BatchStrideB0_(BatchStrideB0),
              BatchStrideB1_(BatchStrideB1),
              BatchStrideC1_(BatchStrideC1)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA0_);
        }

        __host__ __device__ constexpr long_index_t GetBBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB0_);
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB1_);
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC1_);
        }

        private:
        index_t BatchStrideA0_;
        index_t BatchStrideB0_;
        index_t BatchStrideB1_;
        index_t BatchStrideC1_;
    };

    using A0GridDesc_M_K = decltype(MakeA0GridDescriptor_M_K(1, 1, 1));
    using B0GridDesc_N_K = decltype(MakeB0GridDescriptor_N_K(1, 1, 1));
    using B1GridDesc_N_K = decltype(MakeB1GridDescriptor_N_K(1, 1, 1));
    using C1GridDesc_M_N = decltype(MakeC1GridDescriptor_M_N(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseBatchedGemmGemm_Xdl_CShuffle<
        A0DataType, // TODO: distinguish A/B datatype
        Acc0DataType,
        Acc1DataType,
        C1ShuffleDataType,
        C1DataType,
        A0ElementwiseOperation,
        B0ElementwiseOperation,
        C0ElementwiseOperation,
        B1ElementwiseOperation,
        C1ElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        A0GridDesc_M_K,
        B0GridDesc_N_K,
        B1GridDesc_N_K,
        C1GridDesc_M_N,
        NumGemm0KPrefetchStage,
        BlockSize,
        Gemm0MPerBlock,
        Gemm0NPerBlock,
        Gemm0KPerBlock,
        Gemm1NPerBlock,
        Gemm1KPerBlock,
        A0K1,
        B0K1,
        B1K1,
        Gemm0MPerXdl,
        Gemm0NPerXdl,
        Gemm0MXdlPerWave,
        Gemm0NXdlPerWave,
        Gemm1NXdlPerWave,
        A0BlockTransferThreadClusterLengths_AK0_M_AK1,
        A0BlockTransferThreadClusterArrangeOrder,
        A0BlockTransferSrcAccessOrder,
        A0BlockTransferSrcVectorDim,
        A0BlockTransferSrcScalarPerVector,
        A0BlockTransferDstScalarPerVector_AK1,
        true,
        A0BlockLdsExtraM,
        B0BlockTransferThreadClusterLengths_BK0_N_BK1,
        B0BlockTransferThreadClusterArrangeOrder,
        B0BlockTransferSrcAccessOrder,
        B0BlockTransferSrcVectorDim,
        B0BlockTransferSrcScalarPerVector,
        B0BlockTransferDstScalarPerVector_BK1,
        true,
        B0BlockLdsExtraN,
        B1BlockTransferThreadClusterLengths_BK0_N_BK1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_BK1,
        false,
        B1BlockLdsExtraN,
        C1ShuffleMXdlPerWavePerShuffle,
        C1ShuffleGemm0NXdlPerWavePerShuffle,
        C1ShuffleBlockTransferClusterLengths_MBlock_Gemm0MPerBlock_NBlock_Gemm0NPerBlock,
        C1ShuffleBlockTransferScalarPerVector_Gemm0NPerBlock,
        LoopSched>;

    using A0GridDesc_AK0_M_AK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultA0GridDescriptor_AK0_M_AK1(A0GridDesc_M_K{}))>;
    using B0GridDesc_BK0_N_BK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultB0GridDescriptor_BK0_N_BK1(B0GridDesc_N_K{}))>;
    using B1GridDesc_BK0_N_BK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultB1GridDescriptor_BK0_N_BK1(B1GridDesc_N_K{}))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const A0DataType* p_a0_grid,
                 const B0DataType* p_b0_grid,
                 const B1DataType* p_b1_grid,
                 C1DataType* p_c1_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t Gemm1NRaw, // = ORaw
                 index_t Batch,
                 index_t StrideA0,
                 index_t StrideB0,
                 index_t StrideB1,
                 index_t StrideC1,
                 index_t BatchStrideA0,
                 index_t BatchStrideB0,
                 index_t BatchStrideB1,
                 index_t BatchStrideC1,
                 A0ElementwiseOperation a0_element_op,
                 B0ElementwiseOperation b0_element_op,
                 C0ElementwiseOperation c0_element_op,
                 B1ElementwiseOperation b1_element_op,
                 C1ElementwiseOperation c1_element_op)
            : p_a0_grid_{p_a0_grid},
              p_b0_grid_{p_b0_grid},
              p_b1_grid_{p_b1_grid},
              p_c1_grid_{p_c1_grid},
              a0_grid_desc_m_k_{DeviceOp::MakeA0GridDescriptor_M_K(MRaw, KRaw, StrideA0)},
              b0_grid_desc_n_k_{DeviceOp::MakeB0GridDescriptor_N_K(KRaw, NRaw, StrideB0)},
              b1_grid_desc_n_k_{DeviceOp::MakeB1GridDescriptor_N_K(NRaw, Gemm1NRaw, StrideB1)},
              c1_grid_desc_m_n_{DeviceOp::MakeC1GridDescriptor_M_N(MRaw, Gemm1NRaw, StrideC1)},
              a0_grid_desc_ak0_m_ak1_{
                  GridwiseGemm::MakeDefaultA0GridDescriptor_AK0_M_AK1(a0_grid_desc_m_k_)},
              b0_grid_desc_bk0_n_bk1_{
                  GridwiseGemm::MakeDefaultB0GridDescriptor_BK0_N_BK1(b0_grid_desc_n_k_)},
              b1_grid_desc_bk0_n_bk1_{
                  GridwiseGemm::MakeDefaultB1GridDescriptor_BK0_N_BK1(b1_grid_desc_n_k_)},
              c1_grid_desc_mblock_Gemm0MPerBlock_nblock_Gemm0NPerBlock_{},
              block_2_c1tile_map_{GridwiseGemm::MakeDefaultBlock2C1TileMap(c1_grid_desc_m_n_)},
              a0_element_op_{a0_element_op},
              b0_element_op_{b0_element_op},
              c0_element_op_{c0_element_op},
              b1_element_op_{b1_element_op},
              c1_element_op_{c1_element_op},
              batch_count_(Batch),
              compute_base_ptr_of_batch_{BatchStrideA0, BatchStrideB0, BatchStrideB1, BatchStrideC1}
        {
            if(GridwiseGemm::CheckValidity(a0_grid_desc_m_k_,
                                           b0_grid_desc_n_k_,
                                           b1_grid_desc_n_k_,
                                           c1_grid_desc_m_n_,
                                           block_2_c1tile_map_))
            {
                c1_grid_desc_mblock_Gemm0MPerBlock_nblock_Gemm0NPerBlock_ =
                    GridwiseGemm::MakeC1GridDescriptor_MBlock_Gemm0MPerBlock_NBlock_Gemm0NPerBlock(
                        c1_grid_desc_m_n_);
            }
        }

        //  private:
        // pointers
        const A0DataType* p_a0_grid_;
        const B0DataType* p_b0_grid_;
        const B1DataType* p_b1_grid_;
        C1DataType* p_c1_grid_;

        // tensor descriptors for problem definiton
        A0GridDesc_M_K a0_grid_desc_m_k_;
        B0GridDesc_N_K b0_grid_desc_n_k_;
        B1GridDesc_N_K b1_grid_desc_n_k_;
        C1GridDesc_M_N c1_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        A0GridDesc_AK0_M_AK1 a0_grid_desc_ak0_m_ak1_;
        B0GridDesc_BK0_N_BK1 b0_grid_desc_bk0_n_bk1_;
        B1GridDesc_BK0_N_BK1 b1_grid_desc_bk0_n_bk1_;
        typename GridwiseGemm::C1GridDescriptor_MBlock_Gemm0MPerBlock_NBlock_Gemm0NPerBlock
            c1_grid_desc_mblock_Gemm0MPerBlock_nblock_Gemm0NPerBlock_;

        // block-to-c-tile map
        typename GridwiseGemm::DefaultBlock2C1TileMap block_2_c1tile_map_;

        // element-wise op
        A0ElementwiseOperation a0_element_op_;
        B0ElementwiseOperation b0_element_op_;
        C0ElementwiseOperation c0_element_op_;
        B1ElementwiseOperation b1_element_op_;
        C1ElementwiseOperation c1_element_op_;

        // batch
        index_t batch_count_;
        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a0_grid_desc_m_k_,
                                            arg.b0_grid_desc_n_k_,
                                            arg.b1_grid_desc_n_k_,
                                            arg.c1_grid_desc_m_n_,
                                            arg.block_2_c1tile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_c1tile_map_.CalculateGridSize(arg.c1_grid_desc_m_n_) * arg.batch_count_;

            // Gemm0_K
            const auto K = arg.a0_grid_desc_m_k_.GetLength(I1);

            auto launch_kernel = [&](auto has_main_k_block_loop_) {
                const auto kernel = kernel_batched_gemm_gemm_xdl_cshuffle_v1<
                    GridwiseGemm,
                    A0DataType, // TODO: distiguish A/B datatype
                    C1DataType,
                    A0ElementwiseOperation,
                    B0ElementwiseOperation,
                    C0ElementwiseOperation,
                    B1ElementwiseOperation,
                    C1ElementwiseOperation,
                    DeviceOp::A0GridDesc_AK0_M_AK1,
                    DeviceOp::B0GridDesc_BK0_N_BK1,
                    DeviceOp::B1GridDesc_BK0_N_BK1,
                    typename GridwiseGemm::
                        C1GridDescriptor_MBlock_Gemm0MPerBlock_NBlock_Gemm0NPerBlock,
                    typename GridwiseGemm::DefaultBlock2C1TileMap,
                    ComputeBasePtrOfStridedBatch,
                    has_main_k_block_loop_>;

                return launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    arg.p_a0_grid_,
                    arg.p_b0_grid_,
                    arg.p_b1_grid_,
                    arg.p_c1_grid_,
                    arg.a0_element_op_,
                    arg.b0_element_op_,
                    arg.c0_element_op_,
                    arg.b1_element_op_,
                    arg.c1_element_op_,
                    arg.a0_grid_desc_ak0_m_ak1_,
                    arg.b0_grid_desc_bk0_n_bk1_,
                    arg.b1_grid_desc_bk0_n_bk1_,
                    arg.c1_grid_desc_mblock_Gemm0MPerBlock_nblock_Gemm0NPerBlock_,
                    arg.block_2_c1tile_map_,
                    arg.batch_count_,
                    arg.compute_base_ptr_of_batch_);
            };

            // Gemm1_K is split into Gemm1_K0/K1 where K1 is known at compile time, so we only need
            // to concern Gemm0's loop
            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
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
        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a"))
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(arg.a0_grid_desc_m_k_,
                                           arg.b0_grid_desc_n_k_,
                                           arg.b1_grid_desc_n_k_,
                                           arg.c1_grid_desc_m_n_,
                                           arg.block_2_c1tile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const A0DataType* p_a0,
                             const B0DataType* p_b0,
                             const B1DataType* p_b1,
                             C1DataType* p_c1,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t Gemm1NRaw,
                             index_t Batch,
                             index_t StrideA0,
                             index_t StrideB0,
                             index_t StrideB1,
                             index_t StrideC1,
                             index_t BatchStrideA0,
                             index_t BatchStrideB0,
                             index_t BatchStrideB1,
                             index_t BatchStrideC1,
                             A0ElementwiseOperation a0_element_op,
                             B0ElementwiseOperation b0_element_op,
                             C0ElementwiseOperation c0_element_op,
                             B1ElementwiseOperation b1_element_op,
                             C1ElementwiseOperation c1_element_op)
    {
        return Argument{p_a0,          p_b0,          p_b1,          p_c1,          MRaw,
                        NRaw,          KRaw,          Gemm1NRaw,     Batch,         StrideA0,
                        StrideB0,      StrideB1,      StrideC1,      BatchStrideA0, BatchStrideB0,
                        BatchStrideB1, BatchStrideC1, a0_element_op, b0_element_op, c0_element_op,
                        b1_element_op, c1_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a0,
                                                      const void* p_b0,
                                                      const void* p_b1,
                                                      void* p_c1,
                                                      index_t MRaw,
                                                      index_t NRaw,
                                                      index_t KRaw,
                                                      index_t Gemm1NRaw,
                                                      index_t Batch,
                                                      index_t StrideA0,
                                                      index_t StrideB0,
                                                      index_t StrideB1,
                                                      index_t StrideC1,
                                                      index_t BatchStrideA0,
                                                      index_t BatchStrideB0,
                                                      index_t BatchStrideB1,
                                                      index_t BatchStrideC1,
                                                      A0ElementwiseOperation a0_element_op,
                                                      B0ElementwiseOperation b0_element_op,
                                                      C0ElementwiseOperation c0_element_op,
                                                      B1ElementwiseOperation b1_element_op,
                                                      C1ElementwiseOperation c1_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const A0DataType*>(p_a0),
                                          static_cast<const B0DataType*>(p_b0),
                                          static_cast<const B1DataType*>(p_b1),
                                          static_cast<C1DataType*>(p_c1),
                                          MRaw,
                                          NRaw,
                                          KRaw,
                                          Gemm1NRaw,
                                          Batch,
                                          StrideA0,
                                          StrideB0,
                                          StrideB1,
                                          StrideC1,
                                          BatchStrideA0,
                                          BatchStrideB0,
                                          BatchStrideB1,
                                          BatchStrideC1,
                                          a0_element_op,
                                          b0_element_op,
                                          c0_element_op,
                                          b1_element_op,
                                          c1_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchedGemmGemm_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << Gemm0MPerBlock << ", "
            << Gemm0NPerBlock << ", "
            << Gemm0KPerBlock << ", "
            << A0K1 << ", "
            << B0K1 << ", "
            << B1K1 << ", "
            << Gemm0MPerXdl << ", "
            << Gemm0NPerXdl << ", "
            << Gemm0MXdlPerWave << ", "
            << Gemm0NXdlPerWave << ", "
            << Gemm1NXdlPerWave << "> ";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
