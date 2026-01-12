// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multiple_d_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm_arraybase.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DeviceOp, typename GridwiseOp, bool HasMainKBlockLoop, TailNumber TailNum>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_batched_gemm_multiple_d_gemm_multiple_d_wmma_cshuffle_v3(typename DeviceOp::RawArg arg)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__))

    __shared__ char p_shared[GridwiseOp::GetSharedMemoryNumberOfByte()];
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / arg.batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetABasePtr(g_idx)));
    const long_index_t b0_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetB0BasePtr(g_idx)));
    const long_index_t b1_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetB1BasePtr(g_idx)));
    const long_index_t e1_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetE1BasePtr(g_idx)));

    auto p_d0s_grid = GridwiseOp::MakeD0sGridPointer();
    auto p_d1s_grid = GridwiseOp::MakeD1sGridPointer();

    static_for<0, DeviceOp::NumD0Tensor, 1>{}([&](auto In) {
        const long_index_t d0_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(arg.compute_base_ptr_of_batch.GetD0BasePtr(g_idx, In)));
        p_d0s_grid(In) = arg.p_d0s_grid(In) + d0_batch_offset;
    });

    static_for<0, DeviceOp::NumD1Tensor, 1>{}([&](auto In) {
        const long_index_t d1_batch_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(arg.compute_base_ptr_of_batch.GetD1BasePtr(g_idx, In)));
        p_d1s_grid(In) = arg.p_d1s_grid(In) + d1_batch_offset;
    });

    GridwiseOp::template Run<HasMainKBlockLoop, TailNum>(
        arg.p_a_grid + a_batch_offset,
        arg.p_b0_grid + b0_batch_offset,
        p_d0s_grid,
        arg.p_b1_grid + b1_batch_offset,
        p_d1s_grid,
        arg.p_e1_grid + e1_batch_offset,
        p_shared,
        arg.a_grid_desc,
        arg.b0_grid_desc,
        arg.d0s_grid_desc,
        arg.b1_grid_desc,
        arg.d1s_grid_desc_mblock_mperblock_nblock_nperblock,
        arg.e1_grid_desc_mblock_mperblock_nblock_nperblock,
        arg.a_element_op,
        arg.b0_element_op,
        arg.acc_element_op,
        arg.b1_element_op,
        arg.cde1_element_op,
        arg.block_2_etile_map);
#else
    ignore = arg;
#endif // (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__)
}

// Computes:
//         Acc = Acc_Op(A_Op(A) * B0_Op(B0), D0_0, D0_1, ...)
//         E = CDE1_Op(Acc_Op(Acc0) * B1_Op(B1), D1_0, D1_1, ...)
//
// Dimensions:
//         A        = MK
//         B0       = KL
//         Acc/D0s  = ML
//         B1       = LN
//         E/D1s    = MN
template <typename ALayout,
          typename B0layout,
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename E1Layout,
          typename ADataType,
          typename B0DataType,
          typename D0sDataType,
          typename B1DataType,
          typename D1sDataType,
          typename E1DataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t LPerBlock,     // Gemm0NPerBlock
          ck::index_t KPerBlock,     // Gemm0KPerBlock
          ck::index_t NPerBlock,     // Gemm1NPerBlock
          ck::index_t LTilePerBlock, // Gemm1KPerBlock
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t L1,       // B1K1
          ck::index_t MPerWmma, // Gemm0/1 MPerWmma
          ck::index_t LPerWmma, // Gemm0/1 NPerWmma
          ck::index_t MRepeat,  // Gemm0/1 MWmmaPerWave or Mrepeat
          ck::index_t LRepeat,  // Gemm0 NWmmaPerWave or Nrepeat
          ck::index_t NRepeat,  // Gemm1 NWmmaPerWave or Nrepeat
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
          ck::index_t CDE0BlockTransferSrcScalarPerVector,
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
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1>
struct DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3
    : public DeviceBatchedGemmMultipleDGemmMultipleD<ALayout,
                                                     B0layout,
                                                     D0sLayout,
                                                     B1Layout,
                                                     D1sLayout,
                                                     E1Layout,
                                                     ADataType,
                                                     B0DataType,
                                                     D0sDataType,
                                                     B1DataType,
                                                     D1sDataType,
                                                     E1DataType,
                                                     AElementwiseOperation,
                                                     B0ElementwiseOperation,
                                                     AccElementwiseOperation,
                                                     B1ElementwiseOperation,
                                                     CDE1ElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3;

    static constexpr index_t NumD0Tensor = D0sDataType::Size();
    static constexpr index_t NumD1Tensor = D1sDataType::Size();

    static constexpr auto I0 = Number<0>{};

    // To match XDL implementation NPerWmma (A.k.a Gemm1 NPerWmma) is set equal
    // to LPerWmma (A.k.a Gemm0 NPerWmma).
    static constexpr index_t NPerWmma = LPerWmma;

    // TODO: Now that we are no longer using NumDim or TensorSpec, we can probably use a simpler
    // Transform operator or just not use one at all.
    using Transform = TransformBatchedContractionContractionToBatchedGemmGemm_Wmma<
        Sequence<1, 1, 1, 1, 1>,
        Sequence<MPerBlock, LPerBlock, KPerBlock, NPerBlock>,
        GemmSpec,
        TensorSpecialization::Default,  // ASpec
        TensorSpecialization::Default,  // B0Spec
        TensorSpecialization::Default,  // B1Spec
        TensorSpecialization::Default>; // CSpec

    __host__ __device__ static auto
    MakeAGridDescriptor(const std::array<index_t, 3>& a_g_m_k_lengths_vec,
                        const std::array<index_t, 3>& a_g_m_k_strides_vec)
    {
        return Transform::MakeAGridDescriptor_AK0_M_AK1(
            Transform::MakeAGridDescriptor_M_K(a_g_m_k_lengths_vec, a_g_m_k_strides_vec),
            Number<AK1>{});
    }

    __host__ __device__ static auto
    MakeB0GridDescriptor(const std::array<index_t, 3>& b0_g_l_k_lengths_vec,
                         const std::array<index_t, 3>& b0_g_l_k_strides_vec)
    {
        return Transform::MakeB0GridDescriptor_BK0_N_BK1(
            Transform::MakeB0GridDescriptor_N_K(b0_g_l_k_lengths_vec, b0_g_l_k_strides_vec),
            Number<BK1>{});
    }

    __host__ __device__ static auto
    MakeB1GridDescriptor(const std::array<index_t, 3>& b1_g_n_l_lengths_vec,
                         const std::array<index_t, 3>& b1_g_n_l_strides_vec)
    {
        return Transform::MakeB1GridDescriptor_BK0_N_BK1(
            Transform::MakeB1GridDescriptor_N_K(b1_g_n_l_lengths_vec, b1_g_n_l_strides_vec),
            Number<L1>{});
    }

    __host__ __device__ static auto
    MakeD0GridDescriptor(const std::array<index_t, 3>& d0_g_m_n_lengths_vec,
                         const std::array<index_t, 3>& d0_g_m_n_strides_vec)
    {
        return Transform::MakeCGridDescriptor_M_N(d0_g_m_n_lengths_vec, d0_g_m_n_strides_vec);
    }

    __host__ __device__ static auto MakeD0sGridDescriptor(
        const std::array<std::array<index_t, 3>, NumD0Tensor>& d0_g_m_n_lengths_vec,
        const std::array<std::array<index_t, 3>, NumD0Tensor>& d0_g_m_n_strides_vec)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeD0GridDescriptor(d0_g_m_n_lengths_vec[i], d0_g_m_n_strides_vec[i]);
            },
            Number<NumD0Tensor>{});
    }

    __host__ __device__ static auto MakeD1sGridDescriptor(
        const std::array<std::array<index_t, 3>, NumD0Tensor>& d1_g_m_o_lengths_vec,
        const std::array<std::array<index_t, 3>, NumD0Tensor>& d1_g_m_o_strides_vec)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeE1GridDescriptor(d1_g_m_o_lengths_vec[i], d1_g_m_o_strides_vec[i]);
            },
            Number<NumD1Tensor>{});
    }

    __host__ __device__ static auto
    MakeE1GridDescriptor(const std::array<index_t, 3>& e1_g_m_n_lengths_vec,
                         const std::array<index_t, 3>& e1_g_m_n_strides_vec)
    {
        return Transform::MakeCGridDescriptor_M_N(e1_g_m_n_lengths_vec, e1_g_m_n_strides_vec);
    }

    using AGridDesc   = decltype(MakeAGridDescriptor({}, {}));
    using B0GridDesc  = decltype(MakeB0GridDescriptor({}, {}));
    using D0sGridDesc = remove_cvref_t<decltype(MakeD0sGridDescriptor({}, {}))>;
    using B1GridDesc  = decltype(MakeB1GridDescriptor({}, {}));
    using D1sGridDesc = remove_cvref_t<decltype(MakeD1sGridDescriptor({}, {}))>;
    using E1GridDesc  = decltype(MakeE1GridDescriptor({}, {}));

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(index_t BatchStrideA0,
                                     index_t BatchStrideB0,
                                     std::array<index_t, NumD0Tensor> BatchStrideD0s,
                                     index_t BatchStrideB1,
                                     std::array<index_t, NumD1Tensor> BatchStrideD1s,
                                     index_t BatchStrideE1)
            : BatchStrideA0_(BatchStrideA0),
              BatchStrideB0_(BatchStrideB0),
              BatchStrideD0s_(BatchStrideD0s),
              BatchStrideB1_(BatchStrideB1),
              BatchStrideD1s_(BatchStrideD1s),
              BatchStrideE1_(BatchStrideE1)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA0_);
        }

        __host__ __device__ constexpr long_index_t GetB0BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB0_);
        }

        template <index_t I>
        __host__ __device__ constexpr long_index_t GetD0BasePtr(index_t g_idx,
                                                                Number<I> d1_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideD0s_[d1_idx]);
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB1_);
        }

        __host__ __device__ constexpr long_index_t GetE1BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideE1_);
        }

        template <index_t I>
        __host__ __device__ constexpr auto GetD1BasePtr(index_t g_idx, Number<I> d1_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideD1s_[d1_idx]);
        }

        private:
        index_t BatchStrideA0_;
        index_t BatchStrideB0_;
        std::array<index_t, NumD0Tensor> BatchStrideD0s_;
        index_t BatchStrideB1_;
        std::array<index_t, NumD1Tensor> BatchStrideD1s_;
        index_t BatchStrideE1_;
    };

    // GridwiseOp
    using GridwiseOp = GridwiseBatchedGemmGemm_wmma_cshuffle_v3<
        // DataType Family
        ADataType,
        B0DataType,
        D0sDataType,
        AccDataType, // Acc0DataType
        B1DataType,
        D1sDataType,
        AccDataType, // Acc1DataType
        CShuffleDataType,
        E1DataType,
        // ElementwiseOp Family
        AElementwiseOperation,
        B0ElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CDE1ElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        // InMemory Data Descriptor
        AGridDesc,
        B0GridDesc,
        D0sGridDesc,
        B1GridDesc,
        D1sGridDesc,
        E1GridDesc,
        // Tiling Family
        MPerBlock,
        LPerBlock,
        KPerBlock,
        AK1,
        BK1,
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
        ABlockLdsAddExtraM,
        B0BlockTransferThreadClusterLengths_K0_L_K1,
        B0BlockTransferThreadClusterArrangeOrder,
        B0BlockTransferSrcAccessOrder,
        B0BlockTransferSrcVectorDim,
        B0BlockTransferSrcScalarPerVector,
        B0BlockTransferDstScalarPerVector_K1,
        true,
        B0BlockLdsAddExtraL,
        CDE0BlockTransferSrcScalarPerVector,
        B1BlockTransferThreadClusterLengths_L0_N_L1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_L1,
        false,
        B1BlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        Transform::matrix_padder.PadN,
        BlkGemmPipeSched,
        BlkGemmPipelineVer>;

    struct RawArg : public BaseArgument
    {
        using arr3 = std::array<ck::index_t, 3>;

        RawArg(const ADataType* p_a_grid_,
               const B0DataType* p_b0_grid_,
               std::array<const void*, NumD0Tensor> p_d0s_grid_,
               const B1DataType* p_b1_grid_,
               std::array<const void*, NumD1Tensor> p_d1s_grid_,
               E1DataType* p_e1_grid_,
               index_t M_,
               index_t N_,
               index_t K_,
               index_t O_,
               index_t Batch,
               index_t StrideA,
               index_t StrideB0,
               std::array<index_t, NumD0Tensor> StrideD0s,
               index_t StrideB1,
               std::array<index_t, NumD1Tensor> StrideD1s,
               index_t StrideE1,
               index_t BatchStrideA,
               index_t BatchStrideB0,
               std::array<index_t, NumD0Tensor> BatchStrideD0s,
               index_t BatchStrideB1,
               std::array<index_t, NumD1Tensor> BatchStrideD1s,
               index_t BatchStrideE1,
               AElementwiseOperation a_element_op_,
               B0ElementwiseOperation b0_element_op_,
               AccElementwiseOperation acc_element_op_,
               B1ElementwiseOperation b1_element_op_,
               CDE1ElementwiseOperation cde1_element_op_)
            : p_a_grid{p_a_grid_},
              p_b0_grid{p_b0_grid_},
              p_d0s_grid{},
              p_b1_grid{p_b1_grid_},
              p_d1s_grid{},
              p_e1_grid{p_e1_grid_},
              M{M_},
              N{N_},
              K{K_},
              O{O_},
              batch_count{Batch},
              a_element_op{a_element_op_},
              b0_element_op{b0_element_op_},
              acc_element_op{acc_element_op_},
              b1_element_op{b1_element_op_},
              cde1_element_op{cde1_element_op_},
              compute_base_ptr_of_batch{BatchStrideA,
                                        BatchStrideB0,
                                        BatchStrideD0s,
                                        BatchStrideB1,
                                        BatchStrideD1s,
                                        BatchStrideE1}
        {

            a_g_m_k_lengths = arr3{batch_count, M, K};
            a_g_m_k_strides = arr3{BatchStrideA, StrideA, 1}; // A layout [batch_count, M, K]

            b0_g_n_k_lengths = arr3{batch_count, N, K};
            b0_g_n_k_strides = arr3{BatchStrideB0, StrideB0, 1}; // B0 layout [batch_count, N, K]

            b1_g_o_n_lengths = arr3{batch_count, O, N};
            b1_g_o_n_strides =
                is_same_v<B1Layout, tensor_layout::gemm::RowMajor>
                    ? arr3{BatchStrideB1, 1, StrideB1}  // B1 layout [batch_count, N, O]
                    : arr3{BatchStrideB1, StrideB1, 1}; // B1 layout [batch_count, O, N]

            e1_g_m_o_lengths = arr3{batch_count, M, O};
            e1_g_m_o_strides = arr3{BatchStrideE1, StrideE1, 1}; // C layout [batch_count, M, O]

            a_grid_desc      = MakeAGridDescriptor(a_g_m_k_lengths, a_g_m_k_strides);
            b0_grid_desc     = MakeB0GridDescriptor(b0_g_n_k_lengths, b0_g_n_k_strides);
            b1_grid_desc     = MakeB1GridDescriptor(b1_g_o_n_lengths, b1_g_o_n_strides);
            e1_grid_desc_m_n = MakeE1GridDescriptor(e1_g_m_o_lengths, e1_g_m_o_strides);
            e1_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseOp::MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    e1_grid_desc_m_n);
            block_2_etile_map = GridwiseOp::MakeDefaultBlock2ETileMap(e1_grid_desc_m_n, 1, 1);

            static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                using D0DataType = remove_cvref_t<tuple_element_t<i.value, D0sDataType>>;

                // D0s layout [batch_count, M, N]
                d0s_g_m_n_lengths[i] = arr3{batch_count, M, N};
                d0s_g_m_n_strides[i] = arr3{BatchStrideD0s[i], StrideD0s[i], 1};

                // D0 pointer
                p_d0s_grid(i) = static_cast<const D0DataType*>(p_d0s_grid_[i]);

                // D0 desc
                d0s_grid_desc(i) = MakeD0GridDescriptor(d0s_g_m_n_lengths[i], d0s_g_m_n_strides[i]);
            });

            static_for<0, NumD1Tensor, 1>{}([&](auto i) {
                using D1DataType = remove_cvref_t<tuple_element_t<i.value, D1sDataType>>;

                // D1s layout [batch_count, M, O]
                d1s_g_m_o_lengths[i] = arr3{batch_count, M, O};
                d1s_g_m_o_strides[i] = arr3{BatchStrideD1s[i], StrideD1s[i], 1};

                // D1 pointer
                p_d1s_grid(i) = static_cast<const D1DataType*>(p_d1s_grid_[i]);

                // D1 desc
                d1s_grid_desc(i) = MakeE1GridDescriptor(d1s_g_m_o_lengths[i], d1s_g_m_o_strides[i]);
            });

            d1s_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseOp::MakeD1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(d1s_grid_desc);
        }

        // Pointers
        const ADataType* p_a_grid;
        const B0DataType* p_b0_grid;
        typename GridwiseOp::D0sGridPointer p_d0s_grid;
        const B1DataType* p_b1_grid;
        typename GridwiseOp::D1sGridPointer p_d1s_grid;
        E1DataType* p_e1_grid;

        // Raw Problem Size
        index_t M;
        index_t N;
        index_t K;
        index_t O;
        index_t batch_count;

        arr3 a_g_m_k_lengths;
        arr3 a_g_m_k_strides;
        arr3 b0_g_n_k_lengths;
        arr3 b0_g_n_k_strides;
        std::array<arr3, NumD0Tensor> d0s_g_m_n_lengths;
        std::array<arr3, NumD0Tensor> d0s_g_m_n_strides;
        arr3 b1_g_o_n_lengths;
        arr3 b1_g_o_n_strides;
        std::array<arr3, NumD1Tensor> d1s_g_m_o_lengths;
        std::array<arr3, NumD1Tensor> d1s_g_m_o_strides;
        arr3 e1_g_m_o_lengths;
        arr3 e1_g_m_o_strides;

        AElementwiseOperation a_element_op;
        B0ElementwiseOperation b0_element_op;
        AccElementwiseOperation acc_element_op;
        B1ElementwiseOperation b1_element_op;
        CDE1ElementwiseOperation cde1_element_op;

        // Grid descriptors and other mem calculators
        AGridDesc a_grid_desc;
        B0GridDesc b0_grid_desc;
        D0sGridDesc d0s_grid_desc;
        B1GridDesc b1_grid_desc;
        D1sGridDesc d1s_grid_desc;
        typename GridwiseOp::D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            d1s_grid_desc_mblock_mperblock_nblock_nperblock;

        E1GridDesc e1_grid_desc_m_n;
        typename GridwiseOp::E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e1_grid_desc_mblock_mperblock_nblock_nperblock;

        typename GridwiseOp::DefaultBlock2ETileMap block_2_etile_map;

        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch;
    };

    // check if DsLayout is supported
    template <typename RefLayout, typename DsLayout, const index_t NumDTensor>
    static constexpr bool CheckDLayout()
    {
        bool valid = true;
        // iterate over DLayout tuple
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
            // if RefLayout and DLayout are same, keep valid true, otherwise false
            valid = valid && is_same_v<RefLayout, DLayout>;
        });
        return valid;
    }

    static bool IsSupportedArgument([[maybe_unused]] const RawArg& arg)
    {
        // Print lambda with env check and printf() style formmating.
        const char* curFunc = __func__;
        auto print          = [&curFunc](const char* format, ...) -> void {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif
                va_list args;
                va_start(args, format);
                std::vfprintf(stdout, format, args);
                va_end(args);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
                std::cout << "In file: " << __FILE__ << ", function: " << curFunc << "\n";
            }
        };

        if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
        {
            print("DeviceOp: Arch err\n");
            return false;
        }

        if constexpr(std::is_same_v<ADataType, f8_t> || std::is_same_v<ADataType, bf8_t> ||
                     std::is_same_v<B0DataType, f8_t> || std::is_same_v<B0DataType, bf8_t> ||
                     std::is_same_v<B1DataType, f8_t> || std::is_same_v<B1DataType, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                print("DeviceOp: gfx 11 does not support fp8\n");
                return false;
            }
        }

        if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
        {
            print("DeviceOp: Acc0 Type err\n");
            return false;
        }

        if constexpr(!(is_same_v<ALayout, tensor_layout::gemm::RowMajor>))
        {
            print("DeviceOp: A layout must be Row\n");
            return false;
        }

        if constexpr(!(is_same_v<B0layout, tensor_layout::gemm::ColumnMajor>))
        {
            print("DeviceOp: B0 layout must be Column\n");
            return false;
        }

        if constexpr(!(CheckDLayout<tensor_layout::gemm::RowMajor, D0sLayout, NumD0Tensor>()))
        {
            print("DeviceOp: All D0s layout must be Row\n");
            return false;
        }

        if constexpr(!(is_same_v<B1Layout, tensor_layout::gemm::RowMajor> ||
                       is_same_v<B1Layout, tensor_layout::gemm::ColumnMajor>))
        {
            print("DeviceOp: B1 layout must be Column or Row\n");
            return false;
        }

        if constexpr(!(CheckDLayout<tensor_layout::gemm::RowMajor, D1sLayout, NumD1Tensor>()))
        {
            print("DeviceOp: All D1s layout must be Row\n");
            return false;
        }

        if constexpr(!(is_same_v<E1Layout, tensor_layout::gemm::RowMajor>))
        {
            print("DeviceOp: C layout must be Row\n");
            return false;
        }

        // Other padding modes have not been tested and do not get checked individually.
        if constexpr(GemmSpec != GemmSpecialization::Default &&
                     GemmSpec != GemmSpecialization::MNKOPadding)
        {
            print("Padding mode must be default or MNKO\n");
            return false;
        }

        // Per wmma dimensions not equal to 16 are very untested.
        if constexpr(MPerWmma != 16 || LPerWmma != 16 || NPerWmma != 16)
        {
            print("M, L, N per Wmma must be 16\n");
            return false;
        }

        if(!GridwiseOp::CheckValidity(arg.a_grid_desc,
                                      arg.b0_grid_desc,
                                      arg.d0s_grid_desc,
                                      arg.b1_grid_desc,
                                      arg.d1s_grid_desc,
                                      arg.e1_grid_desc_m_n,
                                      arg.block_2_etile_map))
        {
            return false;
        }

        // Check scalar per vector requirement
        const auto a_extent_lowest    = ABlockTransferSrcVectorDim == 2 ? arg.K : arg.M;
        const auto b0_extent_lowest   = B0BlockTransferSrcVectorDim == 2 ? arg.K : arg.N;
        const auto cde0_extent_lowest = arg.N; // D0 tensors forced to be row-major
        const auto b1_extent_lowest   = B1BlockTransferSrcVectorDim == 2 ? arg.N : arg.O;
        const auto cde1_extent_lowest = arg.O;

        if(!(a_extent_lowest % ABlockTransferSrcScalarPerVector == 0 &&
             b0_extent_lowest % B0BlockTransferSrcScalarPerVector == 0 &&
             cde0_extent_lowest % CDE0BlockTransferSrcScalarPerVector == 0 &&
             b1_extent_lowest % B1BlockTransferSrcScalarPerVector == 0 &&
             cde1_extent_lowest % CShuffleBlockTransferScalarPerVector_NPerBlock == 0))
        {
            print("DeviceOp: Data Transfer Vector scalar err\n");
            return false;
        }

        // Check vector load/store requirement
        const auto a_stride_lowest =
            ABlockTransferSrcVectorDim == 2 ? arg.a_g_m_k_strides[2] : arg.a_g_m_k_strides[1];
        const auto b0_stride_lowest =
            B0BlockTransferSrcVectorDim == 2 ? arg.b0_g_n_k_strides[2] : arg.b0_g_n_k_strides[1];
        const auto b1_stride_lowest =
            B1BlockTransferSrcVectorDim == 2 ? arg.b1_g_o_n_strides[2] : arg.b1_g_o_n_strides[1];
        const auto e1_stride_lowest = arg.e1_g_m_o_strides[2];

        // NOTE: We don't check D0s/D1s stride, as they are already forced to be row-major
        // and the lowest dimension stride is hardcoded to 1

        if(!(a_stride_lowest == 1 || b0_stride_lowest == 1 || b1_stride_lowest == 1 ||
             e1_stride_lowest == 1))
        {
            print("DeviceOp: Data Vectorize transfer err\n");
            return false;
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MNKOPadding))
        {
            return false;
        }

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const RawArg*>(p_arg));
    }

    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::RawArg;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto M0 = math::integer_divide_ceil(arg.M, MPerBlock);
            const auto N0 = math::integer_divide_ceil(arg.O, NPerBlock);

            const index_t grid_size = arg.batch_count * M0 * N0;

            auto launch_kernel = [&](auto has_main_k_block_loop, auto tail_number) {
                constexpr bool has_loop = decltype(has_main_k_block_loop)::value;
                constexpr TailNumber tn = tail_number;

                const auto kernel =
                    kernel_batched_gemm_multiple_d_gemm_multiple_d_wmma_cshuffle_v3<DeviceOp,
                                                                                    GridwiseOp,
                                                                                    has_loop,
                                                                                    tn>;

                return launch_and_time_kernel(
                    stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, arg);
            };

            bool HasMainKBlockLoop = GridwiseOp::CalculateHasMainKBlockLoop(arg.K);
            TailNumber TailNum     = GridwiseOp::CalculateKBlockLoopTailNum(arg.K);

            if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
            {
                if(HasMainKBlockLoop && TailNum == TailNumber::Full)
                {
                    return launch_kernel(std::integral_constant<bool, true>{},
                                         std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else if(!HasMainKBlockLoop && TailNum == TailNumber::Full)
                {
                    return launch_kernel(std::integral_constant<bool, false>{},
                                         std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else
                {
                    printf("Invalid HasMainKBlockLoop and TailNum combination for V1!\n");
                    return 0.0f;
                }
            }
            else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
            {
                if(HasMainKBlockLoop && TailNum == TailNumber::Full)
                {
                    return launch_kernel(std::integral_constant<bool, true>{},
                                         std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else if(!HasMainKBlockLoop && TailNum == TailNumber::Even)
                {
                    return launch_kernel(std::integral_constant<bool, false>{},
                                         std::integral_constant<TailNumber, TailNumber::Even>{});
                }
                else if(!HasMainKBlockLoop && TailNum == TailNumber::Odd)
                {
                    return launch_kernel(std::integral_constant<bool, false>{},
                                         std::integral_constant<TailNumber, TailNumber::Odd>{});
                }
                else
                {
                    printf("Invalid HasMainKBlockLoop and TailNum combination for V3!\n");
                    return 0.0f;
                }
            }
            else
            {
                printf("Invalid pipeline version!\n");
                return 0.0f;
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static auto MakeArgument(const ADataType* p_a0,
                             const B0DataType* p_b0,
                             std::array<const void*, NumD0Tensor> p_d0s,
                             const B1DataType* p_b1,
                             std::array<const void*, NumD1Tensor> p_d1s,
                             E1DataType* p_e1,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t Gemm1NRaw,
                             index_t Batch,
                             index_t StrideA0,
                             index_t StrideB0,
                             std::array<index_t, NumD0Tensor> StrideD0s,
                             index_t StrideB1,
                             std::array<index_t, NumD1Tensor> StrideD1s,
                             index_t StrideE1,
                             index_t BatchStrideA0,
                             index_t BatchStrideB0,
                             std::array<index_t, NumD0Tensor> BatchStrideD0s,
                             index_t BatchStrideB1,
                             std::array<index_t, NumD1Tensor> BatchStrideD1s,
                             index_t BatchStrideE1,
                             AElementwiseOperation a0_element_op,
                             B0ElementwiseOperation b0_element_op,
                             AccElementwiseOperation cde0_element_op,
                             B1ElementwiseOperation b1_element_op,
                             CDE1ElementwiseOperation cde1_element_op)
    {
        return RawArg{p_a0,          p_b0,
                      p_d0s,         p_b1,
                      p_d1s,         p_e1,
                      MRaw,          NRaw,
                      KRaw,          Gemm1NRaw,
                      Batch,         StrideA0,
                      StrideB0,      StrideD0s,
                      StrideB1,      StrideD1s,
                      StrideE1,      BatchStrideA0,
                      BatchStrideB0, BatchStrideD0s,
                      BatchStrideB1, BatchStrideD1s,
                      BatchStrideE1, a0_element_op,
                      b0_element_op, cde0_element_op,
                      b1_element_op, cde1_element_op};
    }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b0,
                        std::array<const void*, NumD0Tensor> p_d0s,
                        const void* p_b1,
                        std::array<const void*, NumD1Tensor> p_d1s,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t O,
                        ck::index_t Batch,
                        ck::index_t StrideA,
                        ck::index_t StrideB0,
                        std::array<index_t, NumD0Tensor> StrideD0s,
                        ck::index_t StrideB1,
                        std::array<index_t, NumD1Tensor> StrideD1s,
                        ck::index_t StrideE1,
                        ck::index_t BatchStrideA,
                        ck::index_t BatchStrideB0,
                        std::array<index_t, NumD0Tensor> BatchStrideD0s,
                        ck::index_t BatchStrideB1,
                        std::array<index_t, NumD1Tensor> BatchStrideD1s,
                        ck::index_t BatchStrideE1,
                        AElementwiseOperation a_element_op,
                        B0ElementwiseOperation b0_element_op,
                        AccElementwiseOperation acc_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CDE1ElementwiseOperation c_element_op) override
    {
        return std::make_unique<RawArg>(static_cast<const ADataType*>(p_a),
                                        static_cast<const B0DataType*>(p_b0),
                                        p_d0s,
                                        static_cast<const B1DataType*>(p_b1),
                                        p_d1s,
                                        static_cast<E1DataType*>(p_c),
                                        M,
                                        N,
                                        K,
                                        O,
                                        Batch,
                                        StrideA,
                                        StrideB0,
                                        StrideD0s,
                                        StrideB1,
                                        StrideD1s,
                                        StrideE1,
                                        BatchStrideA,
                                        BatchStrideB0,
                                        BatchStrideD0s,
                                        BatchStrideB1,
                                        BatchStrideD1s,
                                        BatchStrideE1,
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

    template <typename T>
    static constexpr const char* DataTypeToString()
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return "fp32";
        }
        else if constexpr(std::is_same_v<T, ck::half_t>)
        {
            return "fp16";
        }
        else if constexpr(std::is_same_v<T, ck::bhalf_t>)
        {
            return "bf16";
        }
        else if constexpr(std::is_same_v<T, ck::f8_t>)
        {
            return "fp8";
        }
        else if constexpr(std::is_same_v<T, ck::bf8_t>)
        {
            return "bf8";
        }
        else if constexpr(std::is_same_v<T, int32_t>)
        {
            return "int32";
        }
        else if constexpr(std::is_same_v<T, int8_t>)
        {
            return "int8";
        }
        else if constexpr(std::is_same_v<T, ck::int4_t>)
        {
            return "int4";
        }
        else
        {
            return "unknown";
        }
    }

    template <typename DataTypes>
    std::string DataTypeTupleToString() const
    {
        const auto string_types = generate_tuple(
            [&](auto i) {
                using ElementType = remove_cvref_t<tuple_element_t<i.value, DataTypes>>;
                return DataTypeToString<ElementType>();
            },
            Number<DataTypes::Size()>{});

        return TupleReduce<0, DataTypes::Size()>(
            [&](std::string s, std::string a) { return a + ", " + s; }, string_types);
    };

    template <typename Layouts>
    std::string LayoutTupleToString() const
    {
        const auto string_layouts = generate_tuple(
            [&](auto i) {
                using ElementLayout = remove_cvref_t<tuple_element_t<i.value, Layouts>>;
                return std::string(1, ElementLayout::name[0]);
            },
            Number<Layouts::Size()>{});

        return TupleReduce<0, Layouts::Size()>([&](std::string s, std::string a) { return a + s; },
                                               string_layouts);
    };

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3"
            << "<A/B0/B1/E: "
            << ALayout::name[0]
            << B0layout::name[0]
            << B1Layout::name[0]
            << E1Layout::name[0]  << ", "
            << "D0s: " << LayoutTupleToString<D0sLayout>() << " "
            << "D1s: " << LayoutTupleToString<D1sLayout>()
            << ", "
            << "A " << DataTypeToString<ADataType>() << ", "
            << "B0 " << DataTypeToString<B0DataType>() << ", "
            << "D0s (" << DataTypeTupleToString<D0sDataType>() << "), "
            << "B1 " << DataTypeToString<B1DataType>() << ", "
            << "D1s (" << DataTypeTupleToString<D1sDataType>() << "), "
            << "E1 " << DataTypeToString<E1DataType>() << ", "
            << "Acc " << DataTypeToString<AccDataType>() << ", "
            << "Cshuf " << DataTypeToString<CShuffleDataType>() << ", "
            << BlockSize << ", "
            << MPerBlock << ", "
            << LPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << LTilePerBlock << ", "
            << L1 << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseOp::BlockwiseGemmPipe::PrefetchStages 
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
