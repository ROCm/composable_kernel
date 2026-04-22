// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iostream>
#include <cstdarg>
#include <type_traits>
#include <utility>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm_arraybase.hpp"
#include "ck/utility/scheduler_enum.hpp"
#include "ck/utility/integral_constant.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DeviceOp,
          typename GridwiseOp,
          bool HasMainKBlockLoop,
          TailNumber TailNum,
          bool IsMultiD>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_batched_gemm_gemm_wmma_cshuffle_v3(typename DeviceOp::Argument arg)
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
    const long_index_t c_e1_batch_offset =
        __builtin_amdgcn_readfirstlane((arg.compute_base_ptr_of_batch.GetCE1BasePtr(g_idx)));

    auto [p_d0s_grid, p_d1s_grid] = [&]() {
        if constexpr(IsMultiD)
        {
            auto create_grid = [](auto NumTensor, auto func, auto& arg_grid, auto&& grid_pointer) {
                static_for<0, decltype(NumTensor)::value, 1>{}([&](auto In) {
                    const long_index_t batch_offset = __builtin_amdgcn_readfirstlane(func(In));
                    grid_pointer(In)                = arg_grid(In) + batch_offset;
                });
                return std::move(grid_pointer);
            };
            auto get_d0_base_ptr = [&arg, &g_idx](auto d_idx) {
                return arg.compute_base_ptr_of_batch.GetD0BasePtr(g_idx, d_idx);
            };
            auto get_d1_base_ptr = [&arg, &g_idx](auto d_idx) {
                return arg.compute_base_ptr_of_batch.GetD1BasePtr(g_idx, d_idx);
            };
            auto d0s_grid = create_grid(ck::integral_constant<ck::index_t, DeviceOp::NumD0Tensor>{},
                                        get_d0_base_ptr,
                                        arg.p_d0s_grid,
                                        GridwiseOp::MakeD0sGridPointer());
            auto d1s_grid = create_grid(ck::integral_constant<ck::index_t, DeviceOp::NumD1Tensor>{},
                                        get_d1_base_ptr,
                                        arg.p_d1s_grid,
                                        GridwiseOp::MakeD1sGridPointer());
            return std::make_pair(d0s_grid, d1s_grid);
        }
        else
        {
            return std::make_pair(Tuple<>{}, Tuple<>{});
        }
    }();

    GridwiseOp::template Run<HasMainKBlockLoop, TailNum>(
        arg.p_a_grid + a_batch_offset,
        arg.p_b0_grid + b0_batch_offset,
        p_d0s_grid,
        arg.p_b1_grid + b1_batch_offset,
        p_d1s_grid,
        arg.p_c_e1_grid + c_e1_batch_offset,
        p_shared,
        arg.a_grid_desc,
        arg.b0_grid_desc,
        arg.d0s_grid_desc,
        arg.b1_grid_desc,
        arg.d1s_grid_desc_mblock_mperblock_nblock_nperblock,
        arg.c_e1_grid_desc_mblock_mperblock_nblock_nperblock,
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

template <typename DeviceOp,
          GemmSpecialization GemmSpec,
          typename ALayout,
          typename B0layout,
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename CE1Layout,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t LPerBlock, // Gemm0NPerBlock
          ck::index_t KPerBlock, // Gemm0KPerBlock
          ck::index_t NPerBlock, // Gemm1NPerBlock
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename AccDataType,
          typename CE1DataType,
          typename D0sDataType,
          typename D1sDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t L1,       // B1K1
          ck::index_t MPerWmma, // Gemm0/1 MPerWmma
          ck::index_t LPerWmma, // Gemm0/1 NPerWmma
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t B0BlockTransferSrcVectorDim,
          ck::index_t B0BlockTransferSrcScalarPerVector,
          ck::index_t B1BlockTransferSrcVectorDim,
          ck::index_t B1BlockTransferSrcScalarPerVector,
          ck::index_t CDE0BlockTransferSrcScalarPerVector,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          bool IsMultiD = false>
struct DeviceGemmGemm_Wmma_CShuffleV3_Common
{
    static constexpr ck::index_t NumD0Tensor = []() {
        if constexpr(IsMultiD)
        {
            return DeviceOp::NumD0Tensor;
        }
        return 0;
    }();
    static constexpr ck::index_t NumD1Tensor = []() {
        if constexpr(IsMultiD)
        {
            return DeviceOp::NumD1Tensor;
        }
        return 0;
    }();

    struct GridDescriptorCreator
    {
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
            const std::array<std::array<index_t, 3>, NumD1Tensor>& d1_g_m_o_lengths_vec,
            const std::array<std::array<index_t, 3>, NumD1Tensor>& d1_g_m_o_strides_vec)
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
    };

    using AGridDesc  = decltype(GridDescriptorCreator::MakeAGridDescriptor({}, {}));
    using B0GridDesc = decltype(GridDescriptorCreator::MakeB0GridDescriptor({}, {}));
    using D0sGridDesc =
        remove_cvref_t<decltype(GridDescriptorCreator::MakeD0sGridDescriptor({}, {}))>;
    using B1GridDesc = decltype(GridDescriptorCreator::MakeB1GridDescriptor({}, {}));
    using D1sGridDesc =
        remove_cvref_t<decltype(GridDescriptorCreator::MakeD1sGridDescriptor({}, {}))>;
    using E1GridDesc = decltype(GridDescriptorCreator::MakeE1GridDescriptor({}, {}));
    using CGridDesc_M_N =
        decltype(GridDescriptorCreator::Transform::MakeCGridDescriptor_M_N({}, {}));

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(index_t BatchStrideA,
                                     index_t BatchStrideB0,
                                     index_t BatchStrideB1,
                                     index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA),
              BatchStrideB0_(BatchStrideB0),
              BatchStrideB1_(BatchStrideB1),
              BatchStrideC_E1_(BatchStrideC)
        {
        }

        ComputeBasePtrOfStridedBatch(index_t BatchStrideA0,
                                     index_t BatchStrideB0,
                                     std::array<index_t, NumD0Tensor> BatchStrideD0s,
                                     index_t BatchStrideB1,
                                     std::array<index_t, NumD1Tensor> BatchStrideD1s,
                                     index_t BatchStrideE1)
            : BatchStrideA_(BatchStrideA0),
              BatchStrideB0_(BatchStrideB0),
              BatchStrideD0s_(BatchStrideD0s),
              BatchStrideB1_(BatchStrideB1),
              BatchStrideD1s_(BatchStrideD1s),
              BatchStrideC_E1_(BatchStrideE1)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetB0BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB0_);
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB1_);
        }

        __host__ __device__ constexpr long_index_t GetCE1BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_E1_);
        }

        template <index_t I>
        __host__ __device__ constexpr long_index_t GetD0BasePtr(index_t g_idx,
                                                                Number<I> d0_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideD0s_[d0_idx]);
        }

        template <index_t I>
        __host__ __device__ constexpr long_index_t GetD1BasePtr(index_t g_idx,
                                                                Number<I> d1_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideD1s_[d1_idx]);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB0_;
        std::array<index_t, NumD0Tensor> BatchStrideD0s_;
        index_t BatchStrideB1_;
        std::array<index_t, NumD1Tensor> BatchStrideD1s_;
        index_t BatchStrideC_E1_;
    };
};

template <typename DeviceOp,
          GemmSpecialization GemmSpec,
          typename ALayout,
          typename B0layout,
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename CE1Layout,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t LPerBlock, // Gemm0NPerBlock
          ck::index_t KPerBlock, // Gemm0KPerBlock
          ck::index_t NPerBlock, // Gemm1NPerBlock
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename AccDataType,
          typename CE1DataType,
          typename D0sDataType,
          typename D1sDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDE1ElementwiseOperation,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t L1,       // B1K1
          ck::index_t MPerWmma, // Gemm0/1 MPerWmma
          ck::index_t LPerWmma, // Gemm0/1 NPerWmma
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t B0BlockTransferSrcVectorDim,
          ck::index_t B0BlockTransferSrcScalarPerVector,
          ck::index_t B1BlockTransferSrcVectorDim,
          ck::index_t B1BlockTransferSrcScalarPerVector,
          ck::index_t CDE0BlockTransferSrcScalarPerVector,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          bool IsMultiD = false>
struct DeviceGemmGemm_Wmma_CShuffleV3_Common_Invoker_Arg
{
    using GridwiseGemm = typename DeviceOp::GridwiseOp;
    using Common =
        DeviceGemmGemm_Wmma_CShuffleV3_Common<DeviceOp,
                                              GemmSpec,
                                              ALayout,
                                              B0layout,
                                              D0sLayout,
                                              B1Layout,
                                              D1sLayout,
                                              CE1Layout,
                                              BlockSize,
                                              MPerBlock,
                                              LPerBlock,
                                              KPerBlock,
                                              NPerBlock,
                                              ADataType,
                                              B0DataType,
                                              B1DataType,
                                              AccDataType,
                                              CE1DataType,
                                              D0sDataType,
                                              D1sDataType,
                                              AElementwiseOperation,
                                              B0ElementwiseOperation,
                                              AccElementwiseOperation,
                                              B1ElementwiseOperation,
                                              CDE1ElementwiseOperation,
                                              AK1,
                                              BK1,
                                              L1,
                                              MPerWmma,
                                              LPerWmma,
                                              BlkGemmPipelineVer,
                                              ABlockTransferSrcVectorDim,
                                              ABlockTransferSrcScalarPerVector,
                                              B0BlockTransferSrcVectorDim,
                                              B0BlockTransferSrcScalarPerVector,
                                              B1BlockTransferSrcVectorDim,
                                              B1BlockTransferSrcScalarPerVector,
                                              CDE0BlockTransferSrcScalarPerVector,
                                              CShuffleBlockTransferScalarPerVector_NPerBlock,
                                              IsMultiD>;

    static constexpr auto NumD0Tensor = Common::NumD0Tensor;
    static constexpr auto NumD1Tensor = Common::NumD1Tensor;

    struct Argument : public BaseArgument
    {
        using arr3 = std::array<ck::index_t, 3>;

        Argument(const ADataType* p_a_grid_,
                 const B0DataType* p_b0_grid_,
                 std::array<const void*, NumD0Tensor> p_d0s_grid_,
                 const B1DataType* p_b1_grid_,
                 std::array<const void*, NumD1Tensor> p_d1s_grid_,
                 CE1DataType* p_e1_grid_,
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
              p_c_e1_grid{p_e1_grid_},
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

            a_grid_desc  = Common::GridDescriptorCreator::MakeAGridDescriptor(a_g_m_k_lengths,
                                                                             a_g_m_k_strides);
            b0_grid_desc = Common::GridDescriptorCreator::MakeB0GridDescriptor(b0_g_n_k_lengths,
                                                                               b0_g_n_k_strides);
            b1_grid_desc = Common::GridDescriptorCreator::MakeB1GridDescriptor(b1_g_o_n_lengths,
                                                                               b1_g_o_n_strides);
            c_e1_grid_desc_m_n = Common::GridDescriptorCreator::MakeE1GridDescriptor(
                e1_g_m_o_lengths, e1_g_m_o_strides);
            c_e1_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseGemm::MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    c_e1_grid_desc_m_n);
            block_2_etile_map = GridwiseGemm::MakeDefaultBlock2ETileMap(c_e1_grid_desc_m_n, 1, 1);

            if constexpr(IsMultiD)
            {
                static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                    using D0DataType = remove_cvref_t<tuple_element_t<i.value, D0sDataType>>;

                    // D0s layout [batch_count, M, N]
                    d0s_g_m_n_lengths[i] = arr3{batch_count, M, N};
                    d0s_g_m_n_strides[i] = arr3{BatchStrideD0s[i], StrideD0s[i], 1};

                    // D0 pointer
                    p_d0s_grid(i) = static_cast<const D0DataType*>(p_d0s_grid_[i]);
                });
                // D0 desc
                d0s_grid_desc = Common::GridDescriptorCreator::MakeD0sGridDescriptor(
                    d0s_g_m_n_lengths, d0s_g_m_n_strides);

                static_for<0, NumD1Tensor, 1>{}([&](auto i) {
                    using D1DataType = remove_cvref_t<tuple_element_t<i.value, D1sDataType>>;

                    // D1s layout [batch_count, M, O]
                    d1s_g_m_o_lengths[i] = arr3{batch_count, M, O};
                    d1s_g_m_o_strides[i] = arr3{BatchStrideD1s[i], StrideD1s[i], 1};

                    // D1 pointer
                    p_d1s_grid(i) = static_cast<const D1DataType*>(p_d1s_grid_[i]);
                });
                // D1 desc
                d1s_grid_desc = Common::GridDescriptorCreator::MakeD1sGridDescriptor(
                    d1s_g_m_o_lengths, d1s_g_m_o_strides);

                d1s_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeD1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        d1s_grid_desc);
            }
        }

        // Pointers
        const ADataType* p_a_grid;
        const B0DataType* p_b0_grid;
        typename GridwiseGemm::D0sGridPointer p_d0s_grid;
        const B1DataType* p_b1_grid;
        typename GridwiseGemm::D1sGridPointer p_d1s_grid;
        CE1DataType* p_c_e1_grid;

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
        typename Common::AGridDesc a_grid_desc;
        typename Common::B0GridDesc b0_grid_desc;
        std::conditional_t<IsMultiD, typename Common::D0sGridDesc, Tuple<>> d0s_grid_desc;
        typename Common::B1GridDesc b1_grid_desc;
        typename Common::D1sGridDesc d1s_grid_desc;
        std::conditional_t<
            IsMultiD,
            typename GridwiseGemm::D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
            Tuple<>>
            d1s_grid_desc_mblock_mperblock_nblock_nperblock;

        std::conditional_t<IsMultiD, typename Common::E1GridDesc, typename Common::CGridDesc_M_N>
            c_e1_grid_desc_m_n;
        typename GridwiseGemm::E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_e1_grid_desc_mblock_mperblock_nblock_nperblock;

        typename GridwiseGemm::DefaultBlock2ETileMap block_2_etile_map;

        typename Common::ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch;
    };

    /// @brief  Helper structure responsible for kernel invocation.
    ///
    /// @paragraph  The `Invoker` class is responsible for preparation and invocation of actual GPU
    ///             kernel function. It usually determines the launched grid size prepares kernel
    ///             arguments as well as perform specific kernel configuration selection based on
    ///             runtime arguments.
    ///
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto M0 = math::integer_divide_ceil(arg.M, MPerBlock);
            const auto N0 = math::integer_divide_ceil(arg.O, NPerBlock);

            const index_t grid_size = arg.batch_count * M0 * N0;

            auto launch_kernel = [&](auto has_main_k_block_loop, auto tail_number) {
                constexpr bool has_loop       = decltype(has_main_k_block_loop)::value;
                constexpr TailNumber tail_num = decltype(tail_number)::value;
                const auto kernel             = kernel_batched_gemm_gemm_wmma_cshuffle_v3<DeviceOp,
                                                                                          GridwiseGemm,
                                                                                          has_loop,
                                                                                          tail_num,
                                                                                          IsMultiD>;
                return launch_and_time_kernel(
                    stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, arg);
            };

            bool HasMainKBlockLoop = GridwiseGemm::CalculateHasMainKBlockLoop(arg.K);
            TailNumber TailNum     = GridwiseGemm::CalculateKBlockLoopTailNum(arg.K);

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

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

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

    static bool IsSupportedArgument(const Argument& arg)
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

        if constexpr(!(is_same_v<B1Layout, tensor_layout::gemm::RowMajor> ||
                       is_same_v<B1Layout, tensor_layout::gemm::ColumnMajor>))
        {
            print("DeviceOp: B1 layout must be Column or Row\n");
            return false;
        }

        if constexpr(!(is_same_v<CE1Layout, tensor_layout::gemm::RowMajor>))
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
        if constexpr(MPerWmma != 16 || LPerWmma != 16 || DeviceOp::NPerWmma != 16)
        {
            print("M, L, N per Wmma must be 16\n");
            return false;
        }

        if constexpr(IsMultiD)
        {
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

            if constexpr(!(CheckDLayout<tensor_layout::gemm::RowMajor, D1sLayout, NumD1Tensor>()))
            {
                print("DeviceOp: All D1s layout must be Row\n");
                return false;
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc,
                                            arg.b0_grid_desc,
                                            arg.d0s_grid_desc,
                                            arg.b1_grid_desc,
                                            arg.d1s_grid_desc,
                                            arg.c_e1_grid_desc_m_n,
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
            const auto b0_stride_lowest = B0BlockTransferSrcVectorDim == 2
                                              ? arg.b0_g_n_k_strides[2]
                                              : arg.b0_g_n_k_strides[1];
            const auto b1_stride_lowest = B1BlockTransferSrcVectorDim == 2
                                              ? arg.b1_g_o_n_strides[2]
                                              : arg.b1_g_o_n_strides[1];
            const auto e1_stride_lowest = arg.e1_g_m_o_strides[2];

            // NOTE: We don't check D0s/D1s stride, as they are already forced to be row-major
            // and the lowest dimension stride is hardcoded to 1
            if(!(a_stride_lowest == 1 || b0_stride_lowest == 1 || b1_stride_lowest == 1 ||
                 e1_stride_lowest == 1))
            {
                print("DeviceOp: Data Vectorize transfer err\n");
                return false;
            }
        }
        else
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc,
                                            arg.b0_grid_desc,
                                            Tuple<>{},
                                            arg.b1_grid_desc,
                                            Tuple<>{},
                                            arg.c_e1_grid_desc_m_n,
                                            arg.block_2_etile_map))
            {
                return false;
            }

            // Check scalar per vector requirement
            const auto a_extent_lowest  = ABlockTransferSrcVectorDim == 2 ? arg.K : arg.M;
            const auto b0_extent_lowest = B0BlockTransferSrcVectorDim == 2 ? arg.K : arg.N;
            const auto b1_extent_lowest = B1BlockTransferSrcVectorDim == 2 ? arg.N : arg.O;
            const auto c_extent_lowest  = arg.O;

            if(!(a_extent_lowest % ABlockTransferSrcScalarPerVector == 0 &&
                 b0_extent_lowest % B0BlockTransferSrcScalarPerVector == 0 &&
                 b1_extent_lowest % B1BlockTransferSrcScalarPerVector == 0 &&
                 c_extent_lowest % CShuffleBlockTransferScalarPerVector_NPerBlock == 0))
            {
                print("DeviceOp: Data Transfer Vector scalar err\n");
                return false;
            }

            // Check vector load/store requirement
            const auto a_stride_lowest =
                ABlockTransferSrcVectorDim == 2 ? arg.a_g_m_k_strides[2] : arg.a_g_m_k_strides[1];
            const auto b0_stride_lowest = B0BlockTransferSrcVectorDim == 2
                                              ? arg.b0_g_n_k_strides[2]
                                              : arg.b0_g_n_k_strides[1];
            const auto b1_stride_lowest = B1BlockTransferSrcVectorDim == 2
                                              ? arg.b1_g_o_n_strides[2]
                                              : arg.b1_g_o_n_strides[1];
            const auto c_stride_lowest  = arg.e1_g_m_o_strides[2];

            if(!(a_stride_lowest == 1 || b0_stride_lowest == 1 || b1_stride_lowest == 1 ||
                 c_stride_lowest == 1))
            {
                print("DeviceOp: Data Vectorize transfer err\n");
                return false;
            }
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MNKOPadding))
        {
            return false;
        }

        return true;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#pragma clang diagnostic pop
