// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm_arraybase.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DeviceOp,
          typename GridwiseOp,
          typename B1Layout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_gemm_wmma_cshuffle_v3(const ADataType* __restrict__ p_a_grid,
                                                  const B0DataType* __restrict__ p_b0_grid,
                                                  const B1DataType* __restrict__ p_b1_grid,
                                                  CDataType* __restrict__ p_c_grid,
                                                  index_t M,
                                                  index_t N,
                                                  index_t K,
                                                  index_t O,
                                                  index_t batch_count)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__))

    // ***************************************************
    // Make Tensor Descriptors
    constexpr index_t array_size = 3;
    std::array<ck::index_t, array_size> a_gs_ms_ks_lengths{batch_count, M, K};
    std::array<ck::index_t, array_size> a_gs_ms_ks_strides =
        std::array<ck::index_t, array_size>{M * K, K, 1}; // A layout [batch_count, M, K]

    std::array<ck::index_t, array_size> b0_gs_ns_ks_lengths{batch_count, N, K};
    std::array<ck::index_t, array_size> b0_gs_ns_ks_strides =
        std::array<ck::index_t, array_size>{N * K, K, 1}; // B0 layout [batch_count, N, K]

    std::array<ck::index_t, array_size> b1_gs_os_ns_lengths{batch_count, O, N};
    std::array<ck::index_t, array_size> b1_gs_os_ns_strides =
        is_same_v<B1Layout, tensor_layout::gemm::RowMajor>
            ? std::array<ck::index_t, array_size>{N * O, 1, O}  // B1 layout [batch_count, N, O]
            : std::array<ck::index_t, array_size>{N * O, N, 1}; // B1 layout [batch_count, O, N]

    std::array<ck::index_t, array_size> c_gs_ms_os_lengths{batch_count, M, O};
    std::array<ck::index_t, array_size> c_gs_ms_os_strides =
        std::array<ck::index_t, array_size>{M * O, O, 1}; // C layout [batch_count, M, O]

    const auto a_element_op    = AElementwiseOperation{};
    const auto b0_element_op   = B0ElementwiseOperation{};
    const auto acc0_element_op = AccElementwiseOperation{};
    const auto b1_element_op   = B1ElementwiseOperation{};
    const auto c_element_op    = CElementwiseOperation{};
    // fail to reuse DeviceOp::MakeArgument() because of the __device__ function required.

    const auto a_grid_desc = DeviceOp::MakeAGridDescriptor(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    const auto b0_grid_desc =
        DeviceOp::MakeB0GridDescriptor(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    const auto b1_grid_desc =
        DeviceOp::MakeB1GridDescriptor(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    const auto c_grid_desc_m_n =
        DeviceOp::Transform::MakeCGridDescriptor_M_N(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
        GridwiseOp::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);
    const auto block_2_ctile_map = GridwiseOp::MakeDefaultBlock2CTileMap(c_grid_desc_m_n, 1, 1);

    const auto a_grid_desc_g_m_k =
        DeviceOp::Transform::MakeAGridDescriptor_G_M_K(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    const auto b0_grid_desc_g_l_k =
        DeviceOp::Transform::MakeB0GridDescriptor_G_N_K(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    const auto b1_grid_desc_g_n_l =
        DeviceOp::Transform::MakeB1GridDescriptor_G_N_K(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    const auto c_grid_desc_g_m_n =
        DeviceOp::Transform::MakeCGridDescriptor_G_M_N(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    const auto compute_base_ptr_of_batch = typename DeviceOp::ComputeBasePtrOfStridedBatch{
        a_grid_desc_g_m_k, b0_grid_desc_g_l_k, b1_grid_desc_g_n_l, c_grid_desc_g_m_n};

    __shared__ char p_shared[GridwiseOp::GetSharedMemoryNumberOfByte()];
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetABasePtr(g_idx)));
    const long_index_t b0_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetB0BasePtr(g_idx)));
    const long_index_t b1_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetB1BasePtr(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetCBasePtr(g_idx)));

    GridwiseOp::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                p_b0_grid + b0_batch_offset,
                                                p_b1_grid + b1_batch_offset,
                                                p_c_grid + c_batch_offset,
                                                p_shared,
                                                a_grid_desc,
                                                b0_grid_desc,
                                                b1_grid_desc,
                                                c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                a_element_op,
                                                b0_element_op,
                                                acc0_element_op,
                                                b1_element_op,
                                                c_element_op,
                                                block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b0_grid;
    ignore = p_b1_grid;
    ignore = p_c_grid;
    ignore = M;
    ignore = N;
    ignore = K;
    ignore = O;
    ignore = batch_count;
#endif // end of if (defined(__gfx11__))
}

// Computes C = A  * B0 * B1
//         MN = MK * KL * LN
//              ^^^^^^ (Acc0)
//              ^^^^^^^^^^^ (Acc1)
template <typename ALayout,
          typename BLayout, // B0Layout
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumPrefetch, // Not used, determined by pipeline version.
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
struct DeviceBatchedGemmGemm_Wmma_CShuffleV3 : public DeviceBatchedGemmGemm<ALayout,
                                                                            BLayout, // B0Layout
                                                                            B1Layout,
                                                                            CLayout,
                                                                            ADataType,
                                                                            B0DataType,
                                                                            B1DataType,
                                                                            CDataType,
                                                                            AElementwiseOperation,
                                                                            B0ElementwiseOperation,
                                                                            AccElementwiseOperation,
                                                                            B1ElementwiseOperation,
                                                                            CElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmGemm_Wmma_CShuffleV3;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    static constexpr auto WmmaK = 16;

    // To match XDL implementatrion NPerWmma (A.k.a Gemm1 NPerWmma) is set equal
    // to LPerWmma (A.k.a Gemm0 NPerWmma).
    static constexpr index_t NPerWmma = LPerWmma;

    static constexpr auto MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto LWaves = LPerBlock / (LRepeat * LPerWmma);
    static constexpr auto NWaves = NPerBlock / (NRepeat * NPerWmma);

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
    MakeAGridDescriptor(const std::array<index_t, 3>& a_gs_ms_ks_lengths_vec,
                        const std::array<index_t, 3>& a_gs_ms_ks_strides_vec)
    {
        return Transform::MakeAGridDescriptor_AK0_M_AK1(
            Transform::MakeAGridDescriptor_M_K(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec),
            Number<AK1>{});
    }

    __host__ __device__ static auto
    MakeB0GridDescriptor(const std::array<index_t, 3>& b0_gs_ls_ks_lengths_vec,
                         const std::array<index_t, 3>& b0_gs_ls_ks_strides_vec)
    {
        return Transform::MakeB0GridDescriptor_BK0_N_BK1(
            Transform::MakeB0GridDescriptor_N_K(b0_gs_ls_ks_lengths_vec, b0_gs_ls_ks_strides_vec),
            Number<BK1>{});
    }

    __host__ __device__ static auto
    MakeB1GridDescriptor(const std::array<index_t, 3>& b1_gs_ns_ls_lengths_vec,
                         const std::array<index_t, 3>& b1_gs_ns_ls_strides_vec)
    {
        return Transform::MakeB1GridDescriptor_BK0_N_BK1(
            Transform::MakeB1GridDescriptor_N_K(b1_gs_ns_ls_lengths_vec, b1_gs_ns_ls_strides_vec),
            Number<L1>{});
    }

    using AGridDesc        = decltype(MakeAGridDescriptor({}, {}));
    using B0GridDesc       = decltype(MakeB0GridDescriptor({}, {}));
    using B1GridDesc       = decltype(MakeB1GridDescriptor({}, {}));
    using CGridDesc_M_N    = decltype(Transform::MakeCGridDescriptor_M_N({}, {}));
    using AGridDesc_G_M_K  = decltype(Transform::MakeAGridDescriptor_G_M_K({}, {}));
    using B0GridDesc_G_L_K = decltype(Transform::MakeB0GridDescriptor_G_N_K({}, {}));
    using B1GridDesc_G_N_L = decltype(Transform::MakeB1GridDescriptor_G_N_K({}, {}));
    using CGridDesc_G_M_N  = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));

    struct ComputeBasePtrOfStridedBatch
    {
        __host__ __device__ ComputeBasePtrOfStridedBatch(const AGridDesc_G_M_K& a_grid_desc_g_m_k,
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
    using GridwiseOp = GridwiseBatchedGemmGemm_wmma_cshuffle_v3<
        // DataType Family
        ADataType,
        B0DataType,
        AccDataType, // Acc0DataType
        B1DataType,
        AccDataType, // Acc1DataType
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
        RawArg(const ADataType* p_a_grid,
               const B0DataType* p_b0_grid,
               const B1DataType* p_b1_grid,
               CDataType* p_c_grid,
               index_t M,
               index_t N,
               index_t K,
               index_t O,
               index_t Batch)
            : p_a_grid_{p_a_grid},
              p_b0_grid_{p_b0_grid},
              p_b1_grid_{p_b1_grid},
              p_c_grid_{p_c_grid},
              M_{M},
              N_{N},
              K_{K},
              O_{O},
              batch_count_{Batch}
        {
        }
        // Pointers
        const ADataType* p_a_grid_;
        const B0DataType* p_b0_grid_;
        const B1DataType* p_b1_grid_;
        CDataType* p_c_grid_;

        // Raw Problem Size
        index_t M_;
        index_t N_;
        index_t K_;
        index_t O_;
        index_t batch_count_;
    };

    // TODO: Check how padding modes are checked against problem dimensions for different layouts.
    static bool IsSupportedArgument(const RawArg& arg)
    {
        if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
        {
            printf("DeviceOp: Arch err");
            return false;
        }

        if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
        {
            printf("DeviceOp: Acc0 Type err");
            return false;
        }

        // TODO: Handle other layouts.
        if constexpr(!(is_same_v<ALayout, tensor_layout::gemm::RowMajor>))
        {
            printf("DeviceOp: A layout must be Row");
            return false;
        }

        if constexpr(!(is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>))
        {
            printf("DeviceOp: B layout must be Column");
            return false;
        }

        if constexpr(!(is_same_v<B1Layout, tensor_layout::gemm::RowMajor> ||
                       is_same_v<B1Layout, tensor_layout::gemm::ColumnMajor>))
        {
            printf("DeviceOp: B1 layout must be Column or Row");
            return false;
        }

        if constexpr(!(is_same_v<CLayout, tensor_layout::gemm::RowMajor>))
        {
            printf("DeviceOp: C layout must be Row");
            return false;
        }

        if constexpr(NumPrefetch != 1)
        {
            printf("NumPrefetch is not used!");
            return false;
        }

        constexpr index_t array_size = 3;
        ck::index_t G                = arg.batch_count_;
        ck::index_t M                = arg.M_;
        ck::index_t N                = arg.N_;
        ck::index_t K                = arg.K_;
        ck::index_t O                = arg.O_;

        std::array<ck::index_t, array_size> a_gs_ms_ks_lengths{G, M, K};
        std::array<ck::index_t, array_size> a_gs_ms_ks_strides =
            std::array<ck::index_t, array_size>{M * K, K, 1}; // A layout [G, M, K]

        std::array<ck::index_t, array_size> b0_gs_ns_ks_lengths{G, N, K};
        std::array<ck::index_t, array_size> b0_gs_ns_ks_strides =
            std::array<ck::index_t, array_size>{N * K, K, 1}; // B0 layout [G, N, K]

        std::array<ck::index_t, array_size> b1_gs_os_ns_lengths{G, O, N};
        std::array<ck::index_t, array_size> b1_gs_os_ns_strides =
            is_same_v<B1Layout, tensor_layout::gemm::RowMajor>
                ? std::array<ck::index_t, array_size>{N * O, 1, O}  // B1 layout [G, N, O]
                : std::array<ck::index_t, array_size>{N * O, N, 1}; // B1 layout [G, O, N]

        std::array<ck::index_t, array_size> c_gs_ms_os_lengths{G, M, O};
        std::array<ck::index_t, array_size> c_gs_ms_os_strides =
            std::array<ck::index_t, array_size>{M * O, O, 1}; // C layout [G, M, O]

        const auto a_grid_desc =
            DeviceOp::MakeAGridDescriptor(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
        const auto b0_grid_desc =
            DeviceOp::MakeB0GridDescriptor(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
        const auto b1_grid_desc =
            DeviceOp::MakeB1GridDescriptor(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
        const auto c_grid_desc_m_n =
            DeviceOp::Transform::MakeCGridDescriptor_M_N(c_gs_ms_os_lengths, c_gs_ms_os_strides);

        const auto block_2_ctile_map = GridwiseOp::MakeDefaultBlock2CTileMap(c_grid_desc_m_n, 1, 1);

        const auto c_grid_desc_g_m_n =
            DeviceOp::Transform::MakeCGridDescriptor_G_M_N(c_gs_ms_os_lengths, c_gs_ms_os_strides);
        index_t batch_count = c_grid_desc_g_m_n.GetLength(Number<0>{});

        if(!GridwiseOp::CheckValidity(
               a_grid_desc, b0_grid_desc, b1_grid_desc, c_grid_desc_m_n, block_2_ctile_map))
        {
            return false;
        }

        // Check if C permute dimension matches GEMM + GEMM shape
        const index_t c_g = c_grid_desc_g_m_n.GetLength(I0); // unpadded

        if(!(c_g == batch_count))
        {
            printf("DeviceOp: BatchCount err");
            return false;
        }

        // Note: we need raw lengths since threadwise copy can not handle vector load when part of
        // vector is out of bounds
        // Note: need lowest dim in Ms/Ns/Ks/Os, not merged M/N/K/O
        const auto MzRaw = M;
        const auto LzRaw = N;
        const auto KzRaw = K;
        const auto NzRaw = O;

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
            ABlockTransferSrcVectorDim == 2 ? a_gs_ms_ks_strides[2] : a_gs_ms_ks_strides[1];
        const auto b0_stride_lowest =
            B0BlockTransferSrcVectorDim == 2 ? b0_gs_ns_ks_strides[2] : b0_gs_ns_ks_strides[1];
        const auto b1_stride_lowest =
            B1BlockTransferSrcVectorDim == 2 ? b1_gs_os_ns_strides[2] : b1_gs_os_ns_strides[1];
        const auto c_stride_lowest = c_gs_ms_os_strides[2];

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
        return IsSupportedArgument(*dynamic_cast<const RawArg*>(p_arg));
    }

    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::RawArg;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto M0 = math::integer_divide_ceil(arg.M_, MPerBlock);
            const auto N0 = math::integer_divide_ceil(arg.O_, NPerBlock);

            const index_t grid_size = arg.batch_count_ * M0 * N0;
            const auto K            = arg.K_;
            // printf("HasKBlockLoop: %d\n", GridwiseOp::CalculateHasMainKBlockLoop(K));
            auto launch_kernel = [&](auto has_main_k_block_loop) {
                const auto kernel =
                    kernel_batched_gemm_gemm_wmma_cshuffle_v3<DeviceOp,
                                                              GridwiseOp,
                                                              B1Layout,
                                                              ADataType,
                                                              B0DataType,
                                                              B1DataType,
                                                              CDataType,
                                                              AElementwiseOperation,
                                                              B0ElementwiseOperation,
                                                              AccElementwiseOperation,
                                                              B1ElementwiseOperation,
                                                              CElementwiseOperation,
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
                                              arg.M_,
                                              arg.N_,
                                              arg.K_,
                                              arg.O_,
                                              arg.batch_count_);
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

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b0,
                                                      const void* p_b1,
                                                      void* p_c,
                                                      ck::index_t M,
                                                      ck::index_t N,
                                                      ck::index_t K,
                                                      ck::index_t O,
                                                      ck::index_t Batch,
                                                      ck::index_t StrideA,
                                                      ck::index_t StrideB0,
                                                      ck::index_t StrideB1,
                                                      ck::index_t StrideC,
                                                      ck::index_t BatchStrideA,
                                                      ck::index_t BatchStrideB0,
                                                      ck::index_t BatchStrideB1,
                                                      ck::index_t BatchStrideC,
                                                      AElementwiseOperation a_element_op,
                                                      B0ElementwiseOperation b0_element_op,
                                                      AccElementwiseOperation acc0_element_op,
                                                      B1ElementwiseOperation b1_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        (void)StrideA;
        (void)StrideB0;
        (void)StrideB1;
        (void)StrideC;
        (void)BatchStrideA;
        (void)BatchStrideB0;
        (void)BatchStrideB1;
        (void)BatchStrideC;
        (void)a_element_op;
        (void)b0_element_op;
        (void)acc0_element_op;
        (void)b1_element_op;
        (void)c_element_op; // TODO: Most of these arguments should actually be used / checked.
        return std::make_unique<RawArg>(static_cast<const ADataType*>(p_a),
                                        static_cast<const B0DataType*>(p_b0),
                                        static_cast<const B1DataType*>(p_b1),
                                        static_cast<CDataType*>(p_c),
                                        M,
                                        N,
                                        K,
                                        O,
                                        Batch);
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
        str << "DeviceBatchedGemmGemm_Wmma_CShuffleV3"
            << "<"
            << ALayout::name[0]
            << BLayout::name[0]
            << B1Layout::name[0]
            << CLayout::name[0] << ", "
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
            << getGemmSpecializationString(GemmSpec)
            << ">"
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseOp::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
