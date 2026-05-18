// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_waveletmodel_cshuffle_conv_v3.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_offset_utils.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// GPU kernel function for wave-specialized conv bwd weight
template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::LaunchBlockSize, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_weight_xdl_waveletmodel_cshuffle_v3(
        typename GridwiseGemm::Argument karg,
        const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
        const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock,
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
        const index_t num_k_per_block,
        const long_index_t split_k_stride_a,
        const long_index_t split_k_stride_b,
        bool split_k_offset_hack)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);
        const index_t k_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);

        const long_index_t split_k_offset_a = split_k_offset_hack ? k_idx * split_k_stride_a : 0;
        const long_index_t split_k_offset_b = split_k_offset_hack ? k_idx * split_k_stride_b : 0;

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t e_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        // Dispatch SplitKOffsetHack as a template parameter
        if(split_k_offset_hack)
        {
            GridwiseGemm::template Run<decltype(a_grid_desc_ak0_m_ak1),
                                       decltype(b_grid_desc_bk0_n_bk1),
                                       decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                                       HasMainKBlockLoop,
                                       CGlobalMemoryDataOperation,
                                       true>(karg.p_a_grid + a_batch_offset + split_k_offset_a,
                                             karg.p_b_grid + b_batch_offset + split_k_offset_b,
                                             karg.p_c_grid + e_batch_offset,
                                             p_shared,
                                             karg,
                                             a_grid_desc_ak0_m_ak1,
                                             b_grid_desc_bk0_n_bk1,
                                             c_grid_desc_mblock_mperblock_nblock_nperblock,
                                             k_idx * num_k_per_block,
                                             gridDim.y);
        }
        else
        {
            GridwiseGemm::template Run<decltype(a_grid_desc_ak0_m_ak1),
                                       decltype(b_grid_desc_bk0_n_bk1),
                                       decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                                       HasMainKBlockLoop,
                                       CGlobalMemoryDataOperation,
                                       false>(karg.p_a_grid + a_batch_offset + split_k_offset_a,
                                              karg.p_b_grid + b_batch_offset + split_k_offset_b,
                                              karg.p_c_grid + e_batch_offset,
                                              p_shared,
                                              karg,
                                              a_grid_desc_ak0_m_ak1,
                                              b_grid_desc_bk0_n_bk1,
                                              c_grid_desc_mblock_mperblock_nblock_nperblock,
                                              k_idx * num_k_per_block,
                                              gridDim.y);
        }
    }
#else
    ignore = karg;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = compute_ptr_offset_of_batch;
    ignore = num_k_per_block;
    ignore = split_k_stride_a;
    ignore = split_k_stride_b;
    ignore = split_k_offset_hack;
#endif
}

// Device operation for wave-specialized conv backward weight
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t TileLoadThreadGroupSize,
          ck::index_t TileMathThreadGroupSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          typename ComputeTypeA    = InDataType,
          typename ComputeTypeB    = ComputeTypeA,
          index_t NumGroupsToMerge = 1>
struct DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3
    : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementwiseOperation,
                                        WeiElementwiseOperation,
                                        OutElementwiseOperation,
                                        ComputeTypeA,
                                        ComputeTypeB>
{
    static_assert(is_same_v<InElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<WeiElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<OutElementwiseOperation, element_wise::PassThrough>);

    // AK0PerBlock = K0PerBlock / K1. Transfer cluster K0 dim must not exceed it,
    // otherwise thread_slice_k0 = AK0PerBlock / K0_cluster truncates to 0.
    static constexpr index_t AK0PerBlock = K0PerBlock / K1;
    static_assert(ABlockTransferThreadClusterLengths_K0_M_K1::At(0) <= AK0PerBlock,
                  "A transfer K0 cluster dim exceeds AK0PerBlock (= K0PerBlock / K1)");
    static_assert(BBlockTransferThreadClusterLengths_K0_N_K1::At(0) <= AK0PerBlock,
                  "B transfer K0 cluster dim exceeds AK0PerBlock (= K0PerBlock / K1)");

    // Transfer cluster product must match the load thread group size
    static_assert(ABlockTransferThreadClusterLengths_K0_M_K1::At(0) *
                          ABlockTransferThreadClusterLengths_K0_M_K1::At(1) *
                          ABlockTransferThreadClusterLengths_K0_M_K1::At(2) ==
                      TileLoadThreadGroupSize,
                  "A transfer cluster size must match load thread group size");
    static_assert(BBlockTransferThreadClusterLengths_K0_N_K1::At(0) *
                          BBlockTransferThreadClusterLengths_K0_N_K1::At(1) *
                          BBlockTransferThreadClusterLengths_K0_N_K1::At(2) ==
                      TileLoadThreadGroupSize,
                  "B transfer cluster size must match load thread group size");

    using DeviceOp = DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3;

    // BlockSize = TileMathThreadGroupSize for MFMA wave assignment (used by GET_NXDL_PER_WAVE_IMPL)
    static constexpr index_t BlockSize = TileMathThreadGroupSize;

    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    // Conv->GEMM naming: A=output grad, B=input, C=weight grad
    using ADataType = OutDataType;
    using BDataType = InDataType;
    using CDataType = WeiDataType;

    using AElementwiseOperation = OutElementwiseOperation;
    using BElementwiseOperation = InElementwiseOperation;
    using CElementwiseOperation = WeiElementwiseOperation;

    using ABDataType = InDataType;

    static inline auto I0 = Number<0>{};
    static inline auto I1 = Number<1>{};
    static inline auto I2 = Number<2>{};
    static inline auto I3 = Number<3>{};
    static inline auto I4 = Number<4>{};
    static inline auto I5 = Number<5>{};

    static constexpr GemmSpecialization GemmSpec = GemmSpecialization::Default;
    static constexpr auto K1Number               = Number<K1>{};

    // Launch block size = load + math thread groups
    static constexpr index_t LaunchBlockSize = TileLoadThreadGroupSize + TileMathThreadGroupSize;

    static constexpr auto conv_to_gemm_transformer =
        TransformConvBwdWeightToGemmV2<NDimSpatial,
                                       MPerBlock,
                                       NPerBlock,
                                       K1Number,
                                       K0PerBlock / K1Number,
                                       NumGroupsToMerge,
                                       ConvBackwardWeightSpecialization>{};

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_xdl_waveletmodel_cshuffle_conv_v3<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        ADataType,
        BDataType,
        AccDataType,
        CDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        NumGemmKPrefetchStage,
        TileLoadThreadGroupSize,
        TileMathThreadGroupSize,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        K1,
        K1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        ComputeTypeA,
        ComputeTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm64::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}, 1, 1));

    struct ActiveWorkgroupsPerCU
    {
        template <typename GridwiseGemm>
        static int GetMaxOccupancy()
        {
            // Query occupancy for the conservative variant (main loop + atomic)
            int max_occupancy = 0;
            hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_occupancy,
                SelectKernel<GridwiseGemm, true, InMemoryDataOperationEnum::AtomicAdd>(),
                LaunchBlockSize,
                0));
            return std::max(1, max_occupancy);
        }

        ActiveWorkgroupsPerCU()
        {
            max_occupancy_ = 1;
            if(get_warp_size() == 64)
            {
                if constexpr(NXdlPerWave64 > 0)
                    max_occupancy_ = GetMaxOccupancy<GridwiseGemm64>();
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                    max_occupancy_ = GetMaxOccupancy<GridwiseGemm32>();
            }
        }
        int max_occupancy_;
    };

    // Argument
    struct Argument : public BaseArgument, public ArgumentSplitK
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_in_grid},
              p_c_grid_{p_wei_grid},
              a_grid_desc_k0_m_k1_{},
              b_grid_desc_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              compute_ptr_offset_of_batch_{},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              c_element_op_{wei_element_op},
              Conv_G_{b_g_n_c_wis_lengths[0]},
              Conv_N_{b_g_n_c_wis_lengths[1]},
              Conv_K_{e_g_k_c_xs_lengths[1]},
              Conv_C_{b_g_n_c_wis_lengths[2]},
              input_spatial_lengths_{},
              filter_spatial_lengths_{},
              output_spatial_lengths_{},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            c_space_size_bytes =
                ck::accumulate_n<long_index_t>(
                    e_g_k_c_xs_lengths.begin(), NDimSpatial + I3, 1, std::multiplies<>()) *
                sizeof(WeiDataType);

            constexpr index_t spatial_offset = 3;
            std::copy(begin(b_g_n_c_wis_lengths) + spatial_offset,
                      end(b_g_n_c_wis_lengths),
                      begin(input_spatial_lengths_));
            std::copy(begin(e_g_k_c_xs_lengths) + spatial_offset,
                      end(e_g_k_c_xs_lengths),
                      begin(filter_spatial_lengths_));
            std::copy(begin(a_g_n_k_wos_lengths) + spatial_offset,
                      end(a_g_n_k_wos_lengths),
                      begin(output_spatial_lengths_));

            if(split_k < 0)
            {
                static ActiveWorkgroupsPerCU active_workgroups_per_cu;

                ck::index_t gemmM, gemmN, gemmK;
                std::tie(gemmM, gemmN, gemmK) =
                    get_bwd_weight_gemm_sizes<NDimSpatial>(a_g_n_k_wos_lengths, e_g_k_c_xs_lengths);

                const auto grid_size =
                    calculate_mn_grid_size<MPerBlock, NPerBlock>(gemmM, gemmN) * Conv_G_;
                this->k_batch_ = get_best_occupancy_k_batch_value(
                    active_workgroups_per_cu.max_occupancy_, grid_size);

                // Ensure k_batch_ does not exceed the maximum for the GEMM pipeline
                const auto k_batch_max = static_cast<index_t>((gemmK - 1) / K0PerBlock);
                this->k_batch_         = std::max(std::min(this->k_batch_, k_batch_max), 1);

                // Clamp to 128: higher split counts degrade accuracy because
                // inter-split accumulation uses data type (e.g. F16) not compute type (F32).
                this->k_batch_ = std::min(this->k_batch_, index_t{128});
            }
            else
            {
                this->k_batch_ = split_k;
            }

            // Create descriptors first (with hack flags temporarily set to false)
            // so we can check if element space sizes match product of dimensions
            const auto descs_initial =
                conv_to_gemm_transformer
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        this->k_batch_,
                        false, // split_k_offset_b_hack (temporary)
                        true); // use_full_batch_kindex=true for V1-compatible descriptors

            split_k_offset_hack_ =
                SplitKHackEligibility<NDimSpatial, InLayout, WeiLayout, OutLayout>::Check(
                    descs_initial[I0],
                    descs_initial[I1],
                    this->k_batch_,
                    Conv_N_,
                    output_spatial_lengths_,
                    K0PerBlock);

            // Now create descriptors with the correct hack flag
            const auto descs =
                conv_to_gemm_transformer
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        this->k_batch_,
                        split_k_offset_hack_,
                        true); // use_full_batch_kindex=true for V1-compatible descriptors

            a_grid_desc_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_     = descs[I2];

            // Calculate stride using GetElementSpaceSize for accurate stride
            split_k_stride_a_ = a_grid_desc_k0_m_k1_.GetElementSpaceSize();
            if(split_k_offset_hack_)
                split_k_stride_a_ /= this->k_batch_;

            split_k_stride_b_ = b_grid_desc_k0_n_k1_.GetElementSpaceSize();
            if(split_k_offset_hack_)
                split_k_stride_b_ /= this->k_batch_;

            // A/B/C Batch Stride (multiply by NumGroupsToMerge for group merging)
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides[0] * NumGroupsToMerge;
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_n_c_wis_strides[0] * NumGroupsToMerge;
            compute_ptr_offset_of_batch_.BatchStrideC_ =
                Conv_K_ * Conv_C_ *
                std::accumulate(begin(filter_spatial_lengths_),
                                end(filter_spatial_lengths_),
                                index_t{1},
                                std::multiplies<>{}) *
                NumGroupsToMerge;

            const index_t GemmM = a_grid_desc_k0_m_k1_.GetLength(I1);
            const index_t GemmN = b_grid_desc_k0_n_k1_.GetLength(I1);

            c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                GridwiseGemm64::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    c_grid_desc_m_n_,
                    GridwiseGemm64::CalculateMBlock(GemmM),
                    GridwiseGemm64::CalculateNBlock(GemmN));
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;
        ComputePtrOffsetOfStridedBatch<I1, I1, I0> compute_ptr_offset_of_batch_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        index_t Conv_G_;
        index_t Conv_N_;
        index_t Conv_K_;
        index_t Conv_C_;
        std::array<index_t, NDimSpatial> input_spatial_lengths_;
        std::array<index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<index_t, NDimSpatial> output_spatial_lengths_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;
        long_index_t c_space_size_bytes;
        bool split_k_offset_hack_      = false;
        long_index_t split_k_stride_a_ = 0;
        long_index_t split_k_stride_b_ = 0;
    };

    // Dispatch helper: selects the kernel instantiation for a given
    // {HasMainLoop, MemOp} combination, keeping common template args in one place.
    template <typename GridwiseGemm_, bool HasMainLoop, InMemoryDataOperationEnum MemOp>
    static auto SelectKernel()
    {
        return kernel_grouped_conv_bwd_weight_xdl_waveletmodel_cshuffle_v3<
            GridwiseGemm_,
            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
            remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
            HasMainLoop,
            MemOp>;
    }

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument& arg)
        {
            std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                      << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << ", "
                      << arg.a_grid_desc_k0_m_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                      << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << ", "
                      << arg.b_grid_desc_k0_n_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.c_grid_desc_m_n_{" << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const index_t GemmM = arg.a_grid_desc_k0_m_k1_.GetLength(I1);
            const index_t GemmN = arg.b_grid_desc_k0_n_k1_.GetLength(I1);
            const index_t GemmK =
                arg.a_grid_desc_k0_m_k1_.GetLength(I0) * arg.a_grid_desc_k0_m_k1_.GetLength(I2);

            const ADataType* p_a_grid = arg.p_a_grid_;
            const BDataType* p_b_grid = arg.p_b_grid_;
            typename GridwiseGemm::Argument gemm_arg{
                p_a_grid, p_b_grid, arg.p_c_grid_, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(
                gemm_arg.M, gemm_arg.N, gemm_arg.KBatch, arg.Conv_G_ / NumGroupsToMerge);

            float ave_time = 0;

            index_t k_grain                  = gemm_arg.KBatch * K0PerBlock;
            index_t K_split                  = (gemm_arg.K + k_grain - 1) / k_grain * K0PerBlock;
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto num_k_per_block =
                arg.a_grid_desc_k0_m_k1_.GetLength(Number<0>{}) / gemm_arg.KBatch;

            const auto clear_workspace = [&]() {
                if(arg.k_batch_ > 1)
                {
                    hip_check_error(hipMemsetAsync(
                        gemm_arg.p_c_grid, 0, arg.c_space_size_bytes, stream_config.stream_id_));
                }
            };

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    typename GridwiseGemm::Argument gemm_arg_ = gemm_arg;
                    ck::utility::RotatingMemWrapper<typename GridwiseGemm::Argument> rotating_mem(
                        gemm_arg_,
                        stream_config.rotating_count,
                        gemm_arg_.M * gemm_arg_.K * sizeof(ADataType),
                        gemm_arg_.K * gemm_arg_.N * sizeof(BDataType));
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        ck::utility::flush_icache();
                        rotating_mem.Next();
                        clear_workspace();
                    };
                    ave_time += ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(LaunchBlockSize),
                        0,
                        gemm_arg_,
                        arg.a_grid_desc_k0_m_k1_,
                        arg.b_grid_desc_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block,
                        arg.split_k_stride_a_,
                        arg.split_k_stride_b_,
                        arg.split_k_offset_hack_);
                }
                else
                {
                    ave_time += launch_and_time_kernel_with_preprocess(
                        stream_config,
                        clear_workspace,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(LaunchBlockSize),
                        0,
                        gemm_arg,
                        arg.a_grid_desc_k0_m_k1_,
                        arg.b_grid_desc_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block,
                        arg.split_k_stride_a_,
                        arg.split_k_stride_b_,
                        arg.split_k_offset_hack_);
                }
            };

            const bool split_k_active = arg.k_batch_ > 1;
            if(has_main_k_block_loop)
            {
                if(split_k_active)
                    Run(DeviceOp::SelectKernel<GridwiseGemm,
                                               true,
                                               InMemoryDataOperationEnum::AtomicAdd>());
                else
                    Run(DeviceOp::
                            SelectKernel<GridwiseGemm, true, InMemoryDataOperationEnum::Set>());
            }
            else
            {
                if(split_k_active)
                    Run(DeviceOp::SelectKernel<GridwiseGemm,
                                               false,
                                               InMemoryDataOperationEnum::AtomicAdd>());
                else
                    Run(DeviceOp::
                            SelectKernel<GridwiseGemm, false, InMemoryDataOperationEnum::Set>());
            }

            return ave_time;
        }

        INVOKER_RUN_IMPL

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter() { return true; }

    static bool IsSupportedArgument(const Argument& arg)
    {
        // Both thread groups must be a whole number of hardware waves.
        const index_t warp_size = get_warp_size();
        if(TileMathThreadGroupSize % warp_size != 0 || TileLoadThreadGroupSize % warp_size != 0)
        {
            return false;
        }

        if constexpr(NumGroupsToMerge > 1)
        {
            if(arg.Conv_G_ % NumGroupsToMerge != 0)
            {
                return false;
            }
        }

        if(!ck::is_xdl_wmma_supported<ComputeTypeA, ComputeTypeB, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if(is_gfx11_supported() && arg.k_batch_ > 1)
        {
            return false;
        }
        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> &&
           arg.k_batch_ > 1)
        {
            return false;
        }

        if constexpr(NDimSpatial == 1)
        {
            if constexpr(!is_GNWC_GKXC_GNWK<InLayout, WeiLayout, OutLayout>())
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            if constexpr(!(is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNHWC_GKYXC_GNHWK<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            if constexpr(!(is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        if(!(ABlockTransferSrcVectorDim == 1 && BBlockTransferSrcVectorDim == 1 &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        if(!(arg.Conv_C_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        {
            return false;
        }

        constexpr long_index_t TwoGB = (long_index_t{1} << 31);
        const bool a_small_enough    = arg.a_grid_desc_k0_m_k1_.GetElementSpaceSize() /
                                        (arg.split_k_offset_hack_ ? arg.k_batch_ : 1) *
                                        sizeof(ADataType) <=
                                    TwoGB;
        const bool b_small_enough = arg.b_grid_desc_k0_n_k1_.GetElementSpaceSize() /
                                        (arg.split_k_offset_hack_ ? arg.k_batch_ : 1) *
                                        sizeof(BDataType) <=
                                    TwoGB;
        const bool c_small_enough =
            arg.c_grid_desc_m_n_.GetElementSpaceSize() * sizeof(CDataType) <= TwoGB;
        if(!(a_small_enough && b_small_enough && c_small_enough))
        {
            return false;
        }

        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in_grid,
                             WeiDataType* p_wei_grid,
                             const OutDataType* p_out_grid,
                             const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths,
                             const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                             const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths,
                             const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                             const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                             const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                             const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                             const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                             const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                             const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op,
                             const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        b_g_n_c_wis_lengths,
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths,
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths,
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          b_g_n_c_wis_lengths,
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths,
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths,
                                          a_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3"
            << "<";
        str << TileLoadThreadGroupSize << "l+" << TileMathThreadGroupSize << "m, "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization) << ", "
            << K1 << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << ABlockTransferDstScalarPerVector_K1 << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << BBlockTransferDstScalarPerVector_K1 << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << CBlockTransferScalarPerVector_NWaveNPerXdl;
        if constexpr(NumGroupsToMerge > 1)
            str << ", " << NumGroupsToMerge;
        str << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
