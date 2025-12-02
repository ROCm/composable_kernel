// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp"

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
    kernel_gemm_b_preshuffle_wmma_cshuffle_v3(typename GridwiseGemm::Argument karg)
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
        constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
            typename GridwiseGemm::EpilogueCShuffle>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        const index_t num_k_per_block = math::integer_divide_ceil(karg.K, GridwiseGemm::KPack);
        const index_t k_id            = blockIdx.z * num_k_per_block;

        auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared, splitk_batch_offset, karg, epilogue_args, k_id);

#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
#endif
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
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
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA>
struct DeviceGemmMultiD_Wmma_CShuffle_V3_BPreshuffle
    : public DeviceGemmMultipleDSplitKBPreShuffle<ALayout,
                                                  BLayout,
                                                  DsLayout,
                                                  ELayout,
                                                  ADataType,
                                                  BDataType,
                                                  DsDataType,
                                                  EDataType,
                                                  AElementwiseOperation,
                                                  BElementwiseOperation,
                                                  CDEElementwiseOperation>
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
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
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,
        false,
        true>;

    using Argument = typename GridwiseGemm::Argument;
    int GetPreShuffleParameters() override { return NPerWmma; }

    using DeviceGemmCommon =
        DeviceGemm_Wmma_CShuffleV3_Common<GridwiseGemm,
                                          Tuple<ADataType>,
                                          Tuple<BDataType>,
                                          DsDataType,
                                          EDataType,
                                          MPerBlock,
                                          NPerBlock,
                                          KPerBlock,
                                          BlockSize,
                                          AK1,
                                          BK1,
                                          GemmSpec,
                                          CDEShuffleBlockTransferScalarPerVectors,
                                          BlkGemmPipeSched,
                                          BlkGemmPipelineVer,
                                          ComputeTypeA,
                                          ComputeTypeB>;

    // Invoker
    struct Invoker : public BaseInvoker
    {
        /// @brief  This function issues GPU kernel execution.
        /// @param arg           The GPU kernel arguments.
        /// @param stream_config The HIP stream configuration helper structure.
        /// @return              The kernel's average execution time (if time measurement is
        ///                      enabled).
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
                GridwiseGemm::BlockwiseGemmPipe::HotLoopInstList::Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.KBatch);

            float ave_time = 0;

            index_t k_grain = arg.KBatch * KPerBlock;
            index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    Argument arg_ = arg;

                    const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAsGridDescriptor_AK0_M_AK1(
                        arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideAs, arg_.AK0);
                    const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBsGridDescriptor_BK0_N_BK1(
                        arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideBs, arg_.BK0);

                    std::array<std::size_t, 1> size_as_buffers;
                    size_as_buffers[Number<0>{}] =
                        a_grid_desc_ak0_m_ak1[Number<0>{}].GetElementSpaceSize() *
                        sizeof(ADataType) / GridwiseGemm::APackedSize;

                    std::array<std::size_t, 1> size_bs_buffers;
                    size_bs_buffers[Number<0>{}] =
                        b_grid_desc_bk0_n_bk1[Number<0>{}].GetElementSpaceSize() *
                        sizeof(BDataType) / GridwiseGemm::BPackedSize;

                    const auto ds_grid_desc_m_n = GridwiseGemm::MakeDsGridDescriptor_M_N(
                        arg_.M, arg_.MPadded, arg_.N, arg_.NPadded, arg_.StrideDs);

                    std::array<std::size_t, GridwiseGemm::NumDTensor> size_ds_buffers;
                    static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        size_ds_buffers[i] =
                            ds_grid_desc_m_n[i].GetElementSpaceSize() * sizeof(DDataType);
                    });

                    ck::utility::RotatingMemWrapperMultiABD<Argument,
                                                            Tuple<ADataType>,
                                                            Tuple<BDataType>,
                                                            DsDataType>
                        rotating_mem(arg_,
                                     stream_config.rotating_count,
                                     size_as_buffers,
                                     size_bs_buffers,
                                     size_ds_buffers);
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        if(arg_.KBatch > 1)
                            HIP_CHECK_ERROR(hipMemsetAsync(arg_.p_e_grid,
                                                           0,
                                                           arg_.M * arg_.N * sizeof(EDataType),
                                                           stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg_);
                }
                else
                {
                    if(arg.KBatch > 1)
                        HIP_CHECK_ERROR(hipMemsetAsync(arg.p_e_grid,
                                                       0,
                                                       arg.M * arg.N * sizeof(EDataType),
                                                       stream_config.stream_id_));

                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
                }
            };

            constexpr index_t minimum_occupancy = []() {
                if constexpr(BlkGemmPipeSched == BlockGemmPipelineScheduler::Interwave)
                {
                    return 2;
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    return (MPerBlock * NPerBlock / BlockSize <= 128) ? 2 : 1;
                }
                else
                {
                    return 1;
                }
            }();

            // ThreadwiseTensorSliceTransfer_v7r3 (used in ThreadGroupTensorSliceTransfer_v7r3) is
            // currently implemented in such a way that all SrcScalarPerVectors must be the same, so
            // if one of D matrices is column-major, then all SrcScalarPerVectors must be 1. On the
            // other hand, Split K for 16-bit outputs uses packed atomics so ScalarPerVectors cannot
            // be odd.
            constexpr bool AtomicsImplementationExists =
                !(std::is_same_v<EDataType, ck::half_t> || std::is_same_v<EDataType, ck::bhalf_t> ||
                  std::is_same_v<EDataType, int8_t>) ||
                (CDEShuffleBlockTransferScalarPerVectors{}[0] % 2 == 0);

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(arg.KBatch > 1)
                    {
                        if constexpr(AtomicsImplementationExists)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                            {
                                const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                GridwiseGemm,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(arg.KBatch > 1)
                    {
                        if constexpr(AtomicsImplementationExists)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                            {
                                const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    false,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                    GridwiseGemm,
                                    false,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                GridwiseGemm,
                                false,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_gemm_b_preshuffle_wmma_cshuffle_v3<
                                GridwiseGemm,
                                false,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(arg.N % NPerBlock != 0 || arg.K % KPerBlock != 0)
        {
            return false;
        }
        return DeviceGemmCommon::IsSupportedArgument(arg);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_e,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideE,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{std::array<const void*, 1>{p_a},
                        std::array<const void*, 1>{p_b},
                        p_ds,
                        static_cast<EDataType*>(p_e),
                        M,
                        N,
                        K,
                        std::array<index_t, 1>{StrideA},
                        std::array<index_t, 1>{StrideB},
                        StrideDs,
                        StrideE,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t StrideA,
                        index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        index_t StrideE,
                        index_t KBatch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(std::array<const void*, 1>{p_a},
                                          std::array<const void*, 1>{p_b},
                                          p_ds,
                                          static_cast<EDataType*>(p_e),
                                          M,
                                          N,
                                          K,
                                          std::array<index_t, 1>{StrideA},
                                          std::array<index_t, 1>{StrideB},
                                          StrideDs,
                                          StrideE,
                                          KBatch,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
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
        str << "DeviceGemmMultipleD_BPreshuffle_Wmma_CShuffleV3"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0];
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            str << std::string(DLayout::name)[0];
        });
        str << std::string(ELayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock << "x" << NPerBlock << "x" << KPerBlock << ", "
            << "WaveTile: "
            << MPerWmma << "x"<<NPerWmma << ", "
            << "WaveMap: "
            << MRepeat << "x" << NRepeat << ", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector << "x" << BBlockTransferSrcScalarPerVector << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages << ", "
            << "KPack: "
            << GridwiseGemm::KPack;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
