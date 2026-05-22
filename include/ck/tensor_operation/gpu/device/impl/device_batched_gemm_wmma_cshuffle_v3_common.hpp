// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common_kernels.hpp"
#include <optional>
#include <type_traits>

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename AsDataType,
          typename BsDataType,
          typename CDataType,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t BlockSize,
          index_t AK1,
          index_t BK1,
          GemmSpecialization GemmSpec,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename ComputeTypeA,
          typename ComputeTypeB,
          bool IsBScaled,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::index_t BPackedSize = ck::index_t{1},
          typename BScaleDataType = Tuple<>>
struct DeviceBatchedGemm_Wmma_CShuffleV3_Common
{
    struct ComputePtrOffsetOfStridedBatch
    {
        template <bool BScaled = IsBScaled, typename = typename std::enable_if_t<!BScaled>>
        ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                       index_t BatchStrideB,
                                       index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA), BatchStrideB_(BatchStrideB), BatchStrideC_(BatchStrideC)
        {
        }

        template <bool BScaled = IsBScaled, typename = typename std::enable_if_t<BScaled>>
        ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                       index_t BatchStrideB,
                                       index_t BatchStrideC,
                                       index_t BatchStrideScaleB)
            : BatchStrideA_(BatchStrideA),
              BatchStrideB_(BatchStrideB),
              BatchStrideC_(BatchStrideC),
              BatchStrideScaleB_(BatchStrideScaleB)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            static_assert(BPackedSize != 0);
            static_assert(IsBScaled || (!IsBScaled && BPackedSize == 1));
            return g_idx * static_cast<long_index_t>(BatchStrideB_) / BPackedSize;
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_);
        }

        __host__ __device__ constexpr long_index_t GetScaleBPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(*BatchStrideScaleB_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        index_t BatchStrideC_;
        std::optional<index_t> BatchStrideScaleB_;
    };

    struct Argument : public GridwiseGemm::Argument
    {
        using ADataType = typename AsDataType::DataType;
        using BDataType = typename BsDataType::DataType;
        template <bool BScaled = IsBScaled, typename = typename std::enable_if_t<!BScaled>>
        __host__ Argument(const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          CDataType* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_,
                          index_t BatchStrideA_,
                          index_t BatchStrideB_,
                          index_t BatchStrideC_,
                          index_t Batch_,
                          index_t k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CElementwiseOperation cde_element_op_,
                          bool is_reduce_ = false)
            : GridwiseGemm::Argument(std::array<const void*, 1>{p_a_grid_},
                                     std::array<const void*, 1>{p_b_grid_},
                                     std::array<const void*, 0>{}, // p_ds_grid_
                                     p_c_grid_,
                                     M_,
                                     N_,
                                     K_,
                                     std::array<index_t, 1>{StrideA_},
                                     std::array<index_t, 1>{StrideB_},
                                     std::array<index_t, 0>{}, // StrideDs_
                                     StrideC_,
                                     k_batch_,
                                     a_element_op_,
                                     b_element_op_,
                                     cde_element_op_,
                                     is_reduce_),
              Batch(Batch_),
              compute_ptr_offset_of_batch{BatchStrideA_, BatchStrideB_, BatchStrideC_}
        {
            static_assert(std::is_same_v<BScaleDataType, Tuple<>>);
        }

        template <bool BScaled = IsBScaled, typename = typename std::enable_if_t<BScaled>>
        __host__ Argument(const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          CDataType* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_,
                          index_t StrideScaleB_,
                          index_t BatchStrideA_,
                          index_t BatchStrideB_,
                          index_t BatchStrideC_,
                          index_t BatchStrideScaleB_,
                          const BScaleDataType* p_b_scale_grid_,
                          index_t Batch_,
                          index_t k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CElementwiseOperation c_element_op_,
                          bool is_reduce_ = false)
            : GridwiseGemm::Argument(std::array<const void*, 1>{p_a_grid_},
                                     std::array<const void*, 1>{p_b_grid_},
                                     std::array<const void*, 0>{}, // p_ds_grid_
                                     p_c_grid_,
                                     M_,
                                     N_,
                                     K_,
                                     std::array<index_t, 1>{StrideA_},
                                     std::array<index_t, 1>{StrideB_},
                                     std::array<index_t, 0>{}, // StrideDs_
                                     StrideC_,
                                     0, // StrideScaleA
                                     StrideScaleB_,
                                     nullptr,
                                     p_b_scale_grid_,
                                     k_batch_,
                                     a_element_op_,
                                     b_element_op_,
                                     c_element_op_,
                                     is_reduce_),
              Batch(Batch_),
              compute_ptr_offset_of_batch{
                  BatchStrideA_, BatchStrideB_, BatchStrideC_, BatchStrideScaleB_}
        {
            static_assert(!std::is_same_v<BScaleDataType, Tuple<>>);
        }

        index_t Batch;
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch;
    };

    /// @brief  Helper structure responsible for kernel invocation.
    ///
    /// @paragraph  The `Invoker` class is responsible for preparation and invocation of actual GPU
    ///             kernel function. It usually determines the launched grid size prepares kernel
    ///             arguments as well as perform specific kernel configuration selection based on
    ///             runtime arguments.
    ///
    /// @note       If appropriately configured it may measure kernel execution time.
    ///
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

            // The normal approach to batching would be to increase the grid size by just stretching
            // out the grid Z dimension (which is the outermost dimension), but this depends on
            // lower level functions not directly using the Z dimension for other calculations. As
            // it turns out, k batching does rely directly on blockIdx.Z through SplitKBatchOffset.
            // Therefore, for now we will use the grid Y dimension for batching. This may be a bit
            // fragile.
            gdy *= arg.Batch;

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

                    // Packed sizes are 1 for all implemented data types but we include it anyway
                    // for future compatibility.
                    // Note: the grid descriptors and size_a / size_b do *not* take batching into
                    // account, so we have to manually multiply overall buffer sizes for rotating
                    // memory by batch.
                    std::array<std::size_t, 1> size_as_buffers;
                    size_as_buffers[0] = a_grid_desc_ak0_m_ak1[Number<0>{}].GetElementSpaceSize() *
                                         GridwiseGemm::NumATensor / GridwiseGemm::APackedSize *
                                         arg_.Batch;

                    std::array<std::size_t, 1> size_bs_buffers;
                    size_bs_buffers[0] = b_grid_desc_bk0_n_bk1[Number<0>{}].GetElementSpaceSize() *
                                         GridwiseGemm::NumBTensor / GridwiseGemm::BPackedSize *
                                         arg_.Batch;

                    ck::utility::
                        RotatingMemWrapperMultiABD<Argument, AsDataType, BsDataType, Tuple<>>
                            rotating_mem(arg_,
                                         stream_config.rotating_count,
                                         size_as_buffers,
                                         size_bs_buffers,
                                         std::array<std::size_t, 0>{});
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        ck::utility::flush_icache();
                        rotating_mem.Next();
                        // clear c mem
                        if(arg_.KBatch > 1)
                            // Note: we multiply by batch since we want to clear the C matrix for
                            // the whole batch. Untested since we don't have k batching ATM.
                            // Note: This seems incorrect for non-contiguous memory layouts for C
                            // (padding, gaps).
                            HIP_CHECK_ERROR(
                                hipMemsetAsync(arg_.p_e_grid,
                                               0,
                                               arg_.Batch * arg_.M * arg_.N * sizeof(CDataType),
                                               stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg_,
                        arg_.compute_ptr_offset_of_batch);
                }
                else
                {
                    auto clear_workspace = [&]() {
                        // clear c mem
                        if(arg.KBatch > 1)
                            // Note: we multiply by batch since we want to clear the C matrix for
                            // the whole batch. Untested since we don't have k batching ATM.
                            // Note: This seems incorrect for non-contiguous memory layouts for C
                            // (padding, gaps).
                            HIP_CHECK_ERROR(
                                hipMemsetAsync(arg.p_e_grid,
                                               0,
                                               arg.Batch * arg.M * arg.N * sizeof(CDataType),
                                               stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        clear_workspace,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg,
                        arg.compute_ptr_offset_of_batch);
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

            using ComputePtrOffsetOfStridedBatch = decltype(arg.compute_ptr_offset_of_batch);
            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_batched_gemm_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ComputePtrOffsetOfStridedBatch,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy,
                            IsBScaled>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_batched_gemm_wmma_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<ComputePtrOffsetOfStridedBatch>,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            IsBScaled>;
                        Run(kernel);
                    }
                }
                else
                {
                    throw std::runtime_error("Pipeline not implemented");
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_batched_gemm_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ComputePtrOffsetOfStridedBatch,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy,
                            IsBScaled>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_batched_gemm_wmma_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<ComputePtrOffsetOfStridedBatch>,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            IsBScaled>;
                        Run(kernel);
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

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }

        if constexpr(std::is_same_v<CDataType, ck::half_t> ||
                     std::is_same_v<CDataType, ck::bhalf_t>)
        {
            if(arg.KBatch > 1 && ck::is_gfx11_supported())
            {
                // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
                return false;
            }
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                return false;
            }
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        if(ck::is_gfx12_supported() && !GridwiseGemm::CheckValidityAWaveTransfer(arg.M, arg.K))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Wave Transfer not applicable for matrix A" << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if(ck::is_gfx12_supported() && !GridwiseGemm::CheckValidityBWaveTransfer(arg.N, arg.K))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Wave Transfer not applicable for matrix B" << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        return GridwiseGemm::CheckValidity(arg);
    }

    template <ck::index_t MPerWmma,
              ck::index_t NPerWmma,
              ck::index_t MRepeat,
              ck::index_t NRepeat,
              ck::index_t ABlockTransferSrcScalarPerVector,
              ck::index_t BBlockTransferSrcScalarPerVector,
              typename ALayout,
              typename BLayout,
              typename CLayout>
    static std::string GetTypeString()
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

        constexpr auto type = []() {
            if constexpr(IsBScaled)
            {
                return "DeviceBatchedGemm_Wmma_CShuffleV3_BScale";
            }
            else
            {
                return "DeviceBatchedGemm_Wmma_CShuffleV3";
            }
        }();
        // clang-format off
        str << type
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
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
