// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck {

template <typename GridwiseGemm,
          typename ReduceTrait,
          typename ComputePtrOffsetOfStridedBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_batched_gemm_reduce_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg,
        typename ReduceTrait::ReducePtrsGlobal_ p_reduces_grid,
        const typename ReduceTrait::ReduceInElementwiseOperations_ reduce_in_element_ops,
        const typename ReduceTrait::ReduceAccElementwiseOperations_ reduce_out_element_ops,
        const ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch)
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
        using EpilogueType = typename GridwiseGemm::template EpilogueReduceCShuffle<ReduceTrait>;
        constexpr index_t LDS_size =
            GridwiseGemm::template GetSharedMemoryNumberOfByte<EpilogueType>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        const index_t g_idx = amd_wave_read_first_lane(blockIdx.y);

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t c_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx));

        auto reduces_batch = p_reduces_grid;
        compute_ptr_offset_of_batch.OffsetReducePtrs(g_idx, reduces_batch);

        typename GridwiseGemm::AsGridPointer p_as_grid_shift;
        static_for<0, GridwiseGemm::NumATensor, 1>{}([&](auto i) {
            using ADataType_ =
                remove_cvref_t<tuple_element_t<i.value, typename GridwiseGemm::AsDataType_>>;
            p_as_grid_shift(i) = static_cast<const ADataType_*>(karg.p_as_grid[i]) +
                                 splitk_batch_offset.a_k_split_offset[i] + a_batch_offset;
        });

        typename GridwiseGemm::BsGridPointer p_bs_grid_shift;
        static_for<0, GridwiseGemm::NumBTensor, 1>{}([&](auto i) {
            using BDataType_ =
                remove_cvref_t<tuple_element_t<i.value, typename GridwiseGemm::BsDataType_>>;
            p_bs_grid_shift(i) = static_cast<const BDataType_*>(karg.p_bs_grid[i]) +
                                 splitk_batch_offset.b_k_split_offset[i] + b_batch_offset;
        });

        auto epilogue_args = EpilogueType(reduces_batch,
                                          reduce_in_element_ops,
                                          reduce_out_element_ops,
                                          karg.M,
                                          tensor_operation::element_wise::PassThrough{});

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_as_grid_shift,
            p_bs_grid_shift,
            karg.p_ds_grid,
            karg.p_e_grid + splitk_batch_offset.c_reduce_offset + c_batch_offset,
            p_shared,
            karg,
            karg.a_element_op,
            karg.b_element_op,
            karg.cde_element_op,
            epilogue_args);
#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
    ignore = p_reduces_grid;
    ignore = reduce_in_element_ops;
    ignore = reduce_out_element_ops;
    ignore = compute_ptr_offset_of_batch;
#endif
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename ReduceAccDataType, // Reduce
          typename ReducePtrsGlobal,  // Reduce
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ReduceOperations,                // Reduce
          typename ReduceInElementwiseOperations,   // Reduce
          typename ReduceAccElementwiseOperations,  // Reduce
          typename ReduceGlobalMemoryDataOperation, // Reduce
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
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector,
          typename CReduceThreadClusterLengths_MPerBlock_NPerBlock,            // Reduce
          index_t CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,    // Reduce
          index_t CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock, // Reduce
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceBatchedGemmReduce_Wmma_CShuffleV3
    : public DeviceGemmReduce<0, ReduceOperations::Size()>
{
    using DeviceOp = DeviceBatchedGemmReduce_Wmma_CShuffleV3;

    static_assert(PermuteA == false,
                  "Permute A functionality not supported by DeviceBatchedGemm operations.\n");
    static_assert(PermuteB == false,
                  "Permute B functionality not supported by DeviceBatchedGemm operations.\n");

    using CDEShuffleBlockTransferScalarPerVectors =
        Sequence<CDEShuffleBlockTransferScalarPerVector,
                 CDEShuffleBlockTransferScalarPerVector,
                 CDEShuffleBlockTransferScalarPerVector>;

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        Tuple<>,
        ELayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        Tuple<>,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
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
        PermuteA,
        PermuteB,
        false,
        false,
        true>;

    using ReduceTrait = ReduceTrait_<ReduceAccDataType,
                                     ReducePtrsGlobal,
                                     tensor_operation::element_wise::PassThrough,
                                     ReduceOperations,
                                     ReduceInElementwiseOperations,
                                     ReduceAccElementwiseOperations,
                                     ReduceGlobalMemoryDataOperation,
                                     CReduceThreadClusterLengths_MPerBlock_NPerBlock,
                                     CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
                                     CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock>;

    static constexpr index_t NumReduce = ReduceOperations::Size();

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(long_index_t BatchStrideA,
                                       long_index_t BatchStrideB,
                                       long_index_t BatchStrideC,
                                       std::array<long_index_t, NumReduce> BatchStrideReduce)
            : BatchStrideA_{BatchStrideA},
              BatchStrideB_{BatchStrideB},
              BatchStrideC_{BatchStrideC},
              BatchStrideReduce_{BatchStrideReduce}
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * BatchStrideA_;
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return g_idx * BatchStrideB_;
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            return g_idx * BatchStrideC_;
        }

        template <typename ReducePtrs>
        __host__ __device__ void OffsetReducePtrs(index_t g_idx, ReducePtrs& ptrs) const
        {
            static_for<0, NumReduce, 1>{}(
                [&](auto I) { ptrs(I) = ptrs(I) + g_idx * BatchStrideReduce_[I.value]; });
        }

        private:
        long_index_t BatchStrideA_;
        long_index_t BatchStrideB_;
        long_index_t BatchStrideC_;
        std::array<long_index_t, NumReduce> BatchStrideReduce_{};
    };

    private:
    static long_index_t ComputeABatchStride(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
        {
            return static_cast<long_index_t>(MRaw) * StrideA;
        }
        else
        {
            return static_cast<long_index_t>(KRaw) * StrideA;
        }
    }

    static long_index_t ComputeBBatchStride(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
        {
            return static_cast<long_index_t>(KRaw) * StrideB;
        }
        else
        {
            return static_cast<long_index_t>(NRaw) * StrideB;
        }
    }

    static long_index_t ComputeCBatchStride(index_t MRaw, index_t NRaw, index_t StrideC)
    {
        if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ELayout>)
        {
            return static_cast<long_index_t>(MRaw) * StrideC;
        }
        else
        {
            return static_cast<long_index_t>(NRaw) * StrideC;
        }
    }

    public:
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 EDataType* p_e_grid,
                 ReducePtrsGlobal p_reduces_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t Batch,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 ReduceInElementwiseOperations reduce_in_element_ops,
                 ReduceAccElementwiseOperations reduce_out_element_ops,
                 std::array<long_index_t, NumReduce> batch_stride_reduce)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_e_grid_{p_e_grid},
              p_reduces_grid_{p_reduces_grid},
              MRaw_{MRaw},
              NRaw_{NRaw},
              KRaw_{KRaw},
              StrideA_{StrideA},
              StrideB_{StrideB},
              StrideC_{StrideC},
              Batch_{Batch},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              reduce_in_element_ops_{reduce_in_element_ops},
              reduce_out_element_ops_{reduce_out_element_ops},
              batch_stride_reduce_{batch_stride_reduce},
              compute_ptr_offset_of_batch_(
                  ComputePtrOffsetOfStridedBatch{ComputeABatchStride(MRaw, KRaw, StrideA),
                                                 ComputeBBatchStride(KRaw, NRaw, StrideB),
                                                 ComputeCBatchStride(MRaw, NRaw, StrideC),
                                                 batch_stride_reduce})
        {
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        EDataType* p_e_grid_;
        ReducePtrsGlobal p_reduces_grid_;
        index_t MRaw_;
        index_t NRaw_;
        index_t KRaw_;
        index_t StrideA_;
        index_t StrideB_;
        index_t StrideC_;
        index_t Batch_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        ReduceInElementwiseOperations reduce_in_element_ops_;
        ReduceAccElementwiseOperations reduce_out_element_ops_;
        std::array<long_index_t, NumReduce> batch_stride_reduce_{};
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            typename GridwiseGemm::Argument gemm_arg{
                std::array<const void*, 1>{arg.p_a_grid_},
                std::array<const void*, 1>{arg.p_b_grid_},
                std::array<const void*, 0>{},
                static_cast<EDataType*>(arg.p_e_grid_),
                arg.MRaw_,
                arg.NRaw_,
                arg.KRaw_,
                std::array<index_t, 1>{arg.StrideA_}, // StrideAs
                std::array<index_t, 1>{arg.StrideB_}, // StrideBs
                std::array<index_t, 0>{},             // StrideDs
                arg.StrideC_,                         // StrideC
                1,                                    // kbatch
                arg.a_element_op_,
                arg.b_element_op_,
                arg.c_element_op_};

            if(stream_config.log_level_ > 0)
            {
                gemm_arg.Print();
                GridwiseGemm::BlockwiseGemmPipe::HotLoopInstList::Print();
            }

            if(!GridwiseGemm::CheckValidity(gemm_arg, true))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.MRaw_, arg.NRaw_, 1);

            gdy *= arg.Batch_;

            float ave_time = 0;

            const index_t K_split = (arg.KRaw_ + KPerBlock - 1) / KPerBlock * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);
            const TailNumber tail_num        = GridwiseGemm::CalculateKBlockLoopTailNum(K_split);

            const auto Run = [&](const auto& kernel) {
                // Note: cache flushing not supported

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   gemm_arg,
                                                   arg.p_reduces_grid_,
                                                   arg.reduce_in_element_ops_,
                                                   arg.reduce_out_element_ops_,
                                                   arg.compute_ptr_offset_of_batch_);
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

            auto CreateAndRunKernel = [&](auto has_main_k_block_loop_, auto tail_number_) {
                constexpr bool has_loop = decltype(has_main_k_block_loop_)::value;
                constexpr TailNumber tn = tail_number_;

                const auto kernel =
                    kernel_batched_gemm_reduce_wmma_cshuffle_v3<GridwiseGemm,
                                                                typename DeviceOp::ReduceTrait,
                                                                ComputePtrOffsetOfStridedBatch,
                                                                has_loop,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                tn>;

                Run(kernel);
            };

            if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
            {
                if(has_main_k_block_loop && tail_num == TailNumber::Full)
                {
                    CreateAndRunKernel(std::integral_constant<bool, true>{},
                                       std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else if(!has_main_k_block_loop && tail_num == TailNumber::Full)
                {
                    CreateAndRunKernel(std::integral_constant<bool, false>{},
                                       std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else
                {
                    printf("Invalid has_main_k_block_loop and tail_num combination for V1!\n");
                    return 0.0f;
                }
            }
            else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
            {
                if(has_main_k_block_loop && tail_num == TailNumber::Full)
                {
                    CreateAndRunKernel(std::integral_constant<bool, true>{},
                                       std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else if(!has_main_k_block_loop && tail_num == TailNumber::Even)
                {
                    CreateAndRunKernel(std::integral_constant<bool, false>{},
                                       std::integral_constant<TailNumber, TailNumber::Even>{});
                }
                else if(!has_main_k_block_loop && tail_num == TailNumber::Odd)
                {
                    CreateAndRunKernel(std::integral_constant<bool, false>{},
                                       std::integral_constant<TailNumber, TailNumber::Odd>{});
                }
                else
                {
                    printf("Invalid has_main_k_block_loop and tail_num combination for V3!\n");
                    return 0.0f;
                }
            }
            else
            {
                printf("Invalid pipeline version!\n");
                return 0.0f;
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

    static constexpr bool IsValidCompilationParameter() { return true; }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Device implementation supports only gfx11 and gfx12! " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "FP8 and BF8 not supported on gfx11! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if((arg.KRaw_ % AK1 != 0 || arg.KRaw_ % BK1 != 0) &&
           !(GemmSpec == GemmSpecialization::MKPadding ||
             GemmSpec == GemmSpecialization::NKPadding ||
             GemmSpec == GemmSpecialization::MNKPadding ||
             GemmSpec == GemmSpecialization::KPadding))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Without padding, K must be divisible by AK1 and BK1! " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if(ck::is_gfx12_supported() &&
           !GridwiseGemm::CheckValidityAWaveTransfer(arg.MRaw_, arg.KRaw_))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Wave Transfer not applicable for matrix A" << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if(ck::is_gfx12_supported() &&
           !GridwiseGemm::CheckValidityBWaveTransfer(arg.NRaw_, arg.KRaw_))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Wave Transfer not applicable for matrix B" << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        typename GridwiseGemm::Argument gemm_arg{std::array<const void*, 1>{arg.p_a_grid_},
                                                 std::array<const void*, 1>{arg.p_b_grid_},
                                                 std::array<const void*, 0>{},
                                                 static_cast<EDataType*>(arg.p_e_grid_),
                                                 arg.MRaw_,
                                                 arg.NRaw_,
                                                 arg.KRaw_,
                                                 std::array<index_t, 1>{arg.StrideA_}, // StrideAs
                                                 std::array<index_t, 1>{arg.StrideB_}, // StrideBs
                                                 std::array<index_t, 0>{},             // StrideDs
                                                 arg.StrideC_,                         // StrideC
                                                 1,                                    // kbatch
                                                 arg.a_element_op_,
                                                 arg.b_element_op_,
                                                 arg.c_element_op_};

        return GridwiseGemm::CheckValidity(gemm_arg, true);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             const void* p_bias,
                             std::array<const void*, 0> p_ds,
                             void* p_e,
                             std::array<void*, NumReduce> p_reduces,
                             ck::index_t M,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t StrideA,
                             ck::index_t StrideB,
                             ck::index_t StrideC,
                             std::array<ck::index_t, 0> StrideDs,
                             std::array<void*, 3> gemm_element_ops,
                             std::array<void*, 0> d_element_ops,
                             std::array<void*, NumReduce> reduce_in_element_op,
                             std::array<void*, NumReduce> reduce_out_element_op,
                             ck::index_t Batch)
    {
        (void)p_bias;
        (void)p_ds;
        (void)StrideDs;
        (void)d_element_ops;

        ReducePtrsGlobal reduce_tuple = generate_tuple(
            [&](auto I) {
                auto tmp = ReducePtrsGlobal{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return static_cast<T*>(p_reduces[I.value]);
            },
            Number<NumReduce>{});

        ReduceInElementwiseOperations reduce_in_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceInElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_in_element_op[I.value]));
            },
            Number<NumReduce>{});

        ReduceAccElementwiseOperations reduce_out_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceAccElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_out_element_op[I.value]));
            },
            Number<NumReduce>{});

        AElementwiseOperation a_element_op =
            *(static_cast<AElementwiseOperation*>(gemm_element_ops[0]));
        BElementwiseOperation b_element_op =
            *(static_cast<BElementwiseOperation*>(gemm_element_ops[1]));
        CElementwiseOperation c_element_op =
            *(static_cast<CElementwiseOperation*>(gemm_element_ops[2]));

        std::array<long_index_t, NumReduce> batch_stride_reduce{};
        static_for<0, NumReduce, 1>{}(
            [&](auto I) { batch_stride_reduce[I.value] = static_cast<long_index_t>(M); });

        return Argument{static_cast<const ADataType*>(p_a),
                        static_cast<const BDataType*>(p_b),
                        static_cast<EDataType*>(p_e),
                        reduce_tuple,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        Batch,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        reduce_in_element_ops,
                        reduce_out_element_ops,
                        batch_stride_reduce};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const void* p_bias,
                        std::array<const void*, 0> p_ds,
                        void* p_e,
                        std::array<void*, NumReduce> p_reduces,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        std::array<ck::index_t, 0> StrideDs,
                        std::array<void*, 3> gemm_element_ops,
                        std::array<void*, 0> d_element_ops,
                        std::array<void*, NumReduce> reduce_in_element_op,
                        std::array<void*, NumReduce> reduce_out_element_op,
                        ck::index_t Batch = 1) override
    {
        (void)p_bias;
        (void)p_ds;
        (void)StrideDs;
        (void)d_element_ops;

        ReducePtrsGlobal reduce_tuple = generate_tuple(
            [&](auto I) {
                auto tmp = ReducePtrsGlobal{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return static_cast<T*>(p_reduces[I.value]);
            },
            Number<NumReduce>{});

        ReduceInElementwiseOperations reduce_in_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceInElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_in_element_op[I.value]));
            },
            Number<NumReduce>{});

        ReduceAccElementwiseOperations reduce_out_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceAccElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_out_element_op[I.value]));
            },
            Number<NumReduce>{});

        AElementwiseOperation a_element_op =
            *(static_cast<AElementwiseOperation*>(gemm_element_ops[0]));
        BElementwiseOperation b_element_op =
            *(static_cast<BElementwiseOperation*>(gemm_element_ops[1]));
        CElementwiseOperation c_element_op =
            *(static_cast<CElementwiseOperation*>(gemm_element_ops[2]));

        std::array<long_index_t, NumReduce> batch_stride_reduce{};
        static_for<0, NumReduce, 1>{}(
            [&](auto I) { batch_stride_reduce[I.value] = static_cast<long_index_t>(M); });

        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<EDataType*>(p_e),
                                          reduce_tuple,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          Batch,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          reduce_in_element_ops,
                                          reduce_out_element_ops,
                                          batch_stride_reduce);
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

        str << "DeviceBatchedGemmReduce_Wmma_CShuffleV3" << "<" << BlockSize << ", " << MPerBlock
            << ", " << NPerBlock << ", " << KPerBlock << ", " << AK1 << ", " << BK1 << ", "
            << MPerWmma << ", " << NPerWmma << ", " << MRepeat << ", " << NRepeat << ", "
            << ABlockTransferSrcScalarPerVector << ", " << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMRepeatPerShuffle << ", " << CShuffleNRepeatPerShuffle << ">";

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#pragma clang diagnostic pop
