// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file kernel_instantiate.hpp
 * @brief Pure C++ kernel instantiation - NO Python codegen needed!
 *
 * This header provides compile-time kernel instantiation using C++ templates.
 * The Python codegen is essentially doing template instantiation at "codegen time"
 * - this does it at compile time instead.
 *
 * Benefits of pure C++ approach:
 * - Single language, no Python dependency
 * - Better IDE support and type checking
 * - Parallel instantiation handled by compiler (-j N)
 * - Simpler build system
 *
 * Usage:
 *   // Define a kernel configuration at compile time
 *   using MyKernel = GemmKernelInstantiation<
 *       fp16_t, fp16_t, fp16_t, float,           // A, B, C, Acc types
 *       RowMajor, ColMajor, RowMajor,            // Layouts
 *       128, 128, 32,                            // Tile M, N, K
 *       2, 2, 1,                                 // Wave M, N, K
 *       32, 32, 16,                              // Warp M, N, K
 *       Pipeline::CompV4,                        // Pipeline
 *       Scheduler::Intrawave,                    // Scheduler
 *       true, true, true                         // Padding M, N, K
 *   >;
 *
 *   // Launch
 *   float time = MyKernel::launch(args, stream_config);
 */

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// Pipeline and Scheduler enums for template parameters
// =============================================================================

enum class PipelineType
{
    Mem,
    CompV1,
    CompV2,
    CompV3,
    CompV4,
    CompV5,
    PreShuffleV1,
    PreShuffleV2
};

enum class SchedulerType
{
    Intrawave,
    Interwave
};

// =============================================================================
// Layout type traits
// =============================================================================

template <bool IsRowMajor>
struct LayoutTrait
{
    using type = std::
        conditional_t<IsRowMajor, tensor_layout::gemm::RowMajor, tensor_layout::gemm::ColumnMajor>;
};

// =============================================================================
// Primary template for GEMM kernel instantiation
// =============================================================================

/**
 * @brief Compile-time GEMM kernel instantiation
 *
 * This template instantiates a complete GEMM kernel at compile time.
 * No Python codegen needed - the compiler does all the work.
 *
 * @tparam AType      Data type for matrix A
 * @tparam BType      Data type for matrix B
 * @tparam CType      Data type for matrix C
 * @tparam AccType    Accumulator type
 * @tparam ARowMajor  True if A is row-major
 * @tparam BRowMajor  True if B is row-major (false for RCR layout)
 * @tparam CRowMajor  True if C is row-major
 * @tparam TileM_     Tile size M
 * @tparam TileN_     Tile size N
 * @tparam TileK_     Tile size K
 * @tparam WaveM_     Warps per block M
 * @tparam WaveN_     Warps per block N
 * @tparam WaveK_     Warps per block K
 * @tparam WarpM_     Warp tile M
 * @tparam WarpN_     Warp tile N
 * @tparam WarpK_     Warp tile K
 * @tparam Pipe       Pipeline type
 * @tparam Sched      Scheduler type
 * @tparam PadM_      Enable M padding
 * @tparam PadN_      Enable N padding
 * @tparam PadK_      Enable K padding
 */
template <typename AType,
          typename BType,
          typename CType,
          typename AccType,
          bool ARowMajor,
          bool BRowMajor,
          bool CRowMajor,
          index_t TileM_,
          index_t TileN_,
          index_t TileK_,
          index_t WaveM_,
          index_t WaveN_,
          index_t WaveK_,
          index_t WarpM_,
          index_t WarpN_,
          index_t WarpK_,
          PipelineType Pipe,
          SchedulerType Sched,
          bool PadM_,
          bool PadN_,
          bool PadK_,
          index_t BlockSize_ = 256>
struct GemmKernelInstantiation
{
    // Export types for external use
    using ADataType   = AType;
    using BDataType   = BType;
    using CDataType   = CType;
    using AccDataType = AccType;

    // Layouts
    using ALayout = typename LayoutTrait<ARowMajor>::type;
    using BLayout = typename LayoutTrait<BRowMajor>::type;
    using CLayout = typename LayoutTrait<CRowMajor>::type;

    // Configuration constants
    static constexpr index_t BlockSize      = BlockSize_;
    static constexpr index_t TileM          = TileM_;
    static constexpr index_t TileN          = TileN_;
    static constexpr index_t TileK          = TileK_;
    static constexpr index_t WarpPerBlock_M = WaveM_;
    static constexpr index_t WarpPerBlock_N = WaveN_;
    static constexpr index_t WarpPerBlock_K = WaveK_;
    static constexpr index_t WarpTileM      = WarpM_;
    static constexpr index_t WarpTileN      = WarpN_;
    static constexpr index_t WarpTileK      = WarpK_;

    // Traits
    static constexpr bool kPadM                 = PadM_;
    static constexpr bool kPadN                 = PadN_;
    static constexpr bool kPadK                 = PadK_;
    static constexpr bool TransposeC            = false;
    static constexpr bool UsePersistentKernel   = false;
    static constexpr bool DoubleSmemBuffer      = true;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool Preshuffle            = false;
    static constexpr index_t NumWaveGroups      = 1;

    // CK Tile internal types
    using TileShape = TileGemmShape<sequence<TileM, TileN, TileK>,
                                    sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
                                    sequence<WarpTileM, WarpTileN, WarpTileK>,
                                    false,
                                    false>;

    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    using Traits = TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, NumWaveGroups>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, TileShape, Traits>;
    using BaseGemmPipeline = BaseGemmPipelineAgBgCrCompV4<GemmPipelineProblem>;

    /**
     * @brief Launch the kernel
     *
     * Same interface as Python-generated kernels.
     */
    static float launch(const GemmHostArgs& args, const stream_config& stream)
    {
        const index_t k_grain     = args.k_batch * TileK;
        const index_t K_split     = (args.K + k_grain - 1) / k_grain * TileK;
        const index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop   = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        float ave_time{0};

        const auto Run =
            [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
                constexpr bool has_hot_loop_v = has_hot_loop_.value;
                constexpr auto tail_number_v  = tail_number_.value;
                constexpr auto scheduler      = SchedulerToGemmScheduler<Sched>::value;
                [[maybe_unused]] constexpr auto memory_operation = memory_operation_.value;

                using UniversalGemmProblem =
                    UniversalGemmPipelineProblem<ADataType,
                                                 BDataType,
                                                 AccDataType,
                                                 TileShape,
                                                 TileGemmUniversalTraits<kPadM,
                                                                         kPadN,
                                                                         kPadK,
                                                                         DoubleSmemBuffer,
                                                                         ALayout,
                                                                         BLayout,
                                                                         CLayout,
                                                                         TransposeC,
                                                                         UseStructuredSparsity,
                                                                         UsePersistentKernel,
                                                                         NumWaveGroups,
                                                                         Preshuffle>,
                                                 scheduler,
                                                 has_hot_loop_v,
                                                 tail_number_v>;

                using GemmPipeline    = GemmPipelineAgBgCrCompV4<UniversalGemmProblem>;
                using EpilogueProblem = CShuffleEpilogueProblem<CDataType, CDataType, CLayout>;
                using Epilogue        = CShuffleEpilogue<EpilogueProblem>;
                using Kernel          = GemmKernel<TilePartitioner, GemmPipeline, Epilogue>;

                const dim3 grids              = Kernel::GridSize(args.M, args.N, 1);
                const dim3 blocks             = Kernel::BlockSize();
                constexpr index_t kBlockPerCu = 1;

                ave_time = launch_kernel(
                    stream,
                    make_kernel<blocks.x, kBlockPerCu>(Kernel{},
                                                       grids,
                                                       blocks,
                                                       static_cast<const ADataType*>(args.a_ptr),
                                                       static_cast<const BDataType*>(args.b_ptr),
                                                       static_cast<CDataType*>(args.e_ptr),
                                                       args.M,
                                                       args.N,
                                                       args.K_split,
                                                       args.stride_A,
                                                       args.stride_B,
                                                       args.stride_E));
            };

        // Dispatch based on runtime loop conditions
        if(has_hot_loop)
        {
            if(tail_num == TailNumber::Odd)
            {
                Run(std::true_type{},
                    std::integral_constant<TailNumber, TailNumber::Odd>{},
                    std::integral_constant<MemoryOperationEnum, MemoryOperationEnum::Set>{});
            }
            else
            {
                Run(std::true_type{},
                    std::integral_constant<TailNumber, TailNumber::Even>{},
                    std::integral_constant<MemoryOperationEnum, MemoryOperationEnum::Set>{});
            }
        }
        else
        {
            Run(std::false_type{},
                std::integral_constant<TailNumber, TailNumber::Even>{},
                std::integral_constant<MemoryOperationEnum, MemoryOperationEnum::Set>{});
        }

        return ave_time;
    }

    /**
     * @brief Check if this kernel supports the given problem size
     */
    static constexpr bool supports(index_t M, index_t N, index_t K)
    {
        if constexpr(kPadM && kPadN && kPadK)
        {
            return true; // Padding enabled - supports any size
        }
        return (kPadM || M % TileM == 0) && (kPadN || N % TileN == 0) && (kPadK || K % TileK == 0);
    }
};

// =============================================================================
// Scheduler type mapping
// =============================================================================

template <SchedulerType S>
struct SchedulerToGemmScheduler;

template <>
struct SchedulerToGemmScheduler<SchedulerType::Intrawave>
{
    static constexpr auto value = GemmPipelineScheduler::Intrawave;
};

template <>
struct SchedulerToGemmScheduler<SchedulerType::Interwave>
{
    static constexpr auto value = GemmPipelineScheduler::Interwave;
};

// =============================================================================
// Convenience aliases for common configurations
// =============================================================================

// FP16 RCR 128x128x32 (most common)
using Fp16Rcr128x128x32 = GemmKernelInstantiation<fp16_t,
                                                  fp16_t,
                                                  fp16_t,
                                                  float, // Types
                                                  true,
                                                  false,
                                                  true, // RCR layout
                                                  128,
                                                  128,
                                                  32, // Tile
                                                  2,
                                                  2,
                                                  1, // Wave
                                                  32,
                                                  32,
                                                  16, // Warp
                                                  PipelineType::CompV4,
                                                  SchedulerType::Intrawave,
                                                  true,
                                                  true,
                                                  true // Padding
                                                  >;

// FP16 RCR 256x256x64 (compute-bound)
using Fp16Rcr256x256x64 = GemmKernelInstantiation<fp16_t,
                                                  fp16_t,
                                                  fp16_t,
                                                  float,
                                                  true,
                                                  false,
                                                  true,
                                                  256,
                                                  256,
                                                  64,
                                                  4,
                                                  4,
                                                  1,
                                                  32,
                                                  32,
                                                  16,
                                                  PipelineType::CompV4,
                                                  SchedulerType::Intrawave,
                                                  true,
                                                  true,
                                                  true>;

// FP16 RCR 64x64x32 (latency-sensitive)
using Fp16Rcr64x64x32 = GemmKernelInstantiation<fp16_t,
                                                fp16_t,
                                                fp16_t,
                                                float,
                                                true,
                                                false,
                                                true,
                                                64,
                                                64,
                                                32,
                                                2,
                                                2,
                                                1,
                                                16,
                                                16,
                                                16,
                                                PipelineType::CompV4,
                                                SchedulerType::Intrawave,
                                                true,
                                                true,
                                                true>;

// BF16 RCR 128x128x32
using Bf16Rcr128x128x32 = GemmKernelInstantiation<bf16_t,
                                                  bf16_t,
                                                  bf16_t,
                                                  float,
                                                  true,
                                                  false,
                                                  true,
                                                  128,
                                                  128,
                                                  32,
                                                  2,
                                                  2,
                                                  1,
                                                  32,
                                                  32,
                                                  16,
                                                  PipelineType::CompV4,
                                                  SchedulerType::Intrawave,
                                                  true,
                                                  true,
                                                  true>;

// =============================================================================
// Compile-time kernel registration (for multiple kernels)
// =============================================================================

/**
 * @brief Register multiple kernels at compile time
 *
 * Usage:
 *   using KernelSet = KernelRegistry<
 *       Fp16Rcr128x128x32,
 *       Fp16Rcr256x256x64,
 *       Fp16Rcr64x64x32
 *   >;
 *
 *   // At runtime, select based on problem size
 *   if (M >= 2048) {
 *       time = KernelSet::get<1>().launch(args, stream);  // 256x256x64
 *   } else {
 *       time = KernelSet::get<0>().launch(args, stream);  // 128x128x32
 *   }
 */
template <typename... Kernels>
struct KernelRegistry
{
    static constexpr size_t count = sizeof...(Kernels);

    template <size_t I>
    using get = std::tuple_element_t<I, std::tuple<Kernels...>>;

    // Find first kernel that supports the problem
    template <size_t I = 0>
    static constexpr size_t find_supporting(index_t M, index_t N, index_t K)
    {
        if constexpr(I >= count)
        {
            return count; // No kernel found
        }
        else
        {
            if(get<I>::supports(M, N, K))
            {
                return I;
            }
            return find_supporting<I + 1>(M, N, K);
        }
    }
};

} // namespace dispatcher
} // namespace ck_tile
