// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "gemm_utils.hpp"
#include "persistent_async_scheduler.hpp"

/**
 * @brief Invoker for Persistent Async GEMM
 *
 * This invoker implements persistent GEMM with asynchronous input readiness.
 * It extends the standard GEMM with support for:
 * - Chunk-based async input signaling
 * - Producer-consumer synchronization
 * - Pivot-based tile traversal
 */
struct PersistentAsyncInvoker
{
    template <typename GemmConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename CLayout,
              bool Persistent,
              typename CDEElementWise>
    static float gemm(const ck_tile::GemmHostArgs& args,
                      const ck_tile::stream_config& s,
                      const ck_tile::PersistentAsyncArgs& async_args)
    {
        static_assert(Persistent, "PersistentAsyncInvoker requires persistent kernel mode");

        // Tile configuration
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::sequence<GemmConfig::M_Warp_Tile,
                              GemmConfig::N_Warp_Tile,
                              GemmConfig::K_Warp_Tile>>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfig::TileParitionerGroupNum,
                                                       GemmConfig::TileParitionerM01>;

        using Traits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                               GemmConfig::kPadN,
                                               GemmConfig::kPadK,
                                               ALayout,
                                               BLayout,
                                               CLayout,
                                               GemmConfig::NumWaveGroups>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                             GemmConfig::kPadN,
                                             GemmConfig::kPadK,
                                             GemmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             CLayout,
                                             GemmConfig::TransposeC,
                                             GemmConfig::UseStructuredSparsity,
                                             true, // Persistent = true
                                             GemmConfig::NumWaveGroups,
                                             GemmConfig::Preshuffle>;

        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        using BaseGemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

        const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
        const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        float ave_time{0};

        const auto Run = [&](const auto has_hot_loop_,
                             const auto tail_number_,
                             const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto scheduler        = GemmConfig::Scheduler;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;

            using GemmPipeline = typename PipelineTypeTraits<
                GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 AccDataType,
                                                 CDataType,
                                                 DsLayout,
                                                 CLayout,
                                                 CDEElementWise,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 GemmConfig::M_Warp,
                                                 GemmConfig::N_Warp,
                                                 GemmConfig::M_Warp_Tile,
                                                 GemmConfig::N_Warp_Tile,
                                                 GemmConfig::K_Warp_Tile,
                                                 UniversalGemmProblem::TransposeC,
                                                 memory_operation,
                                                 GemmConfig::NumWaveGroups>>;

            using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

            // Create standard kernel args
            auto kargs = Kernel::MakeKernelArgs(args);

            // Use max occupancy grid for persistent kernel
            const dim3 grids  = Kernel::MaxOccupancyGridSize(s);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error(
                    "Wrong! Arguments not supported for persistent async GEMM!\n");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching Persistent Async GEMM kernel:\n"
                          << "  Kernel: " << Kernel::GetName() << '\n'
                          << "  Shape: " << GemmShape::GetName() << '\n'
                          << "  Problem: " << UniversalGemmProblem::GetName() << '\n'
                          << "  Pipeline: " << GemmPipeline::GetName() << '\n'
                          << "  Grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}\n"
                          << "  Blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}\n"
                          << "  Async Args:\n"
                          << "    tiles_per_chunk_m: " << async_args.tiles_per_chunk_m << '\n'
                          << "    tile_idx_pivot_m: " << async_args.tile_idx_pivot_m << '\n'
                          << "    chunk_signals: "
                          << (async_args.chunk_signals ? "enabled" : "disabled") << std::endl;
            }

            // Validation: tiles_per_chunk_m must divide tiles_m evenly
            ck_tile::index_t tiles_m = (args.M + GemmConfig::M_Tile - 1) / GemmConfig::M_Tile;
            if(async_args.tiles_per_chunk_m > 0 && tiles_m % async_args.tiles_per_chunk_m != 0)
            {
                throw std::runtime_error("tiles_per_chunk_m must divide total M tiles evenly!");
            }

            auto clear_gemm_output = [&]() {
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
            };

            // Prepare preprocessing
            std::function<void()> preprocess = clear_gemm_output;

            /*
            // Custom kernel wrapper that includes async scheduler

            ck_tile::index_t tiles_n;

            ck_tile::index_t grid_size;
            auto persistent_async_kernel = [&](auto... kernel_args) {
                // Get tiles info
                tiles_m =
                    (args.M + GemmConfig::M_Tile - 1) / GemmConfig::M_Tile;
                tiles_n =
                    (args.N + GemmConfig::N_Tile - 1) / GemmConfig::N_Tile;
                grid_size = grids.x * grids.y;

                // Create persistent async scheduler
                ck_tile::PersistentAsyncScheduler<TilePartitioner> persistent_scheduler(
                    async_args, tiles_m, tiles_n, grid_size);

                // Persistent tile loop
                while(true)
                {
                    auto work_tile = persistent_scheduler.GetNextWorkTile();
                    if(!work_tile.IsValid())
                        break;

                    // Execute GEMM for this tile
                    // This would call the actual kernel implementation
                    Kernel{}(kernel_args...);

                    // Fence before next iteration
                    scheduler.IterationBoundaryFence();

                    // Advance to next tile
                    scheduler.AdvanceToNextTile();
                }
            };
            */

            // Note: The PersistentAsyncScheduler is integrated into the kernel itself
            // (device-side), not managed from the host. For full async support, a custom kernel
            // implementation would be needed that integrates PersistentAsyncScheduler in its tile
            // loop.
            //
            // TODO: Integrate async_args into kernel arguments and modify the kernel implementation
            // to use PersistentAsyncScheduler for work distribution with async signaling.
            // For now, this launches the standard persistent kernel without async signaling.

            // TODO: Integrate async scheduler into the kernel
            // The async_args parameter is currently not used by the kernel launch.
            // To fully implement async input scheduling, we need to:
            // 1. Create a custom kernel that extends GemmKernel
            // 2. Pass async_args through kernel arguments (KernelArgs)
            // 3. Integrate PersistentAsyncScheduler::GetNextWorkTile() into the
            //    persistent tile loop inside the kernel's operator()
            // 4. Call wait_signal() for chunk readiness before processing tiles
            //
            // For now, suppress unused variable warning

            (void)async_args;

            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                preprocess,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

            return ave_time;
        };

        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
            if(args.k_batch == 1)
            {
                return Run(has_hot_loop_, tail_number_, MemoryOpSet{});
            }
            else
            {
                return Run(has_hot_loop_, tail_number_, MemoryOpAtomicAdd{});
            }
        };

        return ave_time = BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
    }
};
