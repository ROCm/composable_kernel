// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <functional>
#include <chrono>
#include <thread>
#include "gemm_utils.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/device_memory.hpp"

struct UniversalInvoker
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
              typename ELayout,
              bool Persistent,
              typename CDEElementWise>
    static float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

    {
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::
                sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
            GemmConfig::PermuteA,
            GemmConfig::PermuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfig::TileParitionerGroupNum,
                                                       GemmConfig::TileParitionerM01>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                             GemmConfig::kPadN,
                                             GemmConfig::kPadK,
                                             GemmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             ELayout,
                                             GemmConfig::TransposeC,
                                             GemmConfig::UseStructuredSparsity,
                                             Persistent,
                                             GemmConfig::NumWaveGroups,
                                             GemmConfig::Preshuffle>;

        constexpr auto scheduler = GemmConfig::Scheduler;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             GemmConfig::NumWaveGroups,
                                             false, /*FixedVectorSize_*/
                                             1,     /*VectorSizeC_*/
                                             false, /*TiledMMAPermuteN_*/
                                             1,     /*BlockedXDLN_PerWarp_*/
                                             GemmConfig::DoubleSmemBuffer /*DoubleSmemBuffer*/>>;

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Persistent ? Kernel::MaxOccupancyGridSize(s)
                                       : Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << UniversalGemmProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        // Declare rotating_mem_ptr here so it stays in scope until it is needed
        std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr;
        std::function<void()> preprocess;

        auto clear_gemm_output = [&]() {
            if(args.k_batch > 1)
                hipGetErrorString(hipMemsetAsync(
                    args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
        };

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            rotating_mem_ptr = std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
                kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem_ptr->Print();

            preprocess = [&]() {
                ck_tile::flush_icache();
                rotating_mem_ptr->Next();
                clear_gemm_output();
            };
        }
        else
        {
            preprocess = clear_gemm_output;
        }

        return ck_tile::launch_kernel_time_mask(
            s,
            preprocess,
            ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }

    template <typename GemmConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              typename CDEElementWise>
    static void test_async_input_scheduler(const ck_tile::GemmHostArgs& args,
                                           const ck_tile::stream_config& s)
    {
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::
                sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
            GemmConfig::PermuteA,
            GemmConfig::PermuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfig::TileParitionerGroupNum,
                                                       GemmConfig::TileParitionerM01>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                             GemmConfig::kPadN,
                                             GemmConfig::kPadK,
                                             GemmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             ELayout,
                                             GemmConfig::TransposeC,
                                             GemmConfig::UseStructuredSparsity,
                                             true, // Persistent = true for async test
                                             GemmConfig::NumWaveGroups,
                                             GemmConfig::Preshuffle>;

        constexpr auto scheduler = GemmConfig::Scheduler;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             GemmConfig::NumWaveGroups,
                                             false, /*FixedVectorSize_*/
                                             1,     /*VectorSizeC_*/
                                             false, /*TiledMMAPermuteN_*/
                                             1,     /*BlockedXDLN_PerWarp_*/
                                             GemmConfig::DoubleSmemBuffer>>;

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        const ck_tile::index_t tiles_m =
            ck_tile::integer_divide_ceil(args.M, TilePartitioner::MPerBlock);
        // Balance signal granularity (smaller chunks = finer control) vs overhead (more signals)
        const ck_tile::index_t tiles_per_chunk = 2;
        // Shift chunk assignments to test wraparound behavior
        const ck_tile::index_t tile_idx_pivot = tiles_per_chunk;
        // Account for pivot when allocating signal buffer
        const ck_tile::index_t num_chunks =
            ck_tile::integer_divide_ceil(tiles_m + tile_idx_pivot, tiles_per_chunk);

        std::cout << "Async Input Scheduler Test:" << std::endl;
        std::cout << "  M tiles: " << tiles_m << std::endl;
        std::cout << "  Tiles per chunk: " << tiles_per_chunk << std::endl;
        std::cout << "  Tile index pivot: " << tile_idx_pivot << std::endl;
        std::cout << "  Number of signal chunks: " << num_chunks << std::endl;

        // Signals must start as zero so kernel blocks until producer sets them
        ck_tile::DeviceMem signal_buf(num_chunks * sizeof(uint32_t));
        signal_buf.SetZero();
        uint32_t* d_chunk_signals = static_cast<uint32_t*>(signal_buf.GetDeviceBuffer());

        // Setup async input scheduler
        ck_tile::PersistentAsyncInputScheduler async_scheduler;
        async_scheduler.tiles_per_chunk_m = tiles_per_chunk;
        async_scheduler.chunk_signals     = d_chunk_signals;
        async_scheduler.tile_idx_pivot_m  = tile_idx_pivot;
        async_scheduler.num_chunks        = num_chunks;

        // Create modified host args with async scheduler
        ck_tile::UniversalGemmHostArgs<1, 1, 0> host_args({args.a_ptr},
                                                          {args.b_ptr},
                                                          {},
                                                          args.e_ptr,
                                                          args.k_batch,
                                                          args.M,
                                                          args.N,
                                                          args.K,
                                                          {args.stride_A},
                                                          {args.stride_B},
                                                          {},
                                                          args.stride_E,
                                                          async_scheduler);

        auto kargs = Kernel::UniversalGemmKernel::MakeKernelArgs(host_args);

        const dim3 grids  = Kernel::MaxOccupancyGridSize(s);
        const dim3 blocks = Kernel::BlockSize();

        std::cout << "  Grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << std::endl;
        std::cout << "  Blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;

        // Separate stream prevents deadlock: kernel and signal producer must run concurrently
        hipStream_t signal_stream;
        HIP_CHECK_ERROR(hipStreamCreateWithFlags(&signal_stream, hipStreamNonBlocking));

        const auto start = std::chrono::high_resolution_clock::now();

        ck_tile::launch_kernel(
            s, ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        // Simulate incremental input arrival by delaying signal activation
        const int sleep_us = 100;
        for(ck_tile::index_t i = 0; i < num_chunks; ++i)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            const uint32_t signal_val = 1;
            HIP_CHECK_ERROR(hipMemcpyAsync(d_chunk_signals + i,
                                           &signal_val,
                                           sizeof(uint32_t),
                                           hipMemcpyHostToDevice,
                                           signal_stream));
        }
        HIP_CHECK_ERROR(hipStreamSynchronize(signal_stream));
        HIP_CHECK_ERROR(hipStreamDestroy(signal_stream));

        // Wait for kernel completion
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);

        std::cout << "  Total time: " << duration.count() << " us" << std::endl;
        std::cout << "  Sleep time: " << (num_chunks * sleep_us) << " us" << std::endl;
    }
};
