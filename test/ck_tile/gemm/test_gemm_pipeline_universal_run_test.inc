// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "gtest/gtest.h"

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
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

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

    using Traits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                           GemmConfig::kPadN,
                                           GemmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           GemmConfig::NumWaveGroups>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 Persistent,
                                                                 GemmConfig::NumWaveGroups>;
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
                                             memory_operation,
                                             GemmConfig::NumWaveGroups>>;
        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        dim3 grids;
        if constexpr(Persistent)
        {
            grids = Kernel::MaxOccupancyGridSize(s);
        }
        else
        {
            grids = Kernel::GridSize(args.M, args.N, args.k_batch);
        }
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw ArgumentsNotSupportedException(
                "Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << GemmPipelineProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }
        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;
            static constexpr ck_tile::index_t APackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
            static constexpr ck_tile::index_t BPackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
            auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

            ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                // flush icache
                ck_tile::flush_icache();
                // rotating mem
                rotating_mem.Next();
                // clear c mem
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
            };
            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                run_flush_cache,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            ave_time = ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        return ave_time;
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(args.k_batch == 1)
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::atomic_add>{});
        }
    };

    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
    return ave_time;
}

template <typename GemmConfig,
          typename APrecType,
          typename BPrecType = APrecType,
          typename CPrecType = APrecType>
bool run_gemm_test_prec_type(const int M, const int N, const int K)
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    return run_gemm_test_with_layouts<GemmConfig, APrecType, BPrecType, CPrecType, Row, Col, Row>(
        M, N, K);
}

template <typename Tuple>
class TestCkTileGemmPipelineUniversal : public ::testing::Test
{
    protected:
    using GemmConfig = std::tuple_element_t<0, Tuple>;
    using APrecType  = std::tuple_element_t<1, Tuple>;
    using BPrecType  = std::tuple_element_t<2, Tuple>;
    using CPrecType  = std::tuple_element_t<3, Tuple>;

    void run_gemm_combinations(const int m, const int n, const int k)
    {
        // Skip tests that are known to fail or are not supported
        if constexpr((std::is_same_v<GemmConfig, GemmConfigComputeV3<CPrecType>> ||
                      std::is_same_v<GemmConfig, GemmConfigComputeV3_2<CPrecType>>) &&
                     std::is_same_v<APrecType, F8> && std::is_same_v<BPrecType, BF8>)
        {
            GTEST_SKIP()
                << "Skipping this test due to known failures with F8 x BF8 on the V3 pipeline";
        }
        else if constexpr((std::is_same_v<GemmConfig, GemmConfigComputeV4<CPrecType>>) &&
                          std::is_same_v<BPrecType, I4>)
        {
            GTEST_SKIP()
                << "Skipping this test because BPrecType I4 is not supported on the V4 pipeline";
        }
        else
        {
            bool is_success = true;
            // Call the function with the current configuration
            try
            {
                is_success =
                    run_gemm_test_prec_type<GemmConfig, APrecType, BPrecType, CPrecType>(m, n, k);
            }
            catch(const ArgumentsNotSupportedException& e)
            {
                std::cerr << "Caught ArgumentsNotSupportedException: " << e.what() << '\n';
                // ArgumentsNotSupportedException  is not an error. Do not change is_success
            }
            catch(const std::runtime_error& e)
            {
                std::cerr << "Caught runtime error: " << e.what() << '\n';
                is_success = false;
            }
            EXPECT_TRUE(is_success);
        }
    }
};
