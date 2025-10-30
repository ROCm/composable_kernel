// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_smoke_util.hpp"
#include "test_gemm_pipeline_smoke_run_test.inc"

struct GemmConfig_Mfma : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

struct GemmConfig_Wmma : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

#if CK_TILE_USE_WMMA
using GemmConfigs = ::testing::Types<GemmConfig_Wmma>;
#else
using GemmConfigs = ::testing::Types<GemmConfig_Mfma>;
#endif

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
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

{
    if constexpr(Persistent)
        std::cout << "WARNING: Ignoring persistent kernel option for basic gemm." << std::endl;
    // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 1;

    // This part comes from the Codegen
    constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
    constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
    constexpr ck_tile::index_t K_Tile = GemmConfig::K_Tile;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
    constexpr ck_tile::index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    using CodegenPipelineProblem = ck_tile::
        GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;

    using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto memory_operation = memory_operation_.value;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw ArgumentsNotSupportedException(
                "Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << CodegenGemmShape::GetName() << '\n'
                      << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                      << "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    };

    if(args.k_batch == 1)
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::atomic_add>{});
    }
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
class TestCkTileGemmPipelineBasic : public ::testing::Test
{
    protected:
    using GemmConfig = std::tuple_element_t<0, Tuple>;
    using APrecType  = std::tuple_element_t<1, Tuple>;
    using BPrecType  = std::tuple_element_t<2, Tuple>;
    using CPrecType  = std::tuple_element_t<3, Tuple>;

    void run_gemm_combinations(const int m, const int n, const int k)
    {
        // Skip tests that are known to fail
        if constexpr(std::is_same_v<APrecType, F8> && std::is_same_v<BPrecType, BF8>)
        {
            GTEST_SKIP() << "Skipping this test due to known failures with F8 x BF8";
        }
        else if constexpr(std::is_same_v<APrecType, F16> && std::is_same_v<BPrecType, I4>)
        {
            GTEST_SKIP() << "Skipping this test due to known failures with F16 x I4";
        }
        else
        {
            bool is_success = true;
            std::cout << "-m=" << m << " -n=" << n << " -k=" << k << std::endl;

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
