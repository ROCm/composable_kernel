// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/host.hpp"
#include "quant_grouped_gemm.hpp"

template <typename GemmConfig,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout,
          typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          ck_tile::QuantType QuantMode>
float grouped_gemm_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            void* kargs_ptr)
{
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<GemmConfig::kPadM,
                                                             GemmConfig::kPadN,
                                                             GemmConfig::kPadK,
                                                             false,
                                                             false,
                                                             ALayout,
                                                             BLayout,
                                                             CLayout,
                                                             QuantMode,
                                                             AQLayout,
                                                             BQLayout,
                                                             GemmConfig::TransposeC,
                                                             GemmConfig::DoubleSmemBuffer,
                                                             true>;

    float ave_time{0};

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto scheduler        = GemmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;
        constexpr bool transpose_c      = false;

        using QuantGemmProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               transpose_c,
                                                                               BDataType,
                                                                               scheduler>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<QuantGemmProblem>;
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
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             QuantGemmProblem::TransposeC,
                                             memory_operation>>;
        using Kernel      = ck_tile::QuantGroupedGemmKernel<TilePartitioner,
                                                            GemmPipeline,
                                                            GemmEpilogue,
                                                            GemmUniversalTraits::kQuantType>;
        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        return ave_time = ck_tile::launch_kernel(
                   s,
                   ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                       Kernel{},
                       grids,
                       blocks,
                       0,
                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                       num_groups));
    };

    return ave_time = Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                     ck_tile::memory_operation_enum::set>{});
}

#include "quant_run_grouped_gemm_example.inc"

int main(int argc, char* argv[])
{
    return !run_grouped_gemm_example<GemmConfigComputeV3_2>(argc, argv);
}
