#pragma once

#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/epilogue.hpp"

template <typename GroupedGemKernelParam,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename DsDataType,
          typename CDataType,
          typename DsLayout,
          typename ALayout,
          typename BLayout,
          typename CLayout>
float invoke_grouped_gemm_persistent(const ck_tile::stream_config& s,
                                    const ck_tile::index_t num_groups,
                                    void* kargs_ptr,
                                    bool splitk)
{
    constexpr bool TransposeC       = false;
    constexpr bool DoubleSmemBuffer = false;

    constexpr int kBlockPerCu                         = 1;
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape       = ck_tile::TileGemmShape<ck_tile::sequence<GroupedGemKernelParam::M_Tile,
                                                                     GroupedGemKernelParam::N_Tile,
                                                                     GroupedGemKernelParam::K_Tile>,
                                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp,
                                                                     GroupedGemKernelParam::N_Warp,
                                                                     GroupedGemKernelParam::K_Warp>,
                                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp_Tile,
                                                                     GroupedGemKernelParam::N_Warp_Tile,
                                                                     GroupedGemKernelParam::K_Warp_Tile>>;
    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using GemmUniversalTraits =
        ck_tile::PersistentTileGemmUniversalTraits<GroupedGemKernelParam::kPadM,
                                                   GroupedGemKernelParam::kPadN,
                                                   GroupedGemKernelParam::kPadK,
                                                   DoubleSmemBuffer,
                                                   ALayout,
                                                   BLayout,
                                                   CLayout,
                                                   TransposeC>;

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto scheduler        = ck_tile::GemmPipelineScheduler::Intrawave;
        constexpr auto memory_operation = memory_operation_.value;

        // We create the GEMM pipeline without specifying hotloop or tailnumber.
        // These are automatically run inside the kernel based on the given input data.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GroupedGemKernelParam::M_Warp,
                                             GroupedGemKernelParam::N_Warp,
                                             GroupedGemKernelParam::M_Warp_Tile,
                                             GroupedGemKernelParam::N_Warp_Tile,
                                             GroupedGemKernelParam::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation>>;
        using Kernel      = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        return ck_tile::launch_kernel(s,
                               ck_tile::make_kernel<kBlockPerCu>(
                                   Kernel{},
                                   grids,
                                   blocks,
                                   0,
                                   ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                   num_groups));
    };

    if(splitk)
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::atomic_add>{});
    }
    else
    {

        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{});
    }
}
