class GemmKernelBuilder:

    def _generate_kernel_instance(self, is_header=True):

        # Map pipeline names to base pipeline for hot loop detection
        base_pipeline_map = {
            "mem": "ck_tile::BaseGemmPipelineAgBgCrMem",
            "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
            "compv4": "ck_tile::BaseGemmPipelineAgBgCrCompV4",
        }      

        # Map pipeline names to the correct pipeline implementation
        pipeline_impl_map = {
            "mem": "ck_tile::GemmPipelineAgBgCrMem",
            "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
            "compv4": "ck_tile::GemmPipelineAgBgCrCompV4",
        }          

        # Map scheduler names to the correct enum values
        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
            "default": "ck_tile::GemmPipelineScheduler::Default",
        }

        a_type = "ck_tile::fp16_t"
        b_type = "ck_tile::fp16_t"
        c_type = "ck_tile::fp16_t"
        acc_type = "float"

        # ALayout =  RowMajor, BLayout =  ColumnMajor 
        # DeviceGemm_Wmma_CShuffleV3<Default, RCR> 
        a_layout = "ck_tile::tensor_layout::gemm::RowMajor"
        b_layout = "ck_tile::tensor_layout::gemm::ColumnMajor"
        c_layout = "ck_tile::tensor_layout::gemm::RowMajor"

        # BlkTile: 128x256x64, WaveMap: 4x4, WaveTile: 16x16,         
        tile_config = {
            "tile_m": 128,
            "tile_n": 256,
            "tile_k": 64,
            "warp_m": 4,
            "warp_n": 4,
            "warp_k": 1,
            "warp_tile_m": 16,
            "warp_tile_n": 16,
            "warp_tile_k": 16,
        }

        pad_m = "false"
        pad_n = "false"
        pad_k = "false" 

        persistent = "false"

        # BlkGemmPipelineVersion: v1, -> ?????
        pipeline="mem"

        # BlkGemmPipelineScheduler: Intrawave
        scheduler = "intrawave"

        # DeviceGemm_Wmma_CShuffleV3<Default, RCR> 
        epilogue = "cshuffle"

        # VmemReadVec: 8x8  ->  ?????

        kernel_name = f"gemm"

        # Generate kernel instance code using the correct API
        pragma_line = "#pragma once\n" if is_header else ""
        instance_code = f"""// Generated kernel instance for {kernel_name}
{pragma_line}
#include <cstdint>
#include <utility>
#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

using ADataType = {a_type};
using BDataType = {b_type};
using AccDataType = {acc_type};
using CDataType = {c_type};

using ALayout = {a_layout};
using BLayout = {b_layout};
using CLayout = {c_layout};

// Kernel name for display
constexpr const char* KERNEL_NAME = "{kernel_name}";

// Wrapper for simplified launch interface
struct SelectedKernel {{
    // Tile configuration
    static constexpr ck_tile::index_t BlockSize = 256;
    static constexpr ck_tile::index_t TileM = {tile_config["tile_m"]};
    static constexpr ck_tile::index_t TileN = {tile_config["tile_n"]};
    static constexpr ck_tile::index_t TileK = {tile_config["tile_k"]};
    static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_K = {tile_config["warp_k"]};
    static constexpr ck_tile::index_t WarpTileM = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTileN = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t WarpTileK = {tile_config["warp_tile_k"]};

    // Traits
    static constexpr bool kPadM = {"true" if pad_m == "true" else "false"};
    static constexpr bool kPadN = {"true" if pad_n == "true" else "false"};
    static constexpr bool kPadK = {"true" if pad_k == "true" else "false"};
    static constexpr bool TransposeC = false;
    static constexpr bool UsePersistentKernel = {"true" if persistent == "true" else "false"};
    static constexpr bool DoubleSmemBuffer = {"true" if pipeline == "compv4" else "false"};
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool Preshuffle = false;
    static constexpr ck_tile::index_t NumWaveGroups = 1;

    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
        false, false>;
    
    // Tile partitioner
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    
    // Traits
    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, NumWaveGroups>;
    
    // Pipeline problem
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType,
        BDataType,
        AccDataType,
        TileShape,
        Traits>;
    
    // Base pipeline for hot loop detection
    using BaseGemmPipeline = {base_pipeline_map.get(pipeline, "ck_tile::BaseGemmPipelineAgBgCrMem")}<GemmPipelineProblem>;

    static float launch(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& stream) {{
        const ck_tile::index_t k_grain = args.k_batch * TileK;
        const ck_tile::index_t K_split = (args.K + k_grain - 1) / k_grain * TileK;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        
        float ave_time{{0}};

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;
            constexpr auto scheduler = {scheduler_type_map.get(scheduler, "ck_tile::GemmPipelineScheduler::Intrawave")};
            [[maybe_unused]] constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                TileShape,
                ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                                ALayout, BLayout, CLayout, TransposeC,
                                                UseStructuredSparsity, UsePersistentKernel,
                                                NumWaveGroups, Preshuffle>,
                scheduler,
                has_hot_loop_v,
                tail_number_v>;
            
            using GemmPipeline = {pipeline_impl_map.get(pipeline, "ck_tile::GemmPipelineAgBgCrCompV3")}<UniversalGemmProblem>;
            
            // Epilogue
"""

        # Add epilogue configuration based on type
        if epilogue == "cshuffle":
            instance_code += """            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TilePartitioner::MPerBlock,  // kM_
                TilePartitioner::NPerBlock,  // kN_
                WarpPerBlock_M,              // MWave_
                WarpPerBlock_N,              // NWave_
                WarpTileM,                   // MPerXdl_
                WarpTileN,                   // NPerXdl_
                WarpTileK,                   // KPerXdl_
                TransposeC,                  // isCTransposed_
                memory_operation,            // MemoryOperation_
                NumWaveGroups>;              // kNumWaveGroups_
            
            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
"""
        else:  # default epilogue
            instance_code += """            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TilePartitioner::MPerBlock,  // kM_
                TilePartitioner::NPerBlock,  // kN_
                kPadM,
                kPadN,
                WarpTileM,  // kMPerXdl_
                WarpTileN,  // kNPerXdl_
                WarpTileK,  // kKPerXdl_
                TransposeC>;  // isCTransposed_
            
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;
"""

        instance_code += f"""
            
            // Kernel type
            using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            
            // Make kernel arguments
            auto kargs = GemmKernel::MakeKernelArgs(args);
            
            if (!GemmKernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
            }}
            
            // Get grid and block sizes
            const dim3 grids = {"GemmKernel::MaxOccupancyGridSize(stream)" if persistent == "true" else "GemmKernel::GridSize(args.M, args.N, args.k_batch)"};
            const dim3 blocks = GemmKernel::BlockSize();
            
            if(stream.log_level_ > 0) {{
                std::cout << "Launching kernel with args: " << GemmKernel::GetName() << '\\n'
                          << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                          << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                          << std::endl;
            }}
            
            // Launch kernel
            constexpr int kBlockPerCu = 1;
            ave_time = ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
            
            return ave_time;
        }};

        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {{
            if(args.k_batch == 1) {{
                Run(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                            ck_tile::memory_operation_enum::set>{{}});
            }} else {{
                Run(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                            ck_tile::memory_operation_enum::atomic_add>{{}});
            }}
        }};

        BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
        return ave_time;
    }}
}};
"""

        return kernel_name, instance_code
    
def main():
    # Create builder
    builder = GemmKernelBuilder()

    builder._generate_kernel_instance()


if __name__ == "__main__":
    main()    