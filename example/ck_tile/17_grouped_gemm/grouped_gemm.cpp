// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"
#include "utils.hpp"

template <typename TType>
constexpr std::size_t GetWorkspaceSize(const std::vector<TType>& gemm_descs)
{
    return ck_tile::GetWorkSpaceSize<TType>(gemm_descs);
}

template <typename ALayout, typename BLayout, typename CLayout>
float grouped_gemm_calc(const std::vector<grouped_gemm_kargs>& gemm_descs,
                        const ck_tile::stream_config& s,
                        void* p_workspace_)
{
    constexpr bool kPadM        = false;
    constexpr bool kPadN        = false;
    constexpr bool kPadK        = false;
    constexpr bool kTilePermute = false;

    constexpr ck_tile::index_t kOutputRank = 2;

    constexpr int kBlockPerCu = 1;

    // This part comes from the Codegen
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 128;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;

    constexpr bool CShuffleEpilogue =
        std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    using GemmEpilogue = std::conditional_t<
        CShuffleEpilogue,
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<AccDataType,
                                                                   CDataType,
                                                                   kPadM,
                                                                   kPadN,
                                                                   kTilePermute,
                                                                   kOutputRank,
                                                                   1,
                                                                   0,
                                                                   TilePartitioner::MPerBlock,
                                                                   TilePartitioner::NPerBlock>>,
        ck_tile::Default2DEpilogue<
            ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadM, kPadN>>>;

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    using CodegenPipelineProblem = ck_tile::
        GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;

    using CodegenGemmPolicy = ck_tile::UniversalGemmPipelineAgBgCrPolicy;
    using CodegenGemmPipeline =
        ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem, CodegenGemmPolicy>;

    using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;

    auto arguments = Kernel::MakeKargs(gemm_descs);

    const dim3 grids      = Kernel::GridSize(gemm_descs);
    constexpr dim3 blocks = Kernel::BlockSize();

    ck_tile::hip_check_error(
        hipMemcpyWithStream(p_workspace_,
                            arguments.data(),
                            arguments.size() * sizeof(ck_tile::GemmTransKernelArg),
                            hipMemcpyHostToDevice,
                            s.stream_id_));

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel with args:"
                  << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    float ave_time =
        ck_tile::launch_kernel(s,
                               ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                                   Kernel{},
                                   grids,
                                   blocks,
                                   0,
                                   ck_tile::cast_pointer_to_constant_address_space(p_workspace_),
                                   gemm_descs.size()));
    return ave_time;
}

#include "run_grouped_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_gemm_example(argc, argv); }
