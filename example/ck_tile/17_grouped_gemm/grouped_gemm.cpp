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

template <typename ALayout, typename BLayout, typename CLayout>
float gemm_calc(std::vector<const void*>& a_m_k_dev_buf,
                std::vector<const void*>& b_k_n_dev_buf,
                std::vector<void*>& c_m_n_dev_buf,
                const std::vector<ck_tile::GroupedGemmDesc>& gemm_descs,
                const ck_tile::stream_config& s)
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

    auto arguments = Kernel::MakeKargs(a_m_k_dev_buf, b_k_n_dev_buf, c_m_n_dev_buf, gemm_descs);

    std::size_t workspace_size = Kernel::GetWorkSpaceSize(&arguments);
    std::size_t kargs_size     = Kernel::GetDeviceKernelArgSize(&arguments);

    ck_tile::DeviceMem gemm_workspace, gemm_kargs;

    if(kargs_size > 0)
    {
        gemm_kargs.Realloc(kargs_size);
        Kernel::SetDeviceKernelArgs(&arguments, gemm_kargs.GetDeviceBuffer());
    }
    if(workspace_size > 0 && workspace_size != kargs_size)
    {
        gemm_workspace.Realloc(workspace_size);
        Kernel::SetWorkSpacePointer(&arguments, gemm_workspace.GetDeviceBuffer());
    }

    const dim3 grids      = Kernel::GridSize(arguments);
    constexpr dim3 blocks = Kernel::BlockSize();

    ck_tile::hip_check_error(hipMemcpyWithStream(arguments.p_workspace_,
                                                 arguments.gemm_kernel_args_.data(),
                                                 arguments.gemm_kernel_args_.size() *
                                                     sizeof(typename Kernel::GemmTransKernelArg),
                                                 hipMemcpyHostToDevice,
                                                 s.stream_id_));

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel with args:"
                  << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, arguments));

    return ave_time;
}

#include "run_grouped_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_gemm_example(argc, argv); }
