// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"
#include "mx_gemm.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp"
#include "ck_tile/ops/gemm_mx/kernel/gemm_mx_kernel.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/wp_pipeline_agmem_bgmem_creg_v1.hpp"

template <typename Layout>
using is_row_major_t = ck_tile::bool_constant<
    std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>;

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          bool Splitk>
float mx_gemm_calc(const MXGemmHostArgs<ScaleM, ScaleN>& args, const ck_tile::stream_config& s)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using MXGemmTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                          GemmConfig::kPadN,
                                                          GemmConfig::kPadK,
                                                          GemmConfig::DoubleSmemBuffer,
                                                          ALayout,
                                                          BLayout,
                                                          CLayout,
                                                          GemmConfig::TransposeC,
                                                          GemmConfig::UseStructuredSparsity,
                                                          persistent,
                                                          GemmConfig::NumWaveGroups,
                                                          GemmConfig::Preshuffle>;

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_gemm requires ADataType is a wider type than BDataType");

    using MXPipelineProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    GemmShape,
                                                                    MXGemmTraits,
                                                                    GemmConfig::Scheduler>;

    // Use the MX GEMM Preshuffle pipeline or
    // the new MX comp_async pipeline with MX scaling support
    constexpr bool IsEightWave =
        (GemmConfig::M_Warp * GemmConfig::N_Warp * GemmConfig::K_Warp) == 8;
    using MXGemmPipeline = std::conditional_t<
        GemmConfig::Preshuffle,
        ck_tile::MXGemmPreshufflePipelineAGmemBGmemCRegV1<MXPipelineProblem>,
        std::conditional_t<IsEightWave,
                           ck_tile::MXGemmPipelineAgBgCrCompAsyncEightWaves<MXPipelineProblem>,
                           ck_tile::MXGemmPipelineAgBgCrCompAsync<MXPipelineProblem>>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmEpilogue =
        std::conditional_t<GemmConfig::TiledMMAPermuteN,
                           ck_tile::PermuteNEpilogue<
                               ck_tile::PermuteNEpilogueProblem<ComputeDataType,
                                                                ComputeDataType,
                                                                ck_tile::tuple<>, // DsDataType
                                                                AccDataType,
                                                                CDataType,
                                                                ck_tile::tuple<>, // DsLayout
                                                                CLayout,
                                                                ck_tile::element_wise::PassThrough,
                                                                TilePartitioner::MPerBlock,
                                                                TilePartitioner::NPerBlock,
                                                                GemmConfig::M_Warp,
                                                                GemmConfig::N_Warp,
                                                                GemmConfig::M_Warp_Tile,
                                                                GemmConfig::N_Warp_Tile,
                                                                GemmConfig::K_Warp_Tile,
                                                                MXPipelineProblem::TransposeC,
                                                                false, // FixedVectorSize_ (Default)
                                                                1>>,   // VectorSizeC_ (Default)
                           ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                               ComputeDataType,
                               ComputeDataType,
                               ck_tile::tuple<>, // DsDataType
                               AccDataType,
                               CDataType,
                               ck_tile::tuple<>, // DsLayout
                               CLayout,
                               ck_tile::element_wise::PassThrough,
                               TilePartitioner::MPerBlock,
                               TilePartitioner::NPerBlock,
                               GemmConfig::M_Warp,
                               GemmConfig::N_Warp,
                               GemmConfig::M_Warp_Tile,
                               GemmConfig::N_Warp_Tile,
                               GemmConfig::K_Warp_Tile,
                               MXPipelineProblem::TransposeC,
                               GemmConfig::NumWaveGroups,
                               false, // FixedVectorSize_ (Default)
                               1,     // VectorSizeC_ (Default)
                               ck_tile::MXEpilogueTraits<GemmConfig>::BlockedXDLNPerWarp,
                               false,                      // DoubleSmemBuffer_ (Default)
                               ComputeDataType,            // AComputeDataType
                               ComputeDataType,            // BComputeDataType
                               !GemmConfig::Preshuffle>>>; // TilesPacked_ (because of
                                                           // packed scales)

    using Kernel = ck_tile::MXGemmKernel<TilePartitioner, MXGemmPipeline, GemmEpilogue>;

    auto kargs = Kernel::MakeKernelArgs(std::array<const void*, 1>{args.as_ptr},
                                        std::array<const void*, 1>{args.bs_ptr},
                                        std::array<const void*, 0>{},
                                        args.e_ptr,
                                        args.k_batch,
                                        args.M,
                                        args.N,
                                        args.K,
                                        std::array<ck_tile::index_t, 1>{args.stride_As},
                                        std::array<ck_tile::index_t, 1>{args.stride_Bs},
                                        std::array<ck_tile::index_t, 0>{},
                                        args.stride_E,
                                        args.scale_m,
                                        args.scale_n);

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error(
            "MX GEMM: unsupported shape/configuration (set CK_TILE_LOGGING=1 for details).");
    }

    const auto kernel = ck_tile::make_kernel<Kernel::kBlockPerCu>(
        Kernel{}, Kernel::GridSize(kargs), Kernel::BlockSize(), 0, kargs);

    // For split-K (k_batch > 1) the kernel's epilogue uses atomic_add into C, so C must be
    // zeroed before every kernel launch -- not just once before the first warmup iteration.
    // Mirror universal_gemm's preprocess pattern.
    auto clear_gemm_output = [&]() {
        if(args.k_batch > 1)
        {
            (void)hipMemsetAsync(args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_);
        }
    };

    return ck_tile::launch_kernel_time_mask(s, clear_gemm_output, kernel);
}
