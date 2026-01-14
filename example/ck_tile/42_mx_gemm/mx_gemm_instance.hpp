// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"
#include "mx_gemm.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"
#include "ck_tile/ops/gemm_mx/kernel/gemm_mx_kernel.hpp"

template <typename Layout>
using is_row_major_t = ck_tile::bool_constant<
    std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>;

// Problem definition for MX GEMM with comp_async pipeline
// The comp_async pipeline handles MX scaling with OpSel parameters
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename BlockGemmShape,
          typename Traits,
          ck_tile::GemmPipelineScheduler Scheduler_ = ck_tile::GemmPipelineScheduler::Intrawave>
struct MXGemmPipelineProblem : ck_tile::GemmPipelineProblem<ADataType, BDataType, CDataType, BlockGemmShape, Traits>
{
    static constexpr auto Scheduler = Scheduler_;
};

// Epilogue wrapper that adds MemoryOperation member for MX GEMM kernel compatibility
template <typename BaseEpilogue_, ck_tile::memory_operation_enum MemOp_>
struct MXGemmEpilogueWrapper : BaseEpilogue_
{
    static constexpr ck_tile::memory_operation_enum MemoryOperation = MemOp_;
    using BaseEpilogue_::BaseEpilogue_;
    using BaseEpilogue_::operator();
};

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
float mx_gemm_calc(const MXGemmHostArgs<ScaleM, ScaleN>& args,
                   const ck_tile::stream_config& s)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile,
                          GemmConfig::N_Warp_Tile,
                          GemmConfig::K_Warp_Tile>>;

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
                                                          true>;

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_gemm requires ADataType is a wider type than BDataType");

    constexpr auto scheduler = GemmConfig::Scheduler;
    constexpr auto memory_operation =
        Splitk ? ck_tile::memory_operation_enum::atomic_add : ck_tile::memory_operation_enum::set;

    using MXPipelineProblem = MXGemmPipelineProblem<ADataType,
                                                    BDataType,
                                                    AccDataType,
                                                    GemmShape,
                                                    MXGemmTraits,
                                                    scheduler>;

    // Use the new MX comp_async pipeline with MX scaling support
    using MXGemmPipeline = ck_tile::MXGemmPipelineAgBgCrCompAsync<MXPipelineProblem>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;
    
    using BaseEpilogue =
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<ComputeDataType,
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
                                                                   GemmConfig::NumWaveGroups, // kNumWaveGroups
                                                                   false, // FixedVectorSize
                                                                   1, // VectorSizeC
                                                                   false, // TiledMMAPermuteN
                                                                   1, // BlockedXDLN_PerWarp
                                                                   false>>; // DoubleSmemBuffer
    
    using GemmEpilogue = MXGemmEpilogueWrapper<BaseEpilogue, memory_operation>;

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
    
    const auto kernel = ck_tile::make_kernel<Kernel::kBlockPerCu>(
        Kernel{},
        Kernel::GridSize(kargs),
        Kernel::BlockSize(),
        Kernel::GetSmemSize(),
        kargs);

    return ck_tile::launch_kernel(s, kernel);
}
