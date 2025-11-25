// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "mx_moe_flatmm.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_moe_gemm.hpp"
#include "ck_tile/ops/flatmm/kernel/mx_moe_flatmm_kernel.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// ==================== Kernel Implementation ====================
template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck_tile::MoeFlatmmKind moe_kind = ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only,
          typename CDEElementWise         = ck_tile::element_wise::PassThrough,
          typename MoeFlatmmHostArgs>
float mx_moe_flatmm(const MoeFlatmmHostArgs& args, const ck_tile::stream_config& s)
{
    using CodegenFlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                               FlatmmConfig::kPadN,
                                                               FlatmmConfig::kPadK,
                                                               FlatmmConfig::DoubleSmemBuffer,
                                                               ALayout,
                                                               BLayout,
                                                               ELayout,
                                                               FlatmmConfig::TransposeC,
                                                               FlatmmConfig::UseStructuredSparsity,
                                                               false, // UsePersistentKernel_
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>; // Preshuffle_

    // ⭐ FP4×FP4 always uses MX pipeline
    constexpr bool MXFP4_Pipeline = true;
    static_assert(std::is_same_v<ADataType, ck_tile::pk_fp4_t> && 
                  std::is_same_v<BDataType, ck_tile::pk_fp4_t>,
                  "mx_moe_flatmm requires FP4×FP4");

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        static_assert(
            FlatmmConfig::N_Tile % (FlatmmConfig::N_Warp * FlatmmConfig::N_Warp_Tile * 2) == 0,
            "requires NRepeat is multiple of 2 for FFN_gemm1_gate_up");
    }

    using ComputeDataType = ck_tile::pk_fp4_t;  // ⭐ FP4→FP16 dequantize

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ComputeDataType,
                                                             ComputeDataType,
                                                             AccDataType,
                                                             CodegenFlatmmShape,
                                                             ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                                                                      FlatmmConfig::kPadN,
                                                                                      FlatmmConfig::kPadK,
                                                                                      ALayout,
                                                                                      BLayout,
                                                                                      ELayout,
                                                                                      FlatmmConfig::NumWaveGroups>>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = FlatmmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        // ⭐ 使用 MXF4 Pipeline (FP4×FP4)
        using CodegenPipelineProblem =
            ck_tile::MXFlatmmPipelineProblem<ADataType,
                                               BDataType,
                                               AccDataType,
                                               CodegenFlatmmShape,
                                               CodegenGemmTraits,
                                               scheduler,
                                               has_hot_loop_v,
                                               tail_number_v>;

        constexpr int BlockedXDLN_PerWarp = 2;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDatatype,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation,
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             FlatmmConfig::TiledMMAPermuteN,
                                             BlockedXDLN_PerWarp>>;

        // ⭐ 使用 MXF4MoeFlatmmPipeline (FP4×FP4 专用)
        using CodegenFlatmmPipeline = 
            ck_tile::MXF4FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;
        
        // ⭐ FP4×FP4 使用 MoeSilu (不是 Swiglu,因为没有 bias)
        using FusedAct = ck_tile::moe::MoeSilu;

        using Kernel = ck_tile::MXMoeFlatmmKernel<TilePartitioner,
                                                CodegenFlatmmPipeline,
                                                GemmEpilogue,
                                                moe_kind,
                                                FusedAct>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:" << CodegenFlatmmShape::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ave_time = ck_tile::launch_kernel(
            s,
            ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
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

// ==================== Weight Shuffle ====================
template <class FlatmmConfig, ck_tile::MoeFlatmmKind moe_kind, class IterSrc, class IterDst>
void shuffle_mx_moe_weight(const IterSrc src, IterDst dst, int experts_cnt, int N, int K)
{
    int KPack = 16;
    int NLane = FlatmmConfig::N_Warp_Tile;
    int KLane = 64 / NLane;
    int K_pk  = K / 2;  // FP4 packed
    int K0    = K_pk / (KLane * KPack);
    int tempk;

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        int up_stride = N / 2 / NLane;

        for(long eid = 0; eid < experts_cnt; ++eid)
        {
            for(int n = 0; n < N; ++n)
            {
                for(int k = 0; k < K_pk; ++k)
                {
                    int n0 = n / NLane;
                    int n1 = n % NLane;

                    int n0_interleave = n >= N / 2 ? (n0 - up_stride) * 2 + 1 : n0 * 2;

                    int k0 = k / (KLane * KPack);
                    tempk  = k % (KLane * KPack);
                    int k1 = tempk / KPack;
                    int k2 = tempk % KPack;

                    long outputIndex = eid * N * K_pk + n0_interleave * KPack * NLane * KLane * K0 +
                                       k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                                       n1 * KPack + k2;

                    dst[outputIndex] = src[eid * N * K_pk + n * K_pk + k];
                }
            }
        }
    }
    else
    {
        for(long eid = 0; eid < experts_cnt; ++eid)
        {
            for(int n = 0; n < N; ++n)
            {
                for(int k = 0; k < K_pk; ++k)
                {
                    int n0 = n / NLane;
                    int n1 = n % NLane;

                    int k0 = k / (KLane * KPack);
                    tempk  = k % (KLane * KPack);
                    int k1 = tempk / KPack;
                    int k2 = tempk % KPack;

                    long outputIndex = eid * N * K_pk + n0 * KPack * NLane * KLane * K0 +
                                       k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                                       n1 * KPack + k2;

                    dst[outputIndex] = src[eid * N * K_pk + n * K_pk + k];
                }
            }
        }
    }
}

// ==================== Scale Shuffle ====================
template <typename FlatmmConfig, ck_tile::MoeFlatmmKind moe_kind, typename T>
auto shuffle_mx_moe_scale(const ck_tile::HostTensor<T>& scale, int experts_cnt)
{
    assert(scale.get_lengths().size() == 2);
    int n_ = scale.get_lengths()[1];
    int k_ = scale.get_lengths()[0];

    int k_per_expert = k_ / experts_cnt;

    constexpr int K_Pack       = 2;
    constexpr int N_Pack       = 2;
    constexpr int GranularityK = 32;
    constexpr int K_Lane       = 64 / FlatmmConfig::N_Warp_Tile;

    static_assert(FlatmmConfig::N_Warp_Tile == 16, "only support XDL_N == 16");
    static_assert(FlatmmConfig::N_Repeat % N_Pack == 0);
    static_assert(FlatmmConfig::K_Tile % (K_Pack * K_Lane * GranularityK) == 0);

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        ck_tile::HostTensor<T> shfl_scale({
            experts_cnt,
            k_per_expert / K_Pack / K_Lane,
            K_Pack,
            K_Lane,
            N_Pack,
            n_ / FlatmmConfig::N_Warp_Tile / N_Pack,
            FlatmmConfig::N_Warp_Tile,
        });
        std::copy(scale.begin(), scale.end(), shfl_scale.begin());
        return ck_tile::reference_permute(shfl_scale, {0, 5, 1, 3, 6, 2, 4});
    }
    else
    {
        ck_tile::HostTensor<T> shfl_scale({
            experts_cnt,
            k_per_expert / K_Pack / K_Lane,
            K_Pack,
            K_Lane,
            n_ / FlatmmConfig::N_Warp_Tile / N_Pack,
            N_Pack,
            FlatmmConfig::N_Warp_Tile,
        });
        std::copy(scale.begin(), scale.end(), shfl_scale.begin());
        return ck_tile::reference_permute(shfl_scale, {0, 4, 1, 3, 6, 2, 5});
    }
}

// ==================== Include Implementation ====================
#include "run_mx_moe_flatmm.inc"

// ==================== Wrapper Function ====================
template <typename FlatmmConfig>
int run_mx_moe_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout = arg_parser.get_str("a_layout");
    const std::string b_layout = arg_parser.get_str("b_layout");
    const std::string mx_prec  = arg_parser.get_str("mx_prec");

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        const std::string gemm_kind = arg_parser.get_str("gemm_kind");
        if(gemm_kind == "gemm1_gate_up")
        {
            if(mx_prec == "fp4xfp4")
            {
                return run_mx_moe_flatmm_with_layouts<
                    ck_tile::pk_fp4_t,
                    ck_tile::pk_fp4_t,
                    ck_tile::fp16_t,
                    FlatmmConfig,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Only support fp4xfp4 for gemm1_gate_up!");
            }
        }
        else if(gemm_kind == "gemm2")
        {
            if(mx_prec == "fp4xfp4")
            {
                return run_mx_moe_flatmm_with_layouts<ck_tile::pk_fp4_t,
                                                      ck_tile::pk_fp4_t,
                                                      ck_tile::fp16_t,
                                                      FlatmmConfig,
                                                      ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Only support fp4xfp4 for gemm2!");
            }
        }
        else
        {
            throw std::runtime_error("Unrecognized gemm_kind parameter, only accept value "
                                     "[gemm1_gate_up | gemm2]");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
    return -1;
}

// ==================== Main Entry ====================
int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return EXIT_FAILURE;

    try
    {
        int warp_tile = arg_parser.get_int("warp_tile");
        if(warp_tile == 0)
        {
            return !run_mx_moe_flatmm_example<MXfp4_MOE_FlatmmConfig16>(argc, argv);
        }
        else
        {
            throw std::runtime_error("Only warp_tile=0 (16x16) is supported now!");
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}