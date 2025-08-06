#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

struct Default2DEpilogueSelector
{
};
struct CShuffleEpilogueSelector
{
};

template <
    /*Specified by instance builder - start*/
    template <typename>
    typename BaseGemmPipeline_,
    template <typename>
    typename GemmPipeline_,
    bool DoubleSmemBuffer,
    GemmPipelineScheduler Scheduler,
    typename EpilogueSelector,
    int TileM,
    int TileN,
    int TileK,
    int WarpM,
    int WarpN,
    int WarpK,
    int WarpTileM,
    int WarpTileN,
    int WarpTileK,
    bool StructuredSparsity,
    /*Specified by instance builder - end*/
    /*Specified by gemm problem - start*/
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AccDataType,
    bool permuteA,
    bool permuteB,
    bool TransposeC,
    int M,
    int N,
    int K,
    int KBatch,
    bool PadM,
    bool PadN,
    bool PadK
    /*Specified by gemm problem - end*/
    >
struct MGXPrototypeGemmKernel
{
    /* Statically set by instance generator, why? */
    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                             ck_tile::sequence<WarpM, WarpN, WarpK>,
                                             ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
                                             permuteA,
                                             permuteB>;

    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<PadM, PadN, PadK, ALayout, BLayout, CLayout>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<PadM,
                                                                 PadN,
                                                                 PadK,
                                                                 DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 TransposeC,
                                                                 StructuredSparsity>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = BaseGemmPipeline_<GemmPipelineProblem>;

    static constexpr ck_tile::index_t k_grain  = KBatch * TileK;
    static constexpr ck_tile::index_t K_split  = (K + k_grain - 1) / k_grain * TileK;
    static constexpr ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
    static constexpr bool has_hot_loop         = BaseGemmPipeline::BlockHasHotloop(num_loop);
    // TODO If the tail_num is Three, the compv4 pipeline uses it, if it isn't Three, it uses Two.
    // Need to check if this is correct of if it's an error in the instance builder.
    // If it's correct, need to handle that case.
    static constexpr ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    static constexpr auto memory_operation = KBatch == 1
                                                 ? ck_tile::memory_operation_enum::set
                                                 : ck_tile::memory_operation_enum::atomic_add;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                       BDataType,
                                                                       AccDataType,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       Scheduler,
                                                                       has_hot_loop,
                                                                       tail_num>;

    using GemmPipeline = GemmPipeline_<UniversalGemmProblem>;

    template <typename T>
    struct EpilogueWrapper
    {
        using type = void;
    };

    template <>
    struct EpilogueWrapper<Default2DEpilogueSelector>
    {
        using type = ck_tile::DefaultGemm2DEpilogue<
            ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  CDataType,
                                                  CLayout,
                                                  PadM,
                                                  PadN,
                                                  WarpTileM,
                                                  WarpTileN,
                                                  WarpTileK,
                                                  UniversalGemmProblem::TransposeC,
                                                  true,
                                                  memory_operation>>;
    };

    template <>
    struct EpilogueWrapper<CShuffleEpilogueSelector>
    {
        using type = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             GemmPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             WarpM,
                                             WarpN,
                                             WarpTileM,
                                             WarpTileN,
                                             WarpTileK,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation>>;
    };

    using GemmEpilogue = typename EpilogueWrapper<EpilogueSelector>::type;
    using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    using KernelArgs = GemmKernelArgs<0>;

    CK_TILE_DEVICE static constexpr bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if constexpr(has_hot_loop)
        {
            if constexpr(std::is_same_v<GemmPipeline,
                                        ck_tile::GemmPipelineAgBgCrMem<UniversalGemmProblem>>)
            {
                // Handle One and Full cases directly
                // if(tail_num == ck_tile::TailNumber::One)
                // {
                //     RunSplitk(ck_tile::bool_constant<true>{},
                //               ck_tile::integral_constant<ck_tile::TailNumber,
                //                                          ck_tile::TailNumber::One>{});
                // }
                // else if(tail_num == ck_tile::TailNumber::Full)
                // {
                //     RunSplitk(ck_tile::bool_constant<true>{},
                //               ck_tile::integral_constant<ck_tile::TailNumber,
                //                                          ck_tile::TailNumber::Full>{});
                // }

                // auto check_tail = [&](auto... TNs) {
                //     (
                //         [&] {
                //// NOTE: can this condition be false for all enum values between Two and Seven? If
                //// so, need to return false. There are also enum values that are not handled here.
                //// Is it possible that they can occur at all?
                ////             if constexpr(BaseGemmPipeline::PrefetchStages >
                //                          static_cast<int>(decltype(TNs)::value))
                //             {
                //                 if(tail_num == decltype(TNs)::value)
                //                 {
                //                     RunSplitk(ck_tile::bool_constant<true>{},
                //                               ck_tile::integral_constant<ck_tile::TailNumber,
                //                                                          decltype(TNs)::value>{});
                //                 }
                //             }
                //         }(),
                //         ...);
                // };

                // check_tail(
                //     ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{},
                //     ck_tile::integral_constant<ck_tile::TailNumber,
                //     ck_tile::TailNumber::Three>{},
                //     ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{},
                //     ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{},
                //     ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{},
                //     ck_tile::integral_constant<ck_tile::TailNumber,
                //     ck_tile::TailNumber::Seven>{});
                // return true;
            }
            else if constexpr(std::is_same_v<
                                  GemmPipeline,
                                  ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>>)
            {
                // if(tail_num == ck_tile::TailNumber::Full)
                // {
                //     RunSplitk(ck_tile::bool_constant<true>{},
                //               ck_tile::integral_constant<ck_tile::TailNumber,
                //                                          ck_tile::TailNumber::Full>{});
                // }
                // else if(tail_num == ck_tile::TailNumber::Odd)
                // {
                //     RunSplitk(ck_tile::bool_constant<true>{},
                //               ck_tile::integral_constant<ck_tile::TailNumber,
                //                                          ck_tile::TailNumber::Odd>{});
                // }
                // else if(tail_num == ck_tile::TailNumber::Even)
                // {
                //     RunSplitk(ck_tile::bool_constant<true>{},
                //               ck_tile::integral_constant<ck_tile::TailNumber,
                //                                          ck_tile::TailNumber::Even>{});
                // }
                // else
                // {
                //     throw std::runtime_error(
                //         "The tail number is wrong. It should be Full, Odd, or Even.");
                // }
                return false;
            }
            else if constexpr(std::is_same_v<
                                  GemmPipeline,
                                  ck_tile::GemmPipelineAgBgCrCompV4<UniversalGemmProblem>>)
            {
                // if(tail_num == ck_tile::TailNumber::Three)
                // {
                //     RunSplitk(ck_tile::bool_constant<true>{},
                //               ck_tile::integral_constant<ck_tile::TailNumber,
                //                                          ck_tile::TailNumber::Three>{});
                // }
                // else
                // {
                //     RunSplitk(ck_tile::bool_constant<true>{},
                //               ck_tile::integral_constant<ck_tile::TailNumber,
                //                                          ck_tile::TailNumber::Two>{});
                // }
                return false;
            }
            else
            {
                return false;
            }
        }
        else
        {
            if constexpr(tail_num != ck_tile::TailNumber::Full &&
                         tail_num != ck_tile::TailNumber::Odd &&
                         tail_num != ck_tile::TailNumber::Even)
            {
                return false;
            }
        }

        return Kernel::template IsSupportedArgument<false>(kargs);
    }

    CK_TILE_DEVICE static void Run(const KernelArgs& kargs) { Kernel{}(kargs); }

    CK_TILE_DEVICE static constexpr auto GridSize() { return Kernel::GridSize(M, N, KBatch); }

    CK_TILE_DEVICE static constexpr auto BlockSize() { return Kernel::BlockSize(); }
};

} // namespace ck_tile
