// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <functional>
#include "gemm_utils.hpp"

namespace ck_tile::experimental::builder {

template<ck_tile::index_t kMPerBlock,
         ck_tile::index_t kNPerBlock,
         ck_tile::index_t kKPerBlock,
         ck_tile::index_t GroupNum,
         ck_tile::index_t M01>
struct TilePartitionerResolver
{
    struct GemmBlockTileShape
    {
        static constexpr auto kM = kMPerBlock;
        static constexpr auto kN = kNPerBlock;
        static constexpr auto kK = kKPerBlock;
    };
    using Type = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmBlockTileShape,
        GroupNum,
        M01>;
};

// parameterize on the pipeline type
template<auto PipelineId>
struct HotLoopDescription;

template<>
struct HotLoopDescription<CK_TILE_PIPELINE_COMPUTE_V3>
{
    static constexpr index_t PrefetchStages = 2;

    CK_TILE_HOST static constexpr bool has_hot_loop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr ck_tile::TailNumber get_tail_num(index_t num_loop)
    {
        if(has_hot_loop(num_loop))
        {
            return ck_tile::TailNumber::Full;
        }
        else
        {
            if(num_loop == 1)
            {
                return ck_tile::TailNumber::Odd;
            }
            else
            {
                return ck_tile::TailNumber::Even;
            }
        }
    }
};

template<ck_tile::index_t kMPerBlock,
         ck_tile::index_t kNPerBlock,
         ck_tile::index_t kKPerBlock,
         ck_tile::index_t GroupNum,
         ck_tile::index_t M01>
using TilePartitionerType =
    typename TilePartitionerResolver<kMPerBlock, kNPerBlock, kKPerBlock, GroupNum, M01>::Type;

template <class AlgorithmMetadata, class InputMetadata>
struct UniversalFactory
{
    private:
    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<AlgorithmMetadata::M_Tile::value,
                                                 AlgorithmMetadata::N_Tile::value,
                                                 AlgorithmMetadata::K_Tile::value>,
                               ck_tile::sequence<AlgorithmMetadata::M_Warp::value,
                                                 AlgorithmMetadata::N_Warp::value,
                                                 AlgorithmMetadata::K_Warp::value>,
                               ck_tile::sequence<AlgorithmMetadata::M_Warp_Tile::value,
                                                 AlgorithmMetadata::N_Warp_Tile::value,
                                                 AlgorithmMetadata::K_Warp_Tile::value>,
                               AlgorithmMetadata::PermuteA::value,
                               AlgorithmMetadata::PermuteB::value>;

    using TilePartitioner = TilePartitionerType<
        AlgorithmMetadata::M_Tile::value,
        AlgorithmMetadata::N_Tile::value,
        AlgorithmMetadata::K_Tile::value,
        AlgorithmMetadata::TilePartitionerGroupNum::value,
        AlgorithmMetadata::TilePartitionerM01::value>;

    using Traits = ck_tile::TileGemmTraits<AlgorithmMetadata::kPadM::value,
                                           AlgorithmMetadata::kPadN::value,
                                           AlgorithmMetadata::kPadK::value,
                                           typename InputMetadata::InputALayout,
                                           typename InputMetadata::InputBLayout,
                                           typename InputMetadata::InputELayout,
                                           AlgorithmMetadata::NumWaveGroups::value>;

    using GemmUniversalTraits =
        ck_tile::TileGemmUniversalTraits<AlgorithmMetadata::kPadM::value,
                                         AlgorithmMetadata::kPadN::value,
                                         AlgorithmMetadata::kPadK::value,
                                         AlgorithmMetadata::DoubleSmemBuffer::value,
                                         typename InputMetadata::InputALayout,
                                         typename InputMetadata::InputBLayout,
                                         typename InputMetadata::InputELayout,
                                         AlgorithmMetadata::TransposeC::value,
                                         AlgorithmMetadata::UseStructuredSparsity::value,
                                         AlgorithmMetadata::KPersistent::value,
                                         AlgorithmMetadata::NumWaveGroups::value,
                                         AlgorithmMetadata::Preshuffle::value>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<typename InputMetadata::InputADataType,
                                     typename InputMetadata::InputBDataType,
                                     typename InputMetadata::InputAccDataType,
                                     GemmShape,
                                     Traits>;

    using BaseGemmPipeline = typename PipelineTypeTraits<
        AlgorithmMetadata::Pipeline::value>::template UniversalGemmPipeline<GemmPipelineProblem>;

    using UniversalGemmProblem =
        ck_tile::UniversalGemmPipelineProblem<typename InputMetadata::InputADataType,
                                              typename InputMetadata::InputBDataType,
                                              typename InputMetadata::InputAccDataType,
                                              GemmShape,
                                              GemmUniversalTraits,
                                              AlgorithmMetadata::Scheduler::value,
                                              AlgorithmMetadata::HasHotLoop::value,
                                              AlgorithmMetadata::TailNum::value>;

    using GemmPipeline = typename PipelineTypeTraits<
        AlgorithmMetadata::Pipeline::value>::template GemmPipeline<UniversalGemmProblem>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<typename InputMetadata::InputADataType,
                                         typename InputMetadata::InputBDataType,
                                         typename InputMetadata::InputDsDataType,
                                         typename InputMetadata::InputAccDataType,
                                         typename InputMetadata::InputCDataType,
                                         typename InputMetadata::InputDsLayout,
                                         typename InputMetadata::InputELayout,
                                         typename InputMetadata::InputCDEElementWise,
                                         AlgorithmMetadata::M_Tile::value,
                                         AlgorithmMetadata::N_Tile::value,
                                         AlgorithmMetadata::M_Warp::value,
                                         AlgorithmMetadata::N_Warp::value,
                                         AlgorithmMetadata::M_Warp_Tile::value,
                                         AlgorithmMetadata::N_Warp_Tile::value,
                                         AlgorithmMetadata::K_Warp_Tile::value,
                                         AlgorithmMetadata::TransposeC::value,
                                         AlgorithmMetadata::MemoryOperation::value,
                                         AlgorithmMetadata::NumWaveGroups::value>>;

    public:
    using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

    CK_TILE_HOST static constexpr auto make_kernel(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
    {
        auto kargs = Kernel::MakeKernelArgs(args);

        // NB: do we really need the stream to be launched here?
        const dim3 grids  = AlgorithmMetadata::KPersistent::value
                                ? Kernel::MaxOccupancyGridSize(s)
                                : Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        return ck_tile::make_kernel<AlgorithmMetadata::kBlockPerCu::value>(
            Kernel{}, grids, blocks, 0, kargs);
    }
};
} // namespace ck_tile::experimental::builder

struct UniversalInvoker
{
    template <typename GemmConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              bool Persistent,
              typename CDEElementWise>
    static float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

    {
        using TilePartitioner = ck_tile::experimental::builder::TilePartitionerType<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile, GemmConfig::TileParitionerGroupNum, GemmConfig::TileParitionerM01>;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum2(args.K, args.k_batch);
        const bool has_hot_loop            = ck_tile::experimental::builder::HotLoopDescription<CK_TILE_PIPELINE_COMPUTE_V3>::has_hot_loop(num_loop);
        const ck_tile::TailNumber tail_num = ck_tile::experimental::builder::HotLoopDescription<CK_TILE_PIPELINE_COMPUTE_V3>::get_tail_num(num_loop);

        const auto kernel_launch_visitor = [&args, &s](const auto has_hot_loop_,
                                               const auto tail_number_,
                                               const auto memory_operation_) {
            struct Algo
            {
                // can't do `static constexpr` in local structs
                using M_Tile =
                    ck_tile::integral_constant<decltype(GemmConfig::M_Tile), GemmConfig::M_Tile>;
                using N_Tile =
                    ck_tile::integral_constant<decltype(GemmConfig::N_Tile), GemmConfig::N_Tile>;
                using K_Tile =
                    ck_tile::integral_constant<decltype(GemmConfig::K_Tile), GemmConfig::K_Tile>;
                using M_Warp =
                    ck_tile::integral_constant<decltype(GemmConfig::M_Warp), GemmConfig::M_Warp>;
                using N_Warp =
                    ck_tile::integral_constant<decltype(GemmConfig::N_Warp), GemmConfig::N_Warp>;
                using K_Warp =
                    ck_tile::integral_constant<decltype(GemmConfig::K_Warp), GemmConfig::K_Warp>;
                using M_Warp_Tile = ck_tile::integral_constant<decltype(GemmConfig::M_Warp_Tile),
                                                               GemmConfig::M_Warp_Tile>;
                using N_Warp_Tile = ck_tile::integral_constant<decltype(GemmConfig::N_Warp_Tile),
                                                               GemmConfig::N_Warp_Tile>;
                using K_Warp_Tile = ck_tile::integral_constant<decltype(GemmConfig::K_Warp_Tile),
                                                               GemmConfig::K_Warp_Tile>;

                using kPadM =
                    ck_tile::integral_constant<decltype(GemmConfig::kPadM), GemmConfig::kPadM>;
                using kPadN =
                    ck_tile::integral_constant<decltype(GemmConfig::kPadN), GemmConfig::kPadN>;
                using kPadK =
                    ck_tile::integral_constant<decltype(GemmConfig::kPadK), GemmConfig::kPadK>;

                using PermuteA = ck_tile::integral_constant<decltype(GemmConfig::PermuteA),
                                                            GemmConfig::PermuteA>;
                using PermuteB = ck_tile::integral_constant<decltype(GemmConfig::PermuteB),
                                                            GemmConfig::PermuteB>;
                using UseStructuredSparsity =
                    ck_tile::integral_constant<decltype(GemmConfig::UseStructuredSparsity),
                                               GemmConfig::UseStructuredSparsity>;
                using KPersistent = ck_tile::integral_constant<decltype(Persistent), Persistent>;
                using Preshuffle  = ck_tile::integral_constant<decltype(GemmConfig::Preshuffle),
                                                               GemmConfig::Preshuffle>;

                using NumWaveGroups =
                    ck_tile::integral_constant<decltype(GemmConfig::NumWaveGroups),
                                               GemmConfig::NumWaveGroups>;
                using DoubleSmemBuffer =
                    ck_tile::integral_constant<decltype(GemmConfig::DoubleSmemBuffer),
                                               GemmConfig::DoubleSmemBuffer>;
                using TransposeC = ck_tile::integral_constant<decltype(GemmConfig::TransposeC),
                                                              GemmConfig::TransposeC>;

                using HasHotLoop      = decltype(has_hot_loop_);
                using MemoryOperation = decltype(memory_operation_);
                using TailNum         = decltype(tail_number_);

                using Scheduler = ck_tile::integral_constant<decltype(GemmConfig::Scheduler),
                                                             GemmConfig::Scheduler>;
                using TilePartitionerGroupNum =
                    ck_tile::integral_constant<decltype(GemmConfig::TileParitionerGroupNum),
                                               GemmConfig::TileParitionerGroupNum>;
                using TilePartitionerM01 =
                    ck_tile::integral_constant<decltype(GemmConfig::TileParitionerM01),
                                               GemmConfig::TileParitionerM01>;
                using Pipeline = ck_tile::integral_constant<decltype(GemmConfig::Pipeline),
                                                            GemmConfig::Pipeline>;

                using kBlockPerCu = ck_tile::integral_constant<decltype(GemmConfig::kBlockPerCu),
                                                               GemmConfig::kBlockPerCu>;
            };

            struct Inp
            {
                using InputADataType      = ADataType;
                using InputBDataType      = BDataType;
                using InputDsDataType     = DsDataType;
                using InputCDataType      = CDataType;
                using InputAccDataType    = AccDataType;
                using InputALayout        = ALayout;
                using InputBLayout        = BLayout;
                using InputDsLayout       = DsLayout;
                using InputELayout        = ELayout;
                using InputCDEElementWise = CDEElementWise;
            };

            float ave_time = ck_tile::launch_kernel(
                s, ck_tile::experimental::builder::UniversalFactory<Algo, Inp>::make_kernel(args, s));

            return ave_time;
        };

        using KTrue  = ck_tile::integral_constant<bool, true>;
        using KFalse = ck_tile::integral_constant<bool, false>;

        using BoolVariant = std::variant<KTrue, KFalse>;

        auto make_bool_variant = [](bool b) -> BoolVariant {
            if(b)
            {
                return KTrue{};
            }
            else
            {
                return KFalse{};
            }
        };

        using KTailNumOdd =
            ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>;
        using KTailNumEven =
            ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>;
        using KTailNumFull =
            ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>;
        using TailNumVariant = std::variant<KTailNumOdd, KTailNumEven, KTailNumFull>;

        auto make_tailnum_variant = [](ck_tile::TailNumber tail_number) -> TailNumVariant {
            switch(tail_number)
            {
            case ck_tile::TailNumber::Full: return KTailNumFull{};
            case ck_tile::TailNumber::Odd: return KTailNumOdd{};
            case ck_tile::TailNumber::Even: return KTailNumEven{};
            case ck_tile::TailNumber::Empty:
            case ck_tile::TailNumber::One:
            case ck_tile::TailNumber::Two:
            case ck_tile::TailNumber::Three:
            case ck_tile::TailNumber::Four:
            case ck_tile::TailNumber::Five:
            case ck_tile::TailNumber::Six:
            case ck_tile::TailNumber::Seven:
            default: throw std::runtime_error("Case not handled");
            }
        };

        using KSet    = ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                   ck_tile::memory_operation_enum::set>;
        using KAtomic = ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                   ck_tile::memory_operation_enum::atomic_add>;

        using MemoryOperationVariant = std::variant<KSet, KAtomic>;

        auto make_memory_op_variant =
            [](ck_tile::memory_operation_enum ev) -> MemoryOperationVariant {
            if(ev == ck_tile::memory_operation_enum::set)
            {
                return KSet{};
            }
            else
            {
                return KAtomic{};
            }
        };

        float ave_time = std::visit(
            kernel_launch_visitor,
            make_bool_variant(has_hot_loop),
            make_tailnum_variant(tail_num),
            make_memory_op_variant(args.k_batch == 1 ? ck_tile::memory_operation_enum::set
                                                     : ck_tile::memory_operation_enum::atomic_add)

        );

        return ave_time;
    }
};
