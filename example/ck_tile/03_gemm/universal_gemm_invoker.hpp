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
    struct GemmShapeTmp
    {
        static constexpr auto kM = kMPerBlock;
        static constexpr auto kN = kNPerBlock;
        static constexpr auto kK = kKPerBlock;
    };
    using Type = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShapeTmp,
        GroupNum,
        M01>;
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
        // const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
        // const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain *
        // GemmConfig::K_Tile; const ck_tile::index_t num_loop    =
        // TilePartitioner::GetLoopNum(K_split); const bool has_hot_loop            =
        // BaseGemmPipeline::BlockHasHotloop(num_loop); const ck_tile::TailNumber tail_num =
        // BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        // const ck_tile::index_t num_loop = 64;
        const bool has_hot_loop            = true;
        const ck_tile::TailNumber tail_num = ck_tile::TailNumber::Full;

        float ave_time{0};

        const auto kernel_launch_visitor = [&](const auto has_hot_loop_,
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

            // if(s.log_level_ > 0)
            // {
            //     std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
            //               << "shape: " << GemmShape::GetName() << '\n'
            //               << "problem: " << UniversalGemmProblem::GetName() << '\n'
            //               << "pipeline: " << GemmPipeline::GetName() << '\n'
            //               << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
            //               << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
            //               << "}" << std::endl;
            // }

            // Declare rotating_mem_ptr here so it stays in scope until it is needed
            // std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr;
            // std::function<void()> preprocess;

            // auto clear_gemm_output = [&]() {
            //     if(args.k_batch > 1)
            //         hipGetErrorString(hipMemsetAsync(
            //             args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
            // };

            // if(s.flush_cache_)
            // {
            //     std::cout << "Flushing cache..." << std::endl;

            //     ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
            //         args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            //     ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
            //         args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            //     auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            //     auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            //     rotating_mem_ptr =
            //         std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
            //             kargs.as_ptr[0],
            //             kargs.bs_ptr[0],
            //             s.rotating_count_,
            //             size_a_buffer,
            //             size_b_buffer);
            //     rotating_mem_ptr->Print();

            //     preprocess = [&]() {
            //         ck_tile::flush_icache();
            //         rotating_mem_ptr->Next();
            //         clear_gemm_output();
            //     };
            // }
            // else
            // {
            //     preprocess = clear_gemm_output;
            // }

            // ave_time = ck_tile::launch_kernel_time_mask(
            //     s,
            //     preprocess,
            //     ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0,
            //     kargs));

            ave_time = ck_tile::launch_kernel(
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

        ave_time = std::visit(
            kernel_launch_visitor,
            make_bool_variant(has_hot_loop),
            make_tailnum_variant(tail_num),
            make_memory_op_variant(args.k_batch == 1 ? ck_tile::memory_operation_enum::set
                                                     : ck_tile::memory_operation_enum::atomic_add)

        );

        return ave_time;
    }
};
