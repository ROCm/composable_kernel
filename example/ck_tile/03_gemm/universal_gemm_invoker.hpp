// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <functional>
#include "gemm_utils.hpp"

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
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::
                sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
            GemmConfig::PermuteA,
            GemmConfig::PermuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfig::TileParitionerGroupNum,
                                                       GemmConfig::TileParitionerM01>;

        using Traits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                               GemmConfig::kPadN,
                                               GemmConfig::kPadK,
                                               ALayout,
                                               BLayout,
                                               ELayout,
                                               GemmConfig::NumWaveGroups>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                             GemmConfig::kPadN,
                                             GemmConfig::kPadK,
                                             GemmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             ELayout,
                                             GemmConfig::TransposeC,
                                             GemmConfig::UseStructuredSparsity,
                                             Persistent,
                                             GemmConfig::NumWaveGroups,
                                             GemmConfig::Preshuffle>;
        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        using BaseGemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

        const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
        const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        float ave_time{0};

        const auto kernel_launch_visitor = [&](const auto has_hot_loop_,
                                               const auto tail_number_,
                                               const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto scheduler        = GemmConfig::Scheduler;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;

            using GemmPipeline = typename PipelineTypeTraits<
                GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 AccDataType,
                                                 CDataType,
                                                 DsLayout,
                                                 ELayout,
                                                 CDEElementWise,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 GemmConfig::M_Warp,
                                                 GemmConfig::N_Warp,
                                                 GemmConfig::M_Warp_Tile,
                                                 GemmConfig::N_Warp_Tile,
                                                 GemmConfig::K_Warp_Tile,
                                                 UniversalGemmProblem::TransposeC,
                                                 memory_operation,
                                                 GemmConfig::NumWaveGroups>>;

            using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);

            const dim3 grids  = Persistent ? Kernel::MaxOccupancyGridSize(s)
                                           : Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                          << "shape: " << GemmShape::GetName() << '\n'
                          << "problem: " << UniversalGemmProblem::GetName() << '\n'
                          << "pipeline: " << GemmPipeline::GetName() << '\n'
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }

            // Declare rotating_mem_ptr here so it stays in scope until it is needed
            std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr;
            std::function<void()> preprocess;

            auto clear_gemm_output = [&]() {
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
            };

            if(s.flush_cache_)
            {
                std::cout << "Flushing cache..." << std::endl;

                ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                    args.M, args.K, args.stride_A, is_row_major(ALayout{})));
                ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                    args.K, args.N, args.stride_B, is_row_major(BLayout{})));

                auto size_a_buffer = a_m.get_element_space_size_in_bytes();
                auto size_b_buffer = b_n.get_element_space_size_in_bytes();

                rotating_mem_ptr =
                    std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
                        kargs.as_ptr[0],
                        kargs.bs_ptr[0],
                        s.rotating_count_,
                        size_a_buffer,
                        size_b_buffer);
                rotating_mem_ptr->Print();

                preprocess = [&]() {
                    ck_tile::flush_icache();
                    rotating_mem_ptr->Next();
                    clear_gemm_output();
                };
            }
            else
            {
                preprocess = clear_gemm_output;
            }

            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                preprocess,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

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
