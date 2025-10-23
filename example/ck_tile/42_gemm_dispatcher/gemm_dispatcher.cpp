
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

// template <typename GemmConfig,
//          typename ADataType,
//          typename BDataType,
//          typename DsDataType,
//          typename AccDataType,
//          typename CDataType,
//          typename ALayout,
//          typename BLayout,
//          typename DsLayout,
//          typename ELayout,
//          bool Persistent,
//          typename CDEElementWise,
//          bool has_hot_loop,
//          ck_tile::TailNumber tail_num,
//          ck_tile::memory_operation_enum memory_operation>
// float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
// {
//    using GemmShape = ck_tile::TileGemmShape<
//             ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
//             ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
//             ck_tile::
//                 sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile,
//                 GemmConfig::K_Warp_Tile>,
//             GemmConfig::PermuteA,
//             GemmConfig::PermuteB>;

//         using TilePartitioner =
//             ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
//                                                        GemmConfig::TileParitionerGroupNum,
//                                                        GemmConfig::TileParitionerM01>;

//         using Traits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
//                                                GemmConfig::kPadN,
//                                                GemmConfig::kPadK,
//                                                ALayout,
//                                                BLayout,
//                                                ELayout,
//                                                GemmConfig::NumWaveGroups>;

//         using GemmUniversalTraits =
//             ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
//                                              GemmConfig::kPadN,
//                                              GemmConfig::kPadK,
//                                              GemmConfig::DoubleSmemBuffer,
//                                              ALayout,
//                                              BLayout,
//                                              ELayout,
//                                              GemmConfig::TransposeC,
//                                              GemmConfig::UseStructuredSparsity,
//                                              Persistent,
//                                              GemmConfig::NumWaveGroups,
//                                              GemmConfig::Preshuffle>;
//         using GemmPipelineProblem =
//             ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

//         using BaseGemmPipeline = typename PipelineTypeTraits<
//             GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

//       //   const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
//       //   const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain *
//       GemmConfig::K_Tile;
//       //   const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
//       //   const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
//       //   const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

//         constexpr auto scheduler        = GemmConfig::Scheduler;

//         using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
//                                                                                BDataType,
//                                                                                AccDataType,
//                                                                                GemmShape,
//                                                                                GemmUniversalTraits,
//                                                                                scheduler,
//                                                                                has_hot_loop,
//                                                                                tail_number>;

//             using GemmPipeline = typename PipelineTypeTraits<
//                 GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

//             using GemmEpilogue = ck_tile::CShuffleEpilogue<
//                 ck_tile::CShuffleEpilogueProblem<ADataType,
//                                                  BDataType,
//                                                  DsDataType,
//                                                  AccDataType,
//                                                  CDataType,
//                                                  DsLayout,
//                                                  ELayout,
//                                                  CDEElementWise,
//                                                  TilePartitioner::MPerBlock,
//                                                  TilePartitioner::NPerBlock,
//                                                  GemmConfig::M_Warp,
//                                                  GemmConfig::N_Warp,
//                                                  GemmConfig::M_Warp_Tile,
//                                                  GemmConfig::N_Warp_Tile,
//                                                  GemmConfig::K_Warp_Tile,
//                                                  UniversalGemmProblem::TransposeC,
//                                                  memory_operation,
//                                                  GemmConfig::NumWaveGroups>>;

//             using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
//             auto kargs   = Kernel::MakeKernelArgs(args);

//             const dim3 grids  = Persistent ? Kernel::MaxOccupancyGridSize(s)
//                                            : Kernel::GridSize(args.M, args.N, args.k_batch);
//             const dim3 blocks = Kernel::BlockSize();

//             if(!Kernel::IsSupportedArgument(kargs))
//             {
//                 throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
//             }

//             float ave_time = ck_tile::launch_kernel_time_mask(
//                 s,
//                 ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0,
//                 kargs));

//         return ave_time
// }

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

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

    std::visit([](auto&& bv) { std::cout << bv.value << std::endl; }, make_bool_variant(true));
    std::visit([](auto&& bv) { std::cout << bv.value << std::endl; }, make_bool_variant(false));

    using KTailNumOdd  = ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>;
    using KTailNumEven = ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>;
    // using KTailNumFull = ck_tile::integral_constant<ck_tile::TailNumber,
    // ck_tile::TailNumber::Even>;

    using TailNumVariant      = std::variant<KTailNumOdd, KTailNumEven>;
    auto make_tailnum_variant = [](ck_tile::index_t tail_num) -> TailNumVariant {
        if(tail_num % 2 == 1)
        {
            return KTailNumOdd{};
        }
        else
        {
            return KTailNumEven{};
        }
    };

    std::visit(
        [](auto&& bv, auto&& tnv) {
            std::cout << "bool val: " << bv.value << ", tn val: " << tnv.value << std::endl;
        },
        make_bool_variant(false),
        make_tailnum_variant(5));

    using KSet    = ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                               ck_tile::memory_operation_enum::set>;
    using KAtomic = ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                               ck_tile::memory_operation_enum::atomic_add>;

    using MemoryOperationVariant = std::variant<KSet, KAtomic>;
    //    std::cout << make_bool_variant(true).value;
    //    std::cout << make_bool_variant(false).value;

    return 0;
}
