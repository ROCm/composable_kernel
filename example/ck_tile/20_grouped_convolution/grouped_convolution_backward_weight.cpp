// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "grouped_convolution_utils.hpp"

template <ck_tile::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DsDataType     = ck_tile::tuple<>,
          typename DsLayout       = ck_tile::tuple<>,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float grouped_conv_bwd_weight(const ck_tile::GroupedConvBwdWeightHostArgs& args,
                              const ck_tile::stream_config& s)
{
    constexpr int kBlockPerCu = 1;

    constexpr ck_tile::index_t M_Tile = 64;
    constexpr ck_tile::index_t N_Tile = 64;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;

    constexpr ck_tile::index_t VectorSizeA = 8;
    constexpr ck_tile::index_t VectorSizeB = 8;
    constexpr ck_tile::index_t VectorSizeC = 8;

    // Implicit GEMM Traits
    using CodegenShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;
    using TilePartitioner   = ck_tile::GemmTile1DPartitioner<CodegenShape>;
    using GroupedConvTraitsType =
        ck_tile::GroupedConvTraits<NDimSpatial, ConvSpec, InLayout, WeiLayout, DsLayout, OutLayout>;
    using CodegenPipelineProblem =
        ck_tile::GemmPipelineProblem<InDataType,
                                     WeiDataType,
                                     AccDataType,
                                     CodegenShape,
                                     typename GroupedConvTraitsType::GroupedConvImplicitGemmTraits,
                                     InDataType,
                                     true,
                                     VectorSizeA,
                                     VectorSizeB>;
    using CodegenPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto memory_operation = memory_operation_.value;

        using ConvEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<InDataType,
                                             WeiDataType,
                                             DsDataType,
                                             AccDataType,
                                             OutDataType,
                                             typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                                             ck_tile::tensor_layout::gemm::RowMajor,
                                             CDEElementWise,
                                             CodegenPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation,
                                             1,
                                             true,
                                             VectorSizeC>>;

        using Kernel = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType,
                                                                       TilePartitioner,
                                                                       CodegenPipeline,
                                                                       ConvEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << CodegenShape::GetName() << '\n'
                      << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                      << "pipeline: " << CodegenPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << '\n'
                      << "Vector size A: " << CodegenPipeline::GetVectorSizeA()
                      << ", Vector size B: " << CodegenPipeline::GetVectorSizeB()
                      << ", Vector size C: " << ConvEpilogue::GetVectorSizeC() << std::endl;
        }

        float ave_time = ck_tile::launch_kernel_preprocess(
            s,
            Kernel::Preprocess(kargs, s),
            ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    };

    if(args.k_batch == 1)
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::atomic_add>{});
    }
}

#include "run_grouped_convolution_bwd_weight_example.inc"

template <typename InPrecType, typename WeiPrecType = InPrecType, typename OutPrecType = InPrecType>
int run_grouped_conv_bwd_weight_example_prec_type(
    std::string in_layout, std::string wei_layout, std::string out_layout, int argc, char* argv[])
{
    using NWGC   = ck_tile::tensor_layout::convolution::NWGC;
    using NHWGC  = ck_tile::tensor_layout::convolution::NHWGC;
    using NDHWGC = ck_tile::tensor_layout::convolution::NDHWGC;

    using GKXC   = ck_tile::tensor_layout::convolution::GKXC;
    using GKYXC  = ck_tile::tensor_layout::convolution::GKYXC;
    using GKZYXC = ck_tile::tensor_layout::convolution::GKZYXC;

    using NWGK   = ck_tile::tensor_layout::convolution::NWGK;
    using NHWGK  = ck_tile::tensor_layout::convolution::NHWGK;
    using NDHWGK = ck_tile::tensor_layout::convolution::NDHWGK;

    if(in_layout == "NWGC" && wei_layout == "GKXC" && out_layout == "NWGK")
    {
        return run_grouped_conv_bwd_weight_example_with_layouts<ck_tile::number<1>{},
                                                                InPrecType,
                                                                WeiPrecType,
                                                                OutPrecType>(
            argc, argv, NWGC{}, GKXC{}, NWGK{});
    }
    else if(in_layout == "NHWGC" && wei_layout == "GKYXC" && out_layout == "NHWGK")
    {
        return run_grouped_conv_bwd_weight_example_with_layouts<ck_tile::number<2>{},
                                                                InPrecType,
                                                                WeiPrecType,
                                                                OutPrecType>(
            argc, argv, NHWGC{}, GKYXC{}, NHWGK{});
    }
    else if(in_layout == "NDHWGC" && wei_layout == "GKZYXC" && out_layout == "NDHWGK")
    {
        return run_grouped_conv_bwd_weight_example_with_layouts<ck_tile::number<3>{},
                                                                InPrecType,
                                                                WeiPrecType,
                                                                OutPrecType>(
            argc, argv, NDHWGC{}, GKZYXC{}, NDHWGK{});
    }
    else
    {
        throw std::runtime_error("Unsupported memory layout!");
    }
}

int run_grouped_conv_bwd_weight_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type  = arg_parser.get_str("prec");
    std::string in_layout  = arg_parser.get_str("in_layout");
    std::string wei_layout = arg_parser.get_str("wei_layout");
    std::string out_layout = arg_parser.get_str("out_layout");

    if(data_type == "fp16")
    {
        return run_grouped_conv_bwd_weight_example_prec_type<ck_tile::half_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_grouped_conv_bwd_weight_example_prec_type<ck_tile::bf16_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation!");
    }
}

int main(int argc, char* argv[]) { return !run_grouped_conv_bwd_weight_example(argc, argv); }
