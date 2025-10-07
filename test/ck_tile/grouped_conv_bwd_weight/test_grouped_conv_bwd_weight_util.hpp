
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

struct GroupedConvolutionBackwardWeightInvoker
{
    template <ck_tile::index_t NDimSpatial,
              typename GemmConfig,
              typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              typename InLayout,
              typename WeiLayout,
              typename OutLayout,
              ck_tile::index_t  NumGroupMerge  = 1,
              typename DsDataType     = ck_tile::tuple<>,
              typename DsLayout       = ck_tile::tuple<>,
              typename CDEElementWise = ck_tile::element_wise::PassThrough>
    static float grouped_conv_bwd_weight(const ck_tile::GroupedConvBwdWeightHostArgs& args,
                                         const ck_tile::stream_config& s)
    {
        constexpr int kBlockPerCu = 1;

        constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
        constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
        constexpr ck_tile::index_t K_Tile = GemmConfig::K_Tile;

        constexpr ck_tile::index_t M_Warp = GemmConfig::M_Warp;
        constexpr ck_tile::index_t N_Warp = GemmConfig::N_Warp;
        constexpr ck_tile::index_t K_Warp = GemmConfig::K_Warp;

        constexpr ck_tile::index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
        constexpr ck_tile::index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
        constexpr ck_tile::index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;

        constexpr ck_tile::index_t VectorSizeA = GemmConfig::VectorSizeA;
        constexpr ck_tile::index_t VectorSizeB = GemmConfig::VectorSizeB;
        constexpr ck_tile::index_t VectorSizeC = GemmConfig::VectorSizeC;

        // Implicit GEMM Traits
        using CodegenShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        constexpr auto ConvSpec      = ck_tile::ConvolutionSpecialization::Default;
        using TilePartitioner        = ck_tile::GemmTile1DPartitioner<CodegenShape>;
        using GroupedConvTraitsType  = ck_tile::GroupedConvTraits<NDimSpatial,
                                                                  ConvSpec,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  DsLayout,
                                                                  OutLayout,
                                                                  VectorSizeA,
                                                                  VectorSizeB,
                                                                  VectorSizeC>;
        using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<
            InDataType,
            WeiDataType,
            AccDataType,
            CodegenShape,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdWeight,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            InDataType,
            true,
            GroupedConvTraitsType::VectorSizeA,
            GroupedConvTraitsType::VectorSizeB>;
        using CodegenPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        const auto Run = [&](const auto memory_operation_) {
            constexpr auto memory_operation = memory_operation_.value;

            using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                InDataType,
                WeiDataType,
                DsDataType,
                AccDataType,
                OutDataType,
                typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                ck_tile::tensor_layout::gemm::RowMajor,
                CDEElementWise,
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
                GroupedConvTraitsType::VectorSizeC>>;

            using Kernel = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType,
                                                                           TilePartitioner,
                                                                           CodegenPipeline,
                                                                           ConvEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);

            const dim3 grids  = Kernel::GridSize(kargs);
            const dim3 blocks = Kernel::BlockSize();

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
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << '\n'
                          << "Vector size A: " << CodegenPipeline::GetVectorSizeA()
                          << ", Vector size B: " << CodegenPipeline::GetVectorSizeB()
                          << ", Vector size C: " << ConvEpilogue::GetVectorSizeC() << std::endl;
            }

            float ave_time = ck_tile::launch_kernel_time_mask(
                s,
                Kernel::Preprocess(kargs, s),
                ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

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
};

template <typename Tuple>
class TestCkTileGroupedConvBwdWeight : public ::testing::Test
{
  protected:
    using NDimSpatial     = std::tuple_element_t<0, Tuple>;
    using GemmConfig      = std::tuple_element_t<1, Tuple>;
    using InDataType      = std::tuple_element_t<2, Tuple>;
    using WeiDataType     = std::tuple_element_t<3, Tuple>;
    using AccDataType     = std::tuple_element_t<4, Tuple>;
    using OutDataType     = std::tuple_element_t<5, Tuple>;
    using InLayout        = std::tuple_element_t<6, Tuple>;
    using WeiLayout       = std::tuple_element_t<7, Tuple>;
    using OutLayout       = std::tuple_element_t<8, Tuple>;
    using NumGroupMerge   = std::tuple_element_t<9, Tuple>;

  public:
    void run(const ck_tile::GroupedConvBwdWeightHostArgs& args)
    {
      using Invoker = GroupedConvolutionBackwardWeightInvoker;

      ck_tile::stream_config config{nullptr, true, 1, /*warm-up*/0, /*number of iters*/1};
      Invoker::invoke_grouped_conv_bwd_weight<NDimSpatial,
                                                  GemmWarpConfig,
                                                  GemmTileConfig,
                                                  GemmVectorLoads,
                                                  Invoker,
                                                  InDataType,
                                                  WeiDataType,
                                                  AccDataType,
                                                  OutDataType,
                                                  InLayout,
                                                  WeiLayout,
                                                  OutLayout,
                                                  NumGroupsToMerge>(args, config);
    }
}

/// @brief The Grouped Conv kernel host arguments.
///
/// @par Overview
///      This structure is passed to Grouped Convolution Kernels when creating kernel
///      arguments object. It contain all necessary information required to
///      build proper kernel argument and launch kernel on GPU.
template <typename InPtr, typename WeiPtr, typename OutPtr>
struct GroupedConvHostArgs : public conv::ConvParam
{
    CK_TILE_HOST GroupedConvHostArgs() = delete;
    CK_TILE_HOST GroupedConvHostArgs(ConvParam conv_param,
                                     InPtr in_ptr_,
                                     WeiPtr wei_ptr_,
                                     const std::vector<const void*> ds_ptr_,
                                     OutPtr out_ptr_,
                                     index_t k_batch_)
        : conv::ConvParam(conv_param),
          in_ptr(in_ptr_),
          wei_ptr(wei_ptr_),
          ds_ptr(ds_ptr_),
          out_ptr(out_ptr_),
          k_batch(k_batch_)
    {
    }

    InPtr in_ptr;
    WeiPtr wei_ptr;
    const std::vector<const void*> ds_ptr;
    OutPtr out_ptr;
    index_t k_batch;
};

using GroupedConvFwdHostArgs       = GroupedConvHostArgs<const void*, const void*, void*>;
using GroupedConvBwdWeightHostArgs = GroupedConvHostArgs<const void*, void*, const void*>;
using GroupedConvBwdDataHostArgs   = GroupedConvHostArgs<void*, const void*, const void*>;