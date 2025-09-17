#include <gtest/gtest.h>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
//#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"


#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"


using namespace ck_tile;

class TestCShuffleEpilogue : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestCShuffleEpilogue, LdsTileEncoding)
{
  constexpr index_t M_Tile = 8;
  constexpr index_t N_Tile = 128;
  constexpr index_t K_Tile = 64;

  constexpr index_t M_Warp = 2;
  constexpr index_t N_Warp = 2;
  constexpr index_t K_Warp = 1;

  constexpr index_t M_Warp_Tile = 4;
  constexpr index_t N_Warp_Tile = 64;
  constexpr index_t K_Warp_Tile = 16;

  constexpr index_t VectorSizeA = 8;
  constexpr index_t VectorSizeB = 8;
  constexpr index_t VectorSizeC = 8;

  constexpr index_t NumGroupsToMerge = 8;
  constexpr index_t NDimSpatial = 2;

  constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;

  using InDataType  = ck_tile::half_t;
  using WeiDataType = ck_tile::half_t;
  using AccDataType = float;
  using DsDataType  = ck_tile::tuple<>;
  using OutDataType = ck_tile::half_t;

  using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
  using WeiLayout  = ck_tile::tensor_layout::convolution::GKYXC;
  using DsLayout  = ck_tile::tuple<>;
  using OutLayout  = ck_tile::tensor_layout::convolution::NHWGK;

  using GroupedConvTraitsType =
        ck_tile::GroupedConvTraits<NDimSpatial, ConvSpec, InLayout, WeiLayout, DsLayout, OutLayout, NumGroupsToMerge>;

  using CodegenShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

  using TilePartitioner   = ck_tile::GemmTile1DPartitioner<CodegenShape>;

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

  using MemoryOp = ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>;

  using ConvEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<
              InDataType,
              WeiDataType,
              DsDataType,
              AccDataType,
              OutDataType,
              typename GroupedConvTraitsType::ImplicitGemmDsLayout,
              ck_tile::tensor_layout::gemm::RowMajor,
              ck_tile::element_wise::PassThrough,
              TilePartitioner::MPerBlock,
              TilePartitioner::NPerBlock,
              M_Warp,
              N_Warp,
              M_Warp_Tile,
              N_Warp_Tile,
              K_Warp_Tile,
              CodegenPipelineProblem::TransposeC,
              MemoryOp::value,
              1,
              true,
              VectorSizeC>>;

  constexpr auto encoding = ConvEpilogue::MakeLdsDistributionEncode();
  print(encoding);
  constexpr auto lds_dstr = make_static_tile_distribution(encoding);
  
  EXPECT_EQ(lds_dstr.get_num_of_dimension_x(), 2);
  EXPECT_EQ(lds_dstr.get_num_of_dimension_y(), 4);
  EXPECT_EQ(lds_dstr.get_num_of_dimension_p(), 2);
  EXPECT_EQ(lds_dstr.get_num_of_dimension_r(), 0);

  const auto distributed_spans = lds_dstr.get_distributed_spans();
  EXPECT_EQ(distributed_spans.size(), 2);
  EXPECT_EQ(distributed_spans[number<0>{}].impl_.size(), 3);
  EXPECT_EQ(distributed_spans[number<1>{}].impl_.size(), 1);
}
