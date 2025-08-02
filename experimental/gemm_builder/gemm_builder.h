#pragma once
#include <iostream>
#include <concepts>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace ck_tile::builder {

// Some sample host args to describe a GEMM, with defaults set to zero.
struct GemmHostArgs
{
    ck_tile::index_t m   = 0;
    ck_tile::index_t n   = 0;
    ck_tile::index_t k   = 0;
    ck_tile::index_t lda = 0;
    ck_tile::index_t ldb = 0;
    ck_tile::index_t ldc = 0;
    index_t k_batch_     = 0;
    const void* a        = nullptr;
    const void* b        = nullptr;
    void* c              = nullptr;
};

// Tag for column major layout.
using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

// Tag for row major layout.
using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;

// Requirements for struct to define the data types used in the GEMM operation.
//
// Example that satifies this constraint:
// struct GemmTypes {
//     using ADataType = float;
//     using BDataType = float;
//     using CDataType = float;
//     using AccDataType = float;
// };
template <typename T>
concept DefinesGemmTypes =
    requires {
        typename T::ADataType;
        typename T::BDataType;
        typename T::CDataType;
        typename T::AccDataType;
    } && std::is_arithmetic_v<typename T::ADataType> &&
    std::is_arithmetic_v<typename T::BDataType> && std::is_arithmetic_v<typename T::CDataType> &&
    std::is_arithmetic_v<typename T::AccDataType>;

// Requirements for struct that defines the layout used in the GEMM operation.
//
// Example that satisfies this constraint:
// struct Layouts {
//     using ALayout = RowMajor;
//     using BLayout = ColMajor;
//     using CLayout = RowMajor;
// };
template <typename T>
concept DefinesGemmLayout = requires {
    typename T::ALayout;
    typename T::BLayout;
    typename T::CLayout;
};

// A dummy placeholder for a real GEMM.
class Gemm
{
    public:
    void run([[maybe_unused]] GemmHostArgs args) const
    {
        std::cout << "Running fake GEMM" << std::endl;
    }
};

// Returns the GemmUniversal GemmConfig.
//
// Captures all the madness in tile_example_gemm_universal!
template <DefinesGemmTypes Types>
struct GemmConfigForTypes
{
    using PrecType = Types::AccDataType;

    static constexpr int CK_TILE_PIPELINE_COMPUTE_V3 = 1;
    static consteval auto get_k_warp_tile(auto M_Warp_Tile)
    {
        return (M_Warp_Tile == 32) ? 16 : 32;
    }
    // Use GemmConfigComputeV3 from tile_example_gemm_universal.
    struct GemmConfig
    {
        // Compute V3 only support Intrawave scheduler
        static constexpr ck_tile::index_t M_Tile = 16;
        static constexpr ck_tile::index_t N_Tile = 64;
        static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

        static constexpr ck_tile::index_t M_Warp = 1;
        static constexpr ck_tile::index_t N_Warp = 4;
        static constexpr ck_tile::index_t K_Warp = 1;

        static constexpr ck_tile::index_t M_Warp_Tile = 16;
        static constexpr ck_tile::index_t N_Warp_Tile = 16;
        static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile(M_Warp_Tile);

        static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;

        static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        static constexpr bool kPadM = false;
        static constexpr bool kPadN = false;
        static constexpr bool kPadK = false;

        static constexpr bool PermuteA   = false;
        static constexpr bool PermuteB   = false;
        static constexpr bool TransposeC = false;

        static constexpr bool UseStructuredSparsity = false;

        static constexpr int kBlockPerCu                         = 1;
        static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        static constexpr ck_tile::index_t TileParitionerM01      = 4;
        static constexpr ck_tile::index_t NumWaveGroups          = 1;

        static constexpr bool Preshuffle       = false;
        static constexpr bool DoubleSmemBuffer = false;
    };
};

// A minimal GEMM builder, this is where all the work will be.
template <DefinesGemmTypes Types, DefinesGemmLayout Layout>
struct GemmBuilder
{
    using value       = Gemm;
    using types_type  = Types;
    using layout_type = Layout;

    static constexpr bool PERSISTENT = false;

    using GemmConfig = typename GemmConfigForTypes<Types>::GemmConfig;

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
                                           typename Layout::ALayout,
                                           typename Layout::BLayout,
                                           typename Layout::CLayout,
                                           GemmConfig::NumWaveGroups>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 typename Layout::ALayout,
                                                                 typename Layout::BLayout,
                                                                 typename Layout::CLayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 PERSISTENT,
                                                                 GemmConfig::NumWaveGroups,
                                                                 GemmConfig::Preshuffle>;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<typename Types::ADataType,
                                                                       typename Types::BDataType,
                                                                       typename Types::AccDataType,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       GemmConfig::Scheduler>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<typename Types::ADataType,
                                                             typename Types::BDataType,
                                                             typename Types::AccDataType,
                                                             GemmShape,
                                                             Traits>;

    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<typename Types::ADataType,
                                         typename Types::BDataType,
                                         ck_tile::tuple<>,
                                         typename Types::AccDataType,
                                         typename Types::CDataType,
                                         ck_tile::tuple<>,
                                         typename Layout::CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         UniversalGemmProblem::kBlockSize,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC,
                                         ck_tile::memory_operation_enum::set,
                                         GemmConfig::NumWaveGroups>>;

    using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmPipeline>;
};

} // namespace ck_tile::builder
