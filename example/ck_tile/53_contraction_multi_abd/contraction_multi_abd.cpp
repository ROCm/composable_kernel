// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"
#include "contraction_multi_abd_utils.hpp"

template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename EDataType_,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          ck_tile::index_t NumDimG,
          ck_tile::index_t NumDimM,
          ck_tile::index_t NumDimN,
          ck_tile::index_t NumDimK,
          typename AElementWise   = ck_tile::element_wise::PassThrough,
          typename BElementWise   = ck_tile::element_wise::PassThrough,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float contraction_multi_abd_impl(
    const ck_tile::BatchedContractionMultiABDHostArgs<NumDimG,
                                                      NumDimM,
                                                      NumDimN,
                                                      NumDimK,
                                                      AsDataType_::size(),
                                                      BsDataType_::size(),
                                                      DsDataType_::size()>& args,
    const ck_tile::stream_config& s)
{
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

#if CK_TILE_USE_WMMA
    constexpr ck_tile::index_t M_Warp_Tile = 16;
    constexpr ck_tile::index_t N_Warp_Tile = 16;
    constexpr ck_tile::index_t K_Warp_Tile = 16;
#else
    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;
#endif

    constexpr bool DoubleSmemBuffer = false;
    constexpr bool kPadM            = false;
    constexpr bool kPadN            = false;
    constexpr bool kPadK            = false;
    constexpr bool TransposeC       = false;

    constexpr int kBlockPerCu                         = 1;
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                 kPadN,
                                                                 kPadK,
                                                                 DoubleSmemBuffer,
                                                                 AsLayout,
                                                                 BsLayout,
                                                                 ELayout,
                                                                 TransposeC>;

    using Problem = ck_tile::BatchedContractionMultiABDProblem<AsDataType_,
                                                               BsDataType_,
                                                               DsDataType_,
                                                               EDataType_,
                                                               NumDimG,
                                                               NumDimM,
                                                               NumDimN,
                                                               NumDimK>;

    constexpr auto scheduler = GEMM_PIPELINE_SCHEDULER;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<AsDataType_,
                                                                       BsDataType_,
                                                                       AccDataType_,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       scheduler,
                                                                       AElementWise,
                                                                       BElementWise>;

    using GemmPipeline = GEMM_PIPELINE<UniversalGemmProblem>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<AsDataType_,
                                         BsDataType_,
                                         DsDataType_,
                                         AccDataType_,
                                         EDataType_,
                                         DsLayout,
                                         ELayout,
                                         CDEElementWise,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         M_Warp,
                                         N_Warp,
                                         M_Warp_Tile,
                                         N_Warp_Tile,
                                         K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC>>;

    using Kernel = ck_tile::
        BatchedContractionMultiABDKernel<Problem, TilePartitioner, GemmPipeline, GemmEpilogue>;
    auto kargs = Kernel::MakeKernelArgs(args);

    const dim3 grids  = Kernel::GridSize(kargs);
    const dim3 blocks = Kernel::GetBlockSize();

    if(!Kernel::IsSupportedArguments(kargs))
    {
        throw std::runtime_error("Wrong! Arguments not supported!\n");
    }

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel: " << Kernel::GetKernelName() << '\n'
                  << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    auto kernel = ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs);
    return ck_tile::launch_kernel(s, kernel);
}

template <ck_tile::index_t NumDimG_,
          ck_tile::index_t NumDimM_,
          ck_tile::index_t NumDimN_,
          ck_tile::index_t NumDimK_>
struct ContractionMultiABDDimCase
{
    static constexpr ck_tile::index_t NumDimG = NumDimG_;
    static constexpr ck_tile::index_t NumDimM = NumDimM_;
    static constexpr ck_tile::index_t NumDimN = NumDimN_;
    static constexpr ck_tile::index_t NumDimK = NumDimK_;
};

// Supported runtime dimension-count combinations for this example:
//   (G, M, N, K) = (1, 1, 1, 1), (1, 2, 2, 2), (2, 1, 1, 1), (2, 2, 2, 2).
using SupportedContractionMultiABDDimCases = std::tuple<ContractionMultiABDDimCase<1, 1, 1, 1>,
                                                        ContractionMultiABDDimCase<1, 2, 2, 2>,
                                                        ContractionMultiABDDimCase<2, 1, 1, 1>,
                                                        ContractionMultiABDDimCase<2, 2, 2, 2>>;

template <typename DimCase,
          typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename EDataType_,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementWise,
          typename BElementWise,
          typename CDEElementWise>
bool try_contraction_multi_abd_case(
    float& result,
    const std::array<const void*, AsDataType_::size()>& as_ptr,
    const std::array<const void*, BsDataType_::size()>& bs_ptr,
    const std::array<const void*, DsDataType_::size()>& ds_ptr,
    void* e_ptr,
    const std::array<std::vector<ck_tile::index_t>, AsDataType_::size()>& As_dims,
    const std::array<std::vector<ck_tile::index_t>, BsDataType_::size()>& Bs_dims,
    const std::array<std::vector<ck_tile::index_t>, DsDataType_::size()>& Ds_dims,
    const std::vector<ck_tile::index_t>& E_dims,
    const std::array<std::vector<ck_tile::index_t>, AsDataType_::size()>& As_strides,
    const std::array<std::vector<ck_tile::index_t>, BsDataType_::size()>& Bs_strides,
    const std::array<std::vector<ck_tile::index_t>, DsDataType_::size()>& Ds_strides,
    const std::vector<ck_tile::index_t>& E_strides,
    const ck_tile::stream_config& s,
    ck_tile::index_t num_g_dims,
    ck_tile::index_t num_m_dims,
    ck_tile::index_t num_n_dims,
    ck_tile::index_t num_k_dims)
{
    constexpr ck_tile::index_t NumDimG = DimCase::NumDimG;
    constexpr ck_tile::index_t NumDimM = DimCase::NumDimM;
    constexpr ck_tile::index_t NumDimN = DimCase::NumDimN;
    constexpr ck_tile::index_t NumDimK = DimCase::NumDimK;

    if(num_g_dims != NumDimG || num_m_dims != NumDimM || num_n_dims != NumDimN ||
       num_k_dims != NumDimK)
    {
        return false;
    }

    using HostArgs = ck_tile::BatchedContractionMultiABDHostArgs<NumDimG,
                                                                 NumDimM,
                                                                 NumDimN,
                                                                 NumDimK,
                                                                 AsDataType_::size(),
                                                                 BsDataType_::size(),
                                                                 DsDataType_::size()>;
    using ADims    = typename HostArgs::ADims;
    using BDims    = typename HostArgs::BDims;
    using EDims    = typename HostArgs::EDims;

    std::array<ADims, AsDataType_::size()> as_dims_fixed{};
    std::array<BDims, BsDataType_::size()> bs_dims_fixed{};
    std::array<EDims, DsDataType_::size()> ds_dims_fixed{};
    std::array<ADims, AsDataType_::size()> as_strides_fixed{};
    std::array<BDims, BsDataType_::size()> bs_strides_fixed{};
    std::array<EDims, DsDataType_::size()> ds_strides_fixed{};

    for(std::size_t i = 0; i < AsDataType_::size(); ++i)
    {
        as_dims_fixed[i]    = to_fixed_dims<NumDimG + NumDimM + NumDimK>(As_dims[i]);
        as_strides_fixed[i] = to_fixed_dims<NumDimG + NumDimM + NumDimK>(As_strides[i]);
    }
    for(std::size_t i = 0; i < BsDataType_::size(); ++i)
    {
        bs_dims_fixed[i]    = to_fixed_dims<NumDimG + NumDimN + NumDimK>(Bs_dims[i]);
        bs_strides_fixed[i] = to_fixed_dims<NumDimG + NumDimN + NumDimK>(Bs_strides[i]);
    }
    for(std::size_t i = 0; i < DsDataType_::size(); ++i)
    {
        ds_dims_fixed[i]    = to_fixed_dims<NumDimG + NumDimM + NumDimN>(Ds_dims[i]);
        ds_strides_fixed[i] = to_fixed_dims<NumDimG + NumDimM + NumDimN>(Ds_strides[i]);
    }

    HostArgs fixed_args{as_ptr,
                        bs_ptr,
                        ds_ptr,
                        e_ptr,
                        as_dims_fixed,
                        bs_dims_fixed,
                        ds_dims_fixed,
                        to_fixed_dims<NumDimG + NumDimM + NumDimN>(E_dims),
                        as_strides_fixed,
                        bs_strides_fixed,
                        ds_strides_fixed,
                        to_fixed_dims<NumDimG + NumDimM + NumDimN>(E_strides)};

    result = contraction_multi_abd_impl<AsDataType_,
                                        BsDataType_,
                                        DsDataType_,
                                        AccDataType_,
                                        EDataType_,
                                        AsLayout,
                                        BsLayout,
                                        DsLayout,
                                        ELayout,
                                        NumDimG,
                                        NumDimM,
                                        NumDimN,
                                        NumDimK,
                                        AElementWise,
                                        BElementWise,
                                        CDEElementWise>(fixed_args, s);
    return true;
}

template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename EDataType_,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementWise,
          typename BElementWise,
          typename CDEElementWise,
          std::size_t... CaseIds>
bool dispatch_contraction_multi_abd_cases(
    float& result,
    const std::array<const void*, AsDataType_::size()>& as_ptr,
    const std::array<const void*, BsDataType_::size()>& bs_ptr,
    const std::array<const void*, DsDataType_::size()>& ds_ptr,
    void* e_ptr,
    const std::array<std::vector<ck_tile::index_t>, AsDataType_::size()>& As_dims,
    const std::array<std::vector<ck_tile::index_t>, BsDataType_::size()>& Bs_dims,
    const std::array<std::vector<ck_tile::index_t>, DsDataType_::size()>& Ds_dims,
    const std::vector<ck_tile::index_t>& E_dims,
    const std::array<std::vector<ck_tile::index_t>, AsDataType_::size()>& As_strides,
    const std::array<std::vector<ck_tile::index_t>, BsDataType_::size()>& Bs_strides,
    const std::array<std::vector<ck_tile::index_t>, DsDataType_::size()>& Ds_strides,
    const std::vector<ck_tile::index_t>& E_strides,
    const ck_tile::stream_config& s,
    ck_tile::index_t num_g_dims,
    ck_tile::index_t num_m_dims,
    ck_tile::index_t num_n_dims,
    ck_tile::index_t num_k_dims,
    std::index_sequence<CaseIds...>)
{
    return (try_contraction_multi_abd_case<
                std::tuple_element_t<CaseIds, SupportedContractionMultiABDDimCases>,
                AsDataType_,
                BsDataType_,
                DsDataType_,
                AccDataType_,
                EDataType_,
                AsLayout,
                BsLayout,
                DsLayout,
                ELayout,
                AElementWise,
                BElementWise,
                CDEElementWise>(result,
                                as_ptr,
                                bs_ptr,
                                ds_ptr,
                                e_ptr,
                                As_dims,
                                Bs_dims,
                                Ds_dims,
                                E_dims,
                                As_strides,
                                Bs_strides,
                                Ds_strides,
                                E_strides,
                                s,
                                num_g_dims,
                                num_m_dims,
                                num_n_dims,
                                num_k_dims) ||
            ...);
}

template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename EDataType_,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementWise   = ck_tile::element_wise::PassThrough,
          typename BElementWise   = ck_tile::element_wise::PassThrough,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float contraction_multi_abd(
    const std::array<const void*, AsDataType_::size()>& as_ptr,
    const std::array<const void*, BsDataType_::size()>& bs_ptr,
    const std::array<const void*, DsDataType_::size()>& ds_ptr,
    void* e_ptr,
    const std::array<std::vector<ck_tile::index_t>, AsDataType_::size()>& As_dims,
    const std::array<std::vector<ck_tile::index_t>, BsDataType_::size()>& Bs_dims,
    const std::array<std::vector<ck_tile::index_t>, DsDataType_::size()>& Ds_dims,
    const std::vector<ck_tile::index_t>& E_dims,
    const std::array<std::vector<ck_tile::index_t>, AsDataType_::size()>& As_strides,
    const std::array<std::vector<ck_tile::index_t>, BsDataType_::size()>& Bs_strides,
    const std::array<std::vector<ck_tile::index_t>, DsDataType_::size()>& Ds_strides,
    const std::vector<ck_tile::index_t>& E_strides,
    const ck_tile::stream_config& s,
    ck_tile::index_t num_g_dims,
    ck_tile::index_t num_m_dims,
    ck_tile::index_t num_n_dims,
    ck_tile::index_t num_k_dims)
{
    float result = 0;
    if(dispatch_contraction_multi_abd_cases<AsDataType_,
                                            BsDataType_,
                                            DsDataType_,
                                            AccDataType_,
                                            EDataType_,
                                            AsLayout,
                                            BsLayout,
                                            DsLayout,
                                            ELayout,
                                            AElementWise,
                                            BElementWise,
                                            CDEElementWise>(
           result,
           as_ptr,
           bs_ptr,
           ds_ptr,
           e_ptr,
           As_dims,
           Bs_dims,
           Ds_dims,
           E_dims,
           As_strides,
           Bs_strides,
           Ds_strides,
           E_strides,
           s,
           num_g_dims,
           num_m_dims,
           num_n_dims,
           num_k_dims,
           std::make_index_sequence<std::tuple_size_v<SupportedContractionMultiABDDimCases>>{}))
    {
        return result;
    }

    throw std::runtime_error("Unsupported dimension combination: G=" + std::to_string(num_g_dims) +
                             ", M=" + std::to_string(num_m_dims) + ", N=" +
                             std::to_string(num_n_dims) + ", K=" + std::to_string(num_k_dims) +
                             ". Supported combinations are (G,M,N,K): (1,1,1,1), (1,2,2,2), "
                             "(2,1,1,1), (2,2,2,2).");
}

#include "run_contraction_multi_abd_example.inc"

int main(int argc, char* argv[])
{
    try
    {
        return !run_contraction_multi_abd_example(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
