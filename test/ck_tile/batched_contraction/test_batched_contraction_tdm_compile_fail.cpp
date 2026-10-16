// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Negative compile checks for the TDM batched contraction path. Exactly one of the
// CK_TDM_CONTRACTION_FAIL_* macros is defined per build target; each must be rejected at
// compile time by a static_assert. The CMake targets are EXCLUDE_FROM_ALL and are built by a
// ctest entry that expects the build to fail with the matching diagnostic.

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using DataType    = ck_tile::half_t;
using AccDataType = float;

#if defined(CK_TDM_CONTRACTION_FAIL_NUM_D)
using DsDataType                = ck_tile::tuple<DataType>;
using DsLayout                  = ck_tile::tuple<Row>;
constexpr ck_tile::index_t NumD = 1;
constexpr bool kPadN            = false;
#elif defined(CK_TDM_CONTRACTION_FAIL_PAD_N)
using DsDataType                = ck_tile::tuple<>;
using DsLayout                  = ck_tile::tuple<>;
constexpr ck_tile::index_t NumD = 0;
constexpr bool kPadN            = true;
#elif defined(CK_TDM_CONTRACTION_FAIL_MULTI_ABD) || \
    defined(CK_TDM_CONTRACTION_FAIL_CSHUFFLE_EPILOGUE)
using DsDataType                = ck_tile::tuple<>;
using DsLayout                  = ck_tile::tuple<>;
constexpr ck_tile::index_t NumD = 0;
constexpr bool kPadN            = false;
#else
#error "Define one of CK_TDM_CONTRACTION_FAIL_{NUM_D,PAD_N,MULTI_ABD,CSHUFFLE_EPILOGUE}"
#endif

constexpr ck_tile::index_t M_Warp_Tile = 16;
constexpr ck_tile::index_t N_Warp_Tile = 16;
constexpr ck_tile::index_t K_Warp_Tile = 32;

using GemmShape       = ck_tile::TileGemmShape<ck_tile::sequence<64, 64, 32>,
                                               ck_tile::sequence<2, 2, 1>,
                                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;
using Traits = ck_tile::TileGemmUniversalTraits<false, kPadN, false, true, Row, Col, Row, false>;
using GemmProblem =
    ck_tile::UniversalGemmPipelineProblem<DataType,
                                          DataType,
                                          AccDataType,
                                          GemmShape,
                                          Traits,
                                          ck_tile::GemmPipelineScheduler::Intrawave>;
using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompTDMV1<GemmProblem>;

using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<DataType,
                                                         DataType,
                                                         DsDataType,
                                                         AccDataType,
                                                         DataType,
                                                         DsLayout,
                                                         Row,
                                                         ck_tile::element_wise::PassThrough,
                                                         TilePartitioner::MPerBlock,
                                                         TilePartitioner::NPerBlock,
                                                         2,
                                                         2,
                                                         M_Warp_Tile,
                                                         N_Warp_Tile,
                                                         K_Warp_Tile,
                                                         false,
                                                         1,
                                                         false,
                                                         1,
                                                         1,
                                                         true>;
#if defined(CK_TDM_CONTRACTION_FAIL_CSHUFFLE_EPILOGUE)
// A TDM pipeline paired with a non-TDM epilogue.
using Epilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
#else
using Epilogue = ck_tile::TdmEpilogue<EpilogueProblem>;
#endif

#if defined(CK_TDM_CONTRACTION_FAIL_MULTI_ABD)
using Problem = ck_tile::BatchedContractionMultiABDProblem<ck_tile::tuple<DataType, DataType>,
                                                           ck_tile::tuple<DataType>,
                                                           DsDataType,
                                                           DataType,
                                                           1,
                                                           1,
                                                           1,
                                                           1>;
using Kernel =
    ck_tile::BatchedContractionMultiABDKernel<Problem, TilePartitioner, GemmPipeline, Epilogue>;
#else
using Problem =
    ck_tile::BatchedContractionProblem<DataType, DataType, DsDataType, DataType, 1, 1, 1, 1, NumD>;
using Kernel = ck_tile::BatchedContractionKernel<Problem, TilePartitioner, GemmPipeline, Epilogue>;
#endif

} // namespace

#if defined(CK_TDM_CONTRACTION_FAIL_MULTI_ABD)
// The multi-ABD static_assert sits at class scope: instantiating the class definition is enough,
// and nothing else in this TU can fail to compile.
[[maybe_unused]] constexpr std::size_t kMultiAbdKernelSize = sizeof(Kernel);
#else
// Forces instantiation of the kernel entry (and therefore of the device code path).
void instantiate_tdm_contraction_kernel(const typename Kernel::KernelArgs& kargs)
{
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                           ck_tile::make_kernel<1>(Kernel{}, dim3(1), dim3(128), 0, kargs));
}
#endif
