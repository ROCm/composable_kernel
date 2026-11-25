// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled.hpp"

namespace ck_tile {

/**
 * @brief TdmEpilogue for MxGemmPipelineAgBgCrCompTDMScheduled: E = acc staged in LDS
 *        (kTileM * kTileN * sizeof(E) bytes) and stored by TDM. E is row-major and there are no
 *        Ds tensors. An FP8 E saturates: SaturateFp8E4M3 maps each accumulator before the cast.
 */
template <typename Pipeline_,
          typename Pipeline = remove_cvref_t<Pipeline_>,
          typename Shape    = typename Pipeline::BlockGemmShape>
using MxTdmStoreEpilogue =
    TdmEpilogueStaged<CShuffleEpilogueProblem<typename Pipeline::ADataType,
                                              typename Pipeline::BDataType,
                                              tuple<>,
                                              float,
                                              typename Pipeline::ODataType,
                                              tuple<>,
                                              tensor_layout::gemm::RowMajor,
                                              element_wise::PassThrough,
                                              Shape::kM,
                                              Shape::kN,
                                              Shape::BlockWarps::at(number<0>{}),
                                              Shape::BlockWarps::at(number<1>{}),
                                              Shape::WarpTile::at(number<0>{}),
                                              Shape::WarpTile::at(number<1>{}),
                                              Shape::WarpTile::at(number<2>{}),
                                              /*isCTransposed=*/false>,
                      /*MIterPerStage=*/0,
                      /*NumStageBuffers=*/2,
                      std::conditional_t<std::is_same_v<typename Pipeline::ODataType, fp8_t>,
                                         element_wise::SaturateFp8E4M3,
                                         element_wise::PassThrough>>;

} // namespace ck_tile
