// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/device_batched_gemm_multiple_d_gemm_multiple_d/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include <cassert>

namespace ck {
namespace host {
namespace device_batched_gemm_multiple_d_gemm_multiple_d {

// calculate appropriate Gemm Specification based on input tensor dimensions
operation::PaddingDesc GetPaddingDesc(const std::size_t m,
                                      const std::size_t n,
                                      const std::size_t k,
                                      const std::size_t n1,
                                      const std::size_t m_per_block,
                                      const std::size_t n_per_block,
                                      const std::size_t k_per_block,
                                      const std::size_t n1_per_block,
                                      const std::size_t k1_per_block)
{
    operation::PaddingDesc desc;
    if(integer_divide_ceil(m, m_per_block) * m_per_block - m != 0)
        desc.pad_gemm0_m = true;
    if(integer_divide_ceil(n, n_per_block) * n_per_block - n != 0)
        desc.pad_gemm0_n = true;
    if(integer_divide_ceil(k, k_per_block) * k_per_block - k != 0)
        desc.pad_gemm0_k = true;
    if(integer_divide_ceil(n1, n1_per_block) * n1_per_block - n1 != 0)
        desc.pad_gemm1_n = true;
    if(integer_divide_ceil(n, k1_per_block) * k1_per_block - n != 0) // TODO is n == k1 ?
        desc.pad_gemm1_k = true;

    return desc;
}

// function to update prologue/epilogue with user provided operation
void Operation_Xdl_CShuffle::update_prologue(const std::string& pro)
{
    if(!prologue.empty())
    {
        this->prologue = pro;
        // TODO is this right?
        this->cde0_elem_op = "CDE0ElementOp";
        this->cde1_elem_op = "CDE1ElementOp";
    }
    else
    {
        this->prologue = "";
    }
}

void Operation_Xdl_CShuffle::update_epilogue(const std::string& epi)
{
    if(!epilogue.empty())
    {
        this->epilogue = epi;
        // TODO is this right?
        this->cde0_elem_op = "CDE0ElementOp";
        this->cde1_elem_op = "CDE1ElementOp";
    }
    else
    {
        this->epilogue = "";
    }
}

// accounts for all possible combinations of Row/Col major
static Layout ToLayout(bool Trans) { return Trans ? Layout::Column : Layout::Row; }

// Hard-code tuning parameters in modularized fashion, string them together into a vector of
// instances
std::vector<Operation_Xdl_CShuffle> Operation_Xdl_CShuffle::CreateOperations(
    const Problem& prob, const std::string& prologue, const std::string& epilogue)
{
    std::vector<Operation_Xdl_CShuffle> result;

    const auto b1k1 = prob.TransB1 ? 4 : 2;

    std::vector<operation::TileDescGemmGemm> tile_descriptions = {
        // clang-format off
//  Block|  Gemm0| Gemm0| Gemm0| Gemm1| Gemm1|A0K1|B0K1| B1K1| MPer| NPer| Gemm0| Gemm0| Gemm1|NumGemm0K|
//   Size|   MPer|  NPer|  KPer|  NPer|  KPer|    |    |     |  XDL|  XDL|  MXdl|  NXdl|  NXdl| Prefetch|
//       |  Block| Block| Block| Block| Block|    |    |     |     |     |   Per|   Per|   Per|    Stage|
//       |       |      |      |      |      |    |    |     |     |     |  Wave|  Wave|  Wave|         |
//generic
  {   256,    128,    64,    32,   128,    32,   8,   8, b1k1,   32,   32,     1,     2,     4,        1},
// no padding
  {   256,    128,   128,    64,    64,    32,   8,   8, b1k1,   32,   32,     1,     4,     2,        1},
  {   256,    128,   128,    32,    64,    32,   8,   8, b1k1,   32,   32,     1,     4,     2,        1},
  {   256,    128,   128,    64,   128,    32,   8,   8, b1k1,   32,   32,     1,     4,     4,        1},
  {   256,    128,   128,    32,   128,    32,   8,   8, b1k1,   32,   32,     1,     4,     4,        1},
  {   256,     64,   256,    32,   128,    32,   8,   8, b1k1,   16,   16,     1,    16,     8,        1},
  {   256,     64,   256,    32,    64,    32,   8,   8, b1k1,   16,   16,     1,    16,     4,        1},
  {   256,     64,   256,    64,   128,    32,   8,   8, b1k1,   16,   16,     1,    16,     8,        1},
  {   256,     64,   256,    64,    64,    32,   8,   8, b1k1,   16,   16,     1,    16,     4,        1},
// Padded fallback kernel
  {   256,    128,   128,    64,   128,    32,   8,   8, b1k1,   32,   32,     1,     4,     4,        1},
  {   256,    128,    64,    32,   128,    32,   8,   8, b1k1,   32,   32,     1,     2,     4,        1},
        // clang-format on
    };
    if(prob.TransB1)
    {
        // clang-format off
        tile_descriptions.push_back(
  {   256,    256,    128,   32,   128,    32,    8,    8,    4,   32,   32,     2,     4,     4,         1}
        );
        // clang-format on
    }

    std::vector<operation::BlockTransferDesc> a0_block_descriptions = {
        // clang-format off
// A0BlockTransfer|A0BlockTransfer|A0BlockTransfer|A0BlockTransfer|A0BlockTransfer|A0BlockTransfer|A0BlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          |
//                |               |               |               |               |               |          |
//generic
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
// no padding
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
// Padded fallback kernel
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
        // clang-format on
    };
    if(prob.TransB1)
    {
        // clang-format off
        a0_block_descriptions.push_back(
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true}
        );
        // clang-format on
    }

    auto b0_block_descriptions = a0_block_descriptions;
    if(prob.TransB1)
    {
        b0_block_descriptions[1].lds_add_extra_dim = true;
        b0_block_descriptions[3].lds_add_extra_dim = true;
    }

    std::vector<operation::BlockTransferDesc> cde0_block_descriptions = {
        // clang-format off
//       ... | CDE0BlockTransfer| CDE0BlockTransfer| ... |
//       ... |      SrcVectorDim|         SrcScalar| ... |
//       ... |                  |         PerVector| ... |
//           |                  |                  |     |
//generic
  {"", "", "",                 9,                 1, 0, 0},
// no padding
  {"", "", "",                 9,                 4, 0, 0},
  {"", "", "",                 9,                 4, 0, 0},
  {"", "", "",                 9,                 4, 0, 0},
  {"", "", "",                 9,                 4, 0, 0},
  {"", "", "",                 9,                 4, 0, 0},
  {"", "", "",                 9,                 4, 0, 0},
  {"", "", "",                 9,                 4, 0, 0},
  {"", "", "",                 9,                 4, 0, 0},
// Padded fallback kernel
  {"", "", "",                 9,                 4, 0, 0},
  {"", "", "",                 9,                 4, 0, 0},
        // clang-format on
    };
    if(prob.TransB1)
    {
        // clang-format off
        cde0_block_descriptions.push_back(
  {"", "", "",                 9,                 4, 0, 0}
        );
        // clang-format on
    }

    const std::vector<operation::BlockTransferDesc> b1_block_descriptions_rowmajor =
        {
            // clang-format off
//   B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|
//     ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|
//   Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |
//                  |                |                |                |                |                |           |
//generic
  {     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
// no padding
  {     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
  {     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
  {     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
  {     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
  {     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
  {     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
  {     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
  {     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
// Padded fallback kernel
  {     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
  {     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
            // clang-format on
        };

    const std::vector<operation::BlockTransferDesc> b1_block_descriptions_colmajor =
        {
            // clang-format off
//   B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|
//     ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|
//   Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |
//                  |                |                |                |                |                |           |
//generic
  {      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
// no padding
  {      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
  {      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false},
  {      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
  {      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false},
  {      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
  {      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
  {      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
  {      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
  {      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
// Padded fallback kernel
  {      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false},
  {      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,       true},
            // clang-format on
        };

    std::vector<operation::CShuffleDesc> cshuffle_descriptions = {
        // clang-format off
//   C1Shuffle|   C1Shuffle|
// MXdlPerWave| NXdlPerWave|
//  PerShuffle|  PerShuffle|
//            |            |
// generic
  {          1,           2},
// no padding
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           8},
  {          1,           4},
  {          1,           8},
  {          1,           4},
// Padded fallback kernel
  {          1,           2},
  {          1,           2},
        // clang-format on
    };
    if(prob.TransB1)
    {
        // clang-format off
        cshuffle_descriptions.push_back(
  {          1,           2}
        );
        // clang-format on
    }

    std::vector<operation::CBlockTransferDesc> cde1_block_descriptions = {
        // clang-format off
// CDE1BlockTransferClusterLengths| CDE1BlockTransfer|
//            _MBlock_MWaveMPerXdl|   ScalarPerVector|
//            _NBlock_NWaveNPerXdl|     _NWaveNPerXdl|
//                                |                  |
// generic
  {                 S<1, 32, 1, 8>,                 8},
// no padding
  {                 S<1, 32, 1, 8>,                 8},
  {                 S<1, 32, 1, 8>,                 8},
  {                 S<1, 32, 1, 8>,                 8},
  {                 S<1, 32, 1, 8>,                 8},
  {                 S<1, 16, 1,16>,                 8},
  {                 S<1, 32, 1, 8>,                 8},
  {                 S<1, 16, 1,16>,                 8},
  {                 S<1, 32, 1, 8>,                 8},
// Padded fallback kernel
  {                 S<1, 32, 1, 8>,                 8},
  {                 S<1, 32, 1, 8>,                 8},
        // clang-format on
    };
    if(prob.TransB1)
    {
        // clang-format off
        cde1_block_descriptions.push_back(
  {                 S<1, 32, 1, 8>,                 8}
        );
        // clang-format on
    }

    // choose correct arrangement of tuning parameters based on the layout of each tensor
    const auto& b1_block_descriptions =
        prob.TransB1 ? b1_block_descriptions_colmajor : b1_block_descriptions_rowmajor;

    assert(tile_descriptions.size() == a0_block_descriptions.size());
    assert(tile_descriptions.size() == b0_block_descriptions.size());
    assert(tile_descriptions.size() == cde0_block_descriptions.size());
    assert(tile_descriptions.size() == b1_block_descriptions.size());
    assert(tile_descriptions.size() == cshuffle_descriptions.size());
    assert(tile_descriptions.size() == cde1_block_descriptions.size());

    // Put all values together into a single operation > store into the result vector
    for(std::size_t i = 0; i < tile_descriptions.size(); i++)
    {
        Operation_Xdl_CShuffle x;
        x.tile_desc           = tile_descriptions[i];
        x.a0_block_transfer   = a0_block_descriptions[i];
        x.b0_block_transfer   = b0_block_descriptions[i];
        x.cde0_block_transfer = cde0_block_descriptions[i];
        x.b1_block_transfer   = b1_block_descriptions[i];
        x.cshuffle            = cshuffle_descriptions[i];
        x.cde1_block_transfer = cde1_block_descriptions[i];
        x.A0                  = TensorDesc{prob.A0DataType, ToLayout(prob.TransA0)};
        x.B0                  = TensorDesc{prob.B0DataType, ToLayout(prob.TransB0)};
        x.D0s                 = Transform(prob.D0sTrans, prob.D0sDataType, [](auto trans, auto dt) {
            return TensorDesc{dt, ToLayout(trans)};
        });
        x.B1                  = TensorDesc{prob.B1DataType, ToLayout(prob.TransB1)};
        x.D1s                 = Transform(prob.D1sTrans, prob.D1sDataType, [](auto trans, auto dt) {
            return TensorDesc{dt, ToLayout(trans)};
        });
        x.E1                  = TensorDesc{prob.E1DataType, ToLayout(prob.TransE1)};
        x.a0_elem_op          = prob.A0ElementOp;
        x.b0_elem_op          = prob.B0ElementOp;
        x.cde0_elem_op        = prob.CDE0ElementOp;
        x.b1_elem_op          = prob.B1ElementOp;
        x.cde1_elem_op        = prob.CDE1ElementOp;
        x.padding_desc        = GetPaddingDesc(prob.M,
                                        prob.N,
                                        prob.K,
                                        prob.O,
                                        x.tile_desc.gemm0_m_per_block,
                                        x.tile_desc.gemm0_n_per_block,
                                        x.tile_desc.gemm0_k_per_block,
                                        x.tile_desc.gemm1_n_per_block,
                                        x.tile_desc.gemm1_k_per_block);
        x.update_prologue(prologue);
        x.update_epilogue(epilogue);
        result.push_back(x);
    }
    return result;
}

// set up instances when not provided with a problem specification, use default operation values and
// all possible layout combinations
std::vector<std::vector<Operation_Xdl_CShuffle>>
Operation_Xdl_CShuffle::CreateOperations(const std::string& prologue, const std::string& epilogue)
{
    std::vector<std::vector<Operation_Xdl_CShuffle>> operations;

    Problem prob;
    prob.TransB0 = true;
    operations.push_back(CreateOperations(prob, prologue, epilogue));

    prob.TransB1 = true;
    operations.push_back(CreateOperations(prob, prologue, epilogue));

    return operations;
}

static const char* const DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffleTemplate =
    "ck::tensor_operation::device::DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle<"
    "${A0Layout}, ${B0Layout}, ${D0sLayout}, ${B1Layout}, ${D1sLayout}, ${E1Layout}, "

    "${A0DataType}, ${B0DataType}, ${Acc0DataType}, ${D0sDataType}, ${B1DataType}, "
    "${Acc1DataType}, ${C1ShuffleDataType}, ${D1sDataType}, ${E1DataType}, "

    "${A0ElementwiseOperation}, ${B0ElementwiseOperation}, ${CDE0ElementwiseOperation}, "
    "${B1ElementwiseOperation}, ${CDE1ElementwiseOperation}, "

    "${PadGemm0M}, ${PadGemm0N}, ${PadGemm0K}, ${PadGemm1N}, ${PadGemm1K}, "

    "${NumGemm0KPrefetchStage}, ${BlockSize}, ${Gemm0MPerBlock}, ${Gemm0NPerBlock}, "
    "${Gemm0KPerBlock}, ${Gemm1NPerBlock}, ${Gemm1KPerBlock}, ${A0K1}, ${B0K1}, ${B1K1}, "
    "${MPerXDL}, ${NPerXDL}, ${Gemm0MXdlPerWave}, ${Gemm0NXdlPerWave}, ${Gemm1NXdlPerWave}, "

    "${A0BlockTransferThreadClusterLengths_AK0_M_AK1}, "
    "${A0BlockTransferThreadClusterArrangeOrder}, ${A0BlockTransferSrcAccessOrder}, "
    "${A0BlockTransferSrcVectorDim}, ${A0BlockTransferSrcScalarPerVector}, "
    "${A0BlockTransferDstScalarPerVector_AK1}, ${A0BlockLdsExtraM}, "

    "${B0BlockTransferThreadClusterLengths_BK0_N_BK1}, "
    "${B0BlockTransferThreadClusterArrangeOrder}, ${B0BlockTransferSrcAccessOrder}, "
    "${B0BlockTransferSrcVectorDim}, ${B0BlockTransferSrcScalarPerVector}, "
    "${B0BlockTransferDstScalarPerVector_BK1}, ${B0BlockLdsExtraN}, "

    "${CDE0BlockTransferSrcVectorDim}, ${CDE0BlockTransferSrcScalarPerVector}, "

    "${B1BlockTransferThreadClusterLengths_BK0_N_BK1}, "
    "${B1BlockTransferThreadClusterArrangeOrder}, ${B1BlockTransferSrcAccessOrder}, "
    "${B1BlockTransferSrcVectorDim}, ${B1BlockTransferSrcScalarPerVector}, "
    "${B1BlockTransferDstScalarPerVector_BK1}, ${B1BlockLdsExtraN}, "

    "${C1ShuffleMXdlPerWavePerShuffle}, ${C1ShuffleGemm0NXdlPerWavePerShuffle}, "

    "${CDE1ShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock}, "
    "${CDE1ShuffleBlockTransferScalarPerVector_NPerBlock}>";

// use hardcoded instances from vector of operations to substitute values into instance template
Solution Operation_Xdl_CShuffle::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"name",
         std::to_string(this->tile_desc.block_size) + "_" +
             std::to_string(this->tile_desc.gemm0_m_per_block) + "_" +
             std::to_string(this->tile_desc.gemm0_n_per_block) + "_" +
             std::to_string(this->tile_desc.gemm0_k_per_block) + "_" +
             std::to_string(this->tile_desc.gemm1_n_per_block) + "_" +
             std::to_string(this->tile_desc.gemm1_k_per_block) + "_" +
             std::to_string(this->tile_desc.a0k1) + "_" + std::to_string(this->tile_desc.b0k1) +
             "_" + std::to_string(this->tile_desc.b1k1) + "_" +
             std::to_string(this->tile_desc.m_per_XDL) + "_" +
             std::to_string(this->tile_desc.n_per_XDL) + "_" +
             std::to_string(this->tile_desc.gemm0_m_Xdl_per_wave) + "_" +
             std::to_string(this->tile_desc.gemm0_n_Xdl_per_wave) + "_" +
             std::to_string(this->tile_desc.gemm1_n_Xdl_per_wave)},

        {"A0Layout", ToString(this->A0.layout)},
        {"B0Layout", ToString(this->B0.layout)},
        {"D0sLayout",
         MakeTuple(Transform(this->D0s, [](auto tensor) { return ToString(tensor.layout); }))},
        {"B1Layout", ToString(this->B1.layout)},
        {"D1sLayout",
         MakeTuple(Transform(this->D1s, [](auto tensor) { return ToString(tensor.layout); }))},
        {"E1Layout", ToString(this->E1.layout)},

        {"ADataType", ToString(this->A0.element)},
        {"B0DataType", ToString(this->B0.element)},
        {"Acc0DataType", ToString(this->acc_type)},
        {"D0sDataType",
         MakeTuple(Transform(this->D0s, [](auto tensor) { return ToString(tensor.element); }))},
        {"B1DataType", ToString(this->B1.element)},
        {"Acc1DataType", ToString(this->acc_type)},
        {"C1ShuffleDataType", ToString(this->cshuffle_type)},
        {"D1sDataType",
         MakeTuple(Transform(this->D1s, [](auto tensor) { return ToString(tensor.element); }))},
        {"E1DataType", ToString(this->E1.element)},

        {"A0ElementwiseOperation", this->a0_elem_op},
        {"B0ElementwiseOperation", this->b0_elem_op},
        {"CDE0ElementwiseOperation", this->cde0_elem_op},
        {"B1ElementwiseOperation", this->b1_elem_op},
        {"CDE1ElementwiseOperation", this->cde1_elem_op},

        {"PadGemm0M", std::to_string(this->padding_desc.pad_gemm0_m)},
        {"PadGemm0N", std::to_string(this->padding_desc.pad_gemm0_n)},
        {"PadGemm0K", std::to_string(this->padding_desc.pad_gemm0_k)},
        {"PadGemm1N", std::to_string(this->padding_desc.pad_gemm1_n)},
        {"PadGemm1K", std::to_string(this->padding_desc.pad_gemm1_k)},

        {"NumGemm0KPrefetchStage", std::to_string(this->tile_desc.num_gemm0k_prefetch_stage)},
        {"BlockSize", std::to_string(this->tile_desc.block_size)},
        {"Gemm0MPerBlock", std::to_string(this->tile_desc.gemm0_m_per_block)},
        {"Gemm0NPerBlock", std::to_string(this->tile_desc.gemm0_n_per_block)},
        {"Gemm0KPerBlock", std::to_string(this->tile_desc.gemm0_k_per_block)},
        {"Gemm1NPerBlock", std::to_string(this->tile_desc.gemm1_n_per_block)},
        {"Gemm1KPerBlock", std::to_string(this->tile_desc.gemm1_k_per_block)},
        {"A0K1", std::to_string(this->tile_desc.a0k1)},
        {"B0K1", std::to_string(this->tile_desc.b0k1)},
        {"B1K1", std::to_string(this->tile_desc.b1k1)},
        {"MPerXDL", std::to_string(this->tile_desc.m_per_XDL)},
        {"NPerXDL", std::to_string(this->tile_desc.n_per_XDL)},
        {"Gemm0MXdlPerWave", std::to_string(this->tile_desc.gemm0_m_Xdl_per_wave)},
        {"Gemm0NXdlPerWave", std::to_string(this->tile_desc.gemm0_n_Xdl_per_wave)},
        {"Gemm1NXdlPerWave", std::to_string(this->tile_desc.gemm1_n_Xdl_per_wave)},

        {"A0BlockTransferThreadClusterLengths_AK0_M_AK1",
         this->a0_block_transfer.thread_cluster_length},
        {"A0BlockTransferThreadClusterArrangeOrder",
         this->a0_block_transfer.thread_cluster_arrange_order},
        {"A0BlockTransferSrcAccessOrder", this->a0_block_transfer.src_access_order},
        {"A0BlockTransferSrcVectorDim", std::to_string(this->a0_block_transfer.src_vec_dim)},
        {"A0BlockTransferSrcScalarPerVector",
         std::to_string(this->a0_block_transfer.src_scalar_per_vector)},
        {"A0BlockTransferDstScalarPerVector_AK1",
         std::to_string(this->a0_block_transfer.dst_scalar_per_vector_k1)},
        {"A0BlockLdsExtraM", std::to_string(this->a0_block_transfer.lds_add_extra_dim)},

        {"B0BlockTransferThreadClusterLengths_BK0_N_BK1",
         this->b0_block_transfer.thread_cluster_length},
        {"B0BlockTransferThreadClusterArrangeOrder",
         this->b0_block_transfer.thread_cluster_arrange_order},
        {"B0BlockTransferSrcAccessOrder", this->b0_block_transfer.src_access_order},
        {"B0BlockTransferSrcVectorDim", std::to_string(this->b0_block_transfer.src_vec_dim)},
        {"B0BlockTransferSrcScalarPerVector",
         std::to_string(this->b0_block_transfer.src_scalar_per_vector)},
        {"B0BlockTransferDstScalarPerVector_BK1",
         std::to_string(this->b0_block_transfer.dst_scalar_per_vector_k1)},
        {"B0BlockLdsExtraN", std::to_string(this->b0_block_transfer.lds_add_extra_dim)},

        {"CDE0BlockTransferSrcVectorDim", std::to_string(this->cde0_block_transfer.src_vec_dim)},
        {"CDE0BlockTransferSrcScalarPerVector",
         std::to_string(this->cde0_block_transfer.src_scalar_per_vector)},

        {"B1BlockTransferThreadClusterLengths_BK0_N_BK1",
         this->b1_block_transfer.thread_cluster_length},
        {"B1BlockTransferThreadClusterArrangeOrder",
         this->b1_block_transfer.thread_cluster_arrange_order},
        {"B1BlockTransferSrcAccessOrder", this->b1_block_transfer.src_access_order},
        {"B1BlockTransferSrcVectorDim", std::to_string(this->b1_block_transfer.src_vec_dim)},
        {"B1BlockTransferSrcScalarPerVector",
         std::to_string(this->b1_block_transfer.src_scalar_per_vector)},
        {"B1BlockTransferDstScalarPerVector_BK1",
         std::to_string(this->b1_block_transfer.dst_scalar_per_vector_k1)},
        {"B1BlockLdsExtraN", std::to_string(this->b1_block_transfer.lds_add_extra_dim)},

        {"C1ShuffleMXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.m_Xdl_per_wave_per_shuffle)},
        {"C1ShuffleGemm0NXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.n_Xdl_per_wave_per_shuffle)},

        {"CDE1ShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock",
         this->cde1_block_transfer
             .cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl},
        {"CDE1ShuffleBlockTransferScalarPerVector_NPerBlock",
         std::to_string(this->cde1_block_transfer.scalar_per_vector_n_wave_n_per_Xdl)},
    };

    return Solution{
        InterpolateString(DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffleTemplate, values),
        std::move(values)};
}

} // namespace device_batched_gemm_multiple_d_gemm_multiple_d
} // namespace host
} // namespace ck
