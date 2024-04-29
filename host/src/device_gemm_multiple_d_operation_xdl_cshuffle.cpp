// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

// calculate appropriate Gemm Specification based on input tensor dimensions
static std::string GetGemmSpec(const std::size_t m,
                               const std::size_t n,
                               const std::size_t k,
                               const std::size_t m_per_block,
                               const std::size_t n_per_block,
                               const std::size_t k_per_block)
{
    std::string spec = "";
    if(integer_divide_ceil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if(integer_divide_ceil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if(integer_divide_ceil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";
    if(spec == "")
        return "ck::tensor_operation::device::GemmSpecialization::Default";

    return "ck::tensor_operation::device::GemmSpecialization::" + spec + "Padding";
}

// function to update prologue/epilogue with user provided operation
void Operation_Xdl_CShuffle::update_prologue(const std::string& prologue)
{
    if(!prologue.empty())
    {
        this->prologue    = prologue;
        this->cde_elem_op = "CDEElementOp";
    }
    else
    {
        this->prologue = "";
    }
}

void Operation_Xdl_CShuffle::update_epilogue(const std::string& epilogue)
{
    if(!epilogue.empty())
    {
        this->epilogue    = epilogue;
        this->cde_elem_op = "CDEElementOp";
    }
    else
    {
        this->epilogue = "";
    }
}

// Hard-code tuning parameters in modularized fashion, string them together into a vector of
// instances
template <class F>
std::vector<Operation_Xdl_CShuffle> CreateOperationsImpl(
    F f, Layout ALayout, Layout BLayout, const std::string& prologue, const std::string& epilogue)
{
    std::vector<Operation_Xdl_CShuffle> result;

    std::vector<operation::TileDesc> tile_descriptions = {
        // clang-format off
//  Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| NumGemmK|
//   Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per| Prefetch|
//       |      |      |      |    |    |     |     | Wave| Wave|    Stage|
//       |      |      |      |    |    |     |     |     |     |         |
  {   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,        1},
  {   256,   128,   256,    32,   8,   8,   32,   32,    2,    4,        1},
  {   128,   128,   128,    32,   8,   8,   32,   32,    4,    2,        1},
  {   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,        1},
  {   128,   128,    64,    32,   8,   8,   32,   32,    2,    2,        1},
  {   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,        1},
  {   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,        1},
  {   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,        1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> a_block_descriptions_rowmajor = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> a_block_descriptions_colmajor = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
        // clang-format on
        {S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1},
        {S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1},
        {S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1},
        {S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1},
        {S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1},
        {S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1},
        {S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1},
        {S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 1},
    };

    std::vector<operation::BlockTransferDesc> b_block_descriptions_rowmajor = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> b_block_descriptions_colmajor = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
        // clang-format on
    };

    std::vector<operation::CShuffleDesc> cshuffle_descriptions = {
        // clang-format off
//    CShuffle|    CShuffle|
// MXdlPerWave| NXdlPerWave|
//  PerShuffle|  PerShuffle|
//            |            |
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1},
        // clang-format on
    };

    std::vector<operation::CBlockTransferDesc> c_block_descriptions = {
        // clang-format off
// CBlockTransferClusterLengths|  CBlockTransfer
//         _MBlock_MWaveMPerXdl| ScalarPerVector
//         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl
//                             |                
  {              S<1, 32, 1, 8>,               8},
  {              S<1, 32, 1, 8>,               8},
  {              S<1, 16, 1, 8>,               8},
  {              S<1, 32, 1, 8>,               8},
  {              S<1, 32, 1, 4>,               8},
  {              S<1, 16, 1, 8>,               8},
  {              S<1, 32, 1, 8>,               8},
  {              S<1, 32, 1, 8>,               8},
        // clang-format on
    };

    const auto a_block_descriptions =
        (ALayout == Layout::Row) ? a_block_descriptions_rowmajor : a_block_descriptions_colmajor;
    const auto b_block_descriptions =
        (BLayout == Layout::Row) ? b_block_descriptions_rowmajor : b_block_descriptions_colmajor;

    assert(tile_descriptions.size() == a_block_descriptions.size());
    assert(tile_descriptions.size() == b_block_descriptions.size());
    assert(tile_descriptions.size() == cshuffle_descriptions.size());
    assert(tile_descriptions.size() == c_block_descriptions.size());

    // Put all values together into a single operation > store into the result vector
    for(std::size_t i = 0; i < tile_descriptions.size(); i++)
    {
        Operation_Xdl_CShuffle x;
        x.tile_desc        = tile_descriptions[i];
        x.a_block_transfer = a_block_descriptions[i];
        x.b_block_transfer = b_block_descriptions[i];
        x.cshuffle         = cshuffle_descriptions[i];
        x.c_block_transfer = c_block_descriptions[i];
        x.update_prologue(prologue);
        x.update_epilogue(epilogue);
        auto all = f(x);
        result.insert(result.end(), all.begin(), all.end());
    }

    return result;
}

// set layout to row/col based on whether trans is false/true
static Layout ToLayout(bool Trans) { return Trans ? Layout::Column : Layout::Row; }

// set up instances when not provided with a problem specification, use default operation values
std::vector<Operation_Xdl_CShuffle>
Operation_Xdl_CShuffle::CreateOperations(const std::string& prologue, const std::string& epilogue)
{
    return CreateOperationsImpl([](auto x) -> std::vector<Operation_Xdl_CShuffle> { return {x}; },
                                Layout::Column,
                                Layout::Row,
                                prologue,
                                epilogue);
}

// set up instances when given problem specifications
std::vector<Operation_Xdl_CShuffle> Operation_Xdl_CShuffle::CreateOperations(
    const Problem& prob, const std::string& prologue, const std::string& epilogue)
{
    return CreateOperationsImpl(
        [&](Operation_Xdl_CShuffle x) -> std::array<Operation_Xdl_CShuffle, 1> {
            x.A           = TensorDesc{prob.ADataType, ToLayout(prob.TransA)};
            x.B           = TensorDesc{prob.BDataType, ToLayout(prob.TransB)};
            x.E           = TensorDesc{prob.EDataType, ToLayout(prob.TransE)};
            x.Ds          = Transform(prob.DsTrans, prob.DsDataType, [](auto trans, auto dt) {
                return TensorDesc{dt, ToLayout(trans)};
            });
            x.a_elem_op   = prob.AElementOp;
            x.b_elem_op   = prob.BElementOp;
            x.cde_elem_op = prob.CDEElementOp;
            x.gemm_specialization = GetGemmSpec(prob.M,
                                                prob.N,
                                                prob.K,
                                                x.tile_desc.m_per_block,
                                                x.tile_desc.n_per_block,
                                                x.tile_desc.k_per_block);
            return {x};
        },
        ToLayout(prob.TransA),
        ToLayout(prob.TransB),
        prologue,
        epilogue);
}

static const char* const DeviceGemmMultipleD_Xdl_CShuffleTemplate = R"(

ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<${LayoutA}, ${LayoutB}, ${LayoutDs}, ${LayoutE}, ${ADataType}, ${BDataType}, ${AccDataType}, ${CShuffleDataType}, ${DsDataType}, ${EDataType}, ${AElementwiseOperation}, ${BElementwiseOperation}, ${CDEElementwiseOperation}, ${GemmSpecialization}, ${NumGemmkPrefetchStage}, ${BlockSize}, ${MPerBlock}, ${NPerBlock}, ${KPerBlock}, ${AK1}, ${BK1}, ${MPerXDL}, ${NPerXDL}, ${MXdlPerWave}, ${NXdlPerWave}, ${ABlockTransferThreadClusterLengths_AK0_M_AK1}, ${ABlockTransferThreadClusterArrangeOrder}, ${ABlockTransferSrcAccessOrder}, ${ABlockTransferSrcVectorDim}, ${ABlockTransferSrcScalarPerVector}, ${ABlockTransferDstScalarPerVector_AK1}, ${ABlockLdsExtraM}, ${BBlockTransferThreadClusterLengths_BK0_N_BK1}, ${BBlockTransferThreadClusterArrangeOrder}, ${BBlockTransferSrcAccessOrder}, ${BBlockTransferSrcVectorDim}, ${BBlockTransferSrcScalarPerVector}, ${BBlockTransferDstScalarPerVector_BK1}, ${BBlockLdsExtraN}, ${CShuffleMXdlPerWavePerShuffle}, ${CShuffleNXdlPerWavePerShuffle}, ${CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock}, ${CDEBlockTransferScalarPerVector_NPerBlock}>
)";

// use hardcoded instances to substitute values into instance template
Solution Operation_Xdl_CShuffle::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"name",
         std::to_string(this->tile_desc.block_size) + "_" +
             std::to_string(this->tile_desc.m_per_block) + "_" +
             std::to_string(this->tile_desc.n_per_block) + "_" +
             std::to_string(this->tile_desc.k_per_block) + "_" +
             std::to_string(this->tile_desc.ak1) + "_" + std::to_string(this->tile_desc.bk1) + "_" +
             std::to_string(this->tile_desc.m_per_XDL) + "_" +
             std::to_string(this->tile_desc.n_per_XDL) + "_" +
             std::to_string(this->tile_desc.m_Xdl_per_wave) + "_" +
             std::to_string(this->tile_desc.n_Xdl_per_wave)},
        {"LayoutA", ToString(this->A.layout)},
        {"LayoutB", ToString(this->B.layout)},
        {"LayoutDs",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.layout); }))},
        {"LayoutE", ToString(this->E.layout)},
        {"ADataType", ToString(this->A.element)},
        {"BDataType", ToString(this->B.element)},
        {"ComputeDataType", ToString(this->E.element)},
        {"AccDataType", ToString(this->acc)},
        {"CShuffleDataType", ToString(this->cs_type)},
        {"DsDataType",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.element); }))},
        {"EDataType", ToString(this->E.element)},
        {"AElementwiseOperation", this->a_elem_op},
        {"BElementwiseOperation", this->b_elem_op},
        {"CDEElementwiseOperation", this->cde_elem_op},
        {"Prologue", this->prologue},
        {"Epilogue", this->epilogue},
        {"GemmSpecialization", this->gemm_specialization},
        {"NumGemmkPrefetchStage", std::to_string(this->tile_desc.num_gemmk_prefetch_stage)},
        {"BlockSize", std::to_string(this->tile_desc.block_size)},
        {"MPerBlock", std::to_string(this->tile_desc.m_per_block)},
        {"NPerBlock", std::to_string(this->tile_desc.n_per_block)},
        {"KPerBlock", std::to_string(this->tile_desc.k_per_block)},
        {"AK1", std::to_string(this->tile_desc.ak1)},
        {"BK1", std::to_string(this->tile_desc.bk1)},
        {"MPerXDL", std::to_string(this->tile_desc.m_per_XDL)},
        {"NPerXDL", std::to_string(this->tile_desc.n_per_XDL)},
        {"MXdlPerWave", std::to_string(this->tile_desc.m_Xdl_per_wave)},
        {"NXdlPerWave", std::to_string(this->tile_desc.n_Xdl_per_wave)},
        {"ABlockTransferThreadClusterLengths_AK0_M_AK1",
         this->a_block_transfer.thread_cluster_length},
        {"ABlockTransferThreadClusterArrangeOrder",
         this->a_block_transfer.thread_cluster_arrange_order},
        {"ABlockTransferSrcAccessOrder", this->a_block_transfer.src_access_order},
        {"ABlockTransferSrcVectorDim", std::to_string(this->a_block_transfer.src_vec_dim)},
        {"ABlockTransferSrcScalarPerVector",
         std::to_string(this->a_block_transfer.src_scalar_per_vector)},
        {"ABlockTransferDstScalarPerVector_AK1",
         std::to_string(this->a_block_transfer.dst_scalar_per_vector_k1)},
        {"ABlockLdsExtraM", std::to_string(this->a_block_transfer.lds_add_extra_dim)},
        {"BBlockTransferThreadClusterLengths_BK0_N_BK1",
         this->b_block_transfer.thread_cluster_length},
        {"BBlockTransferThreadClusterArrangeOrder",
         this->b_block_transfer.thread_cluster_arrange_order},
        {"BBlockTransferSrcAccessOrder", this->b_block_transfer.src_access_order},
        {"BBlockTransferSrcVectorDim", std::to_string(this->b_block_transfer.src_vec_dim)},
        {"BBlockTransferSrcScalarPerVector",
         std::to_string(this->b_block_transfer.src_scalar_per_vector)},
        {"BBlockTransferDstScalarPerVector_BK1",
         std::to_string(this->b_block_transfer.dst_scalar_per_vector_k1)},
        {"BBlockLdsExtraN", std::to_string(this->b_block_transfer.lds_add_extra_dim)},
        {"CShuffleMXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.m_Xdl_per_wave_per_shuffle)},
        {"CShuffleNXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.n_Xdl_per_wave_per_shuffle)},
        {"CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock",
         this->c_block_transfer.cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl},
        {"CDEBlockTransferScalarPerVector_NPerBlock",
         std::to_string(this->c_block_transfer.scalar_per_vector_n_wave_n_per_Xdl)},
    };

    return Solution{InterpolateString(DeviceGemmMultipleD_Xdl_CShuffleTemplate, values),
                    std::move(values)};
}

/**constexpr**/ bool
Operation_Xdl_CShuffle::IsSupported(std::size_t MRaw_, std::size_t NRaw_, std::size_t KRaw_)
{
    // check vector load/store
    std::string Row = "ck::tensor_layout::gemm::RowMajor";
    std::string Col = "ck::tensor_layout::gemm::ColumnMajor";
    // FIXME: use strings for comparison, can't use CK terms
    // check vector load of A
    if(ToString(this->A.layout) == Row && this->a_block_transfer.src_vec_dim == 2)
    {
        if(KRaw_ % this->a_block_transfer.src_scalar_per_vector != 0)
        {
            return false;
        }
    }
    else if(ToString(this->A.layout) == Col && this->a_block_transfer.src_vec_dim == 1)
    {
        // FIXME: not rigorous
        if(MRaw_ % this->a_block_transfer.src_scalar_per_vector != 0)
        {
            return false;
        }
    }
    else
    {
        return false;
    }
    // check vector laod of B
    if(ToString(this->B.layout) == Col && this->b_block_transfer.src_vec_dim == 2)
    {
        if(KRaw_ % this->b_block_transfer.src_scalar_per_vector != 0)
        {
            return false;
        }
    }
    else if(ToString(this->B.layout) == Row && this->b_block_transfer.src_vec_dim == 1)
    {
        // FIXME: not rigorous
        if(NRaw_ % this->b_block_transfer.src_scalar_per_vector != 0)
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    // check vector load of Ds
    // only support RowMajor for now
    /** bool all_valid = true;

     static_for<0, NumDTensor, 1>{}([&](auto i) {
         using DLayout = remove_cvref_t<tuple_element_t<i.value, this->Ds.layout>>;
         //MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.layout); }))
         if constexpr(!std::is_same_v<DLayout, Row>)
         {
             all_valid = false;
         }
     });


     if(!all_valid)
     {
         return false;
     }

     // check vector store of E
     // only support RowMajor for now
     {
         if(NRaw_ % this->c_block_transfer.scalar_per_vector_n_wave_n_per_Xdl != 0)
         {
             return false;
         }
     }
     else { return false; }**/
    return true;
}

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
