// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/conv/conv_op.hpp"
#include <iostream>
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include <cassert>

namespace ck {
namespace host {
namespace conv {

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

void Operation_Conv::update_prologue(const std::string& prologue)
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

void Operation_Conv::update_epilogue(const std::string& epilogue)
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

template <class F>
std::vector<Operation_Conv> CreateOperationsImpl(
    F f, Layout ALayout, Layout BLayout, const std::string& prologue, const std::string& epilogue)
{
    std::vector<Operation_Conv> result;

    std::vector<operation::TileDesc> tile_descriptions = {
        // clang-format off
//  Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| NumGemmK|
//   Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per| Prefetch|
//       |      |      |      |    |    |     |     | Wave| Wave|    Stage|
//       |      |      |      |    |    |     |     |     |     |         |
  {   256,   256,   128,    32,   8,   2,   32,   32,    4,    2,        1}
  //{   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,        1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> a_block_descriptions_rowmajor = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1}
  //{    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> b_block_descriptions_rowmajor = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |
//                |               |               |              |               |               |          |
  {    S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0}
  //{    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1},
        // clang-format on
    };

    std::vector<operation::CShuffleDesc> cshuffle_descriptions = {
        // clang-format off
//    CShuffle|    CShuffle|
// MXdlPerWave| NXdlPerWave|
//  PerShuffle|  PerShuffle|
//            |            |
  {          1,           1}
  //{          1,           1},
        // clang-format on
    };

    std::vector<operation::CBlockTransferDesc> c_block_descriptions = {
        // clang-format off
// CBlockTransferClusterLengths|  CBlockTransfer
//         _MBlock_MWaveMPerXdl| ScalarPerVector
//         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl
//                             |                
  {              S<1, 32, 1, 8>,               8}
  //{              S<1, 32, 1, 8>,               8},
        // clang-format on
    };

    const auto a_block_descriptions =
        (ALayout == Layout::Row) ? a_block_descriptions_rowmajor : b_block_descriptions_rowmajor;
    const auto b_block_descriptions =
        (BLayout == Layout::Row) ? b_block_descriptions_rowmajor : a_block_descriptions_rowmajor;

    assert(tile_descriptions.size() == a_block_descriptions.size());
    assert(tile_descriptions.size() == b_block_descriptions.size());
    assert(tile_descriptions.size() == cshuffle_descriptions.size());
    assert(tile_descriptions.size() == c_block_descriptions.size());

    std::cout << "starting transfer" << std::endl;
    for(std::size_t i = 0; i < tile_descriptions.size(); i++)
    {
        Operation_Conv x;
        x.NumDim           = 2;
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
    std::cout << "finished loading" << std::endl;
    return result;
}

static Layout ToLayout(bool Trans) { return Trans ? Layout::Column : Layout::Row; }
std::vector<Operation_Conv> Operation_Conv::CreateOperations(const std::string& prologue,
                                                             const std::string& epilogue)
{
    return CreateOperationsImpl([](auto x) -> std::vector<Operation_Conv> { return {x}; },
                                Layout::NHWGC,
                                Layout::GKYXC,
                                prologue,
                                epilogue);
}
std::vector<Operation_Conv> Operation_Conv::CreateOperations(const Problem_Conv& prob,
                                                             const std::string& prologue,
                                                             const std::string& epilogue)
{
    return CreateOperationsImpl(
        [&](Operation_Conv x) -> std::array<Operation_Conv, 1> {
            x.NumDim      = prob.NumDim;
            x.A           = TensorDesc{prob.ADataType, prob.ALayout};
            x.B           = TensorDesc{prob.BDataType, prob.BLayout};
            x.E           = TensorDesc{prob.EDataType, prob.ELayout};
            x.Ds          = Transform(prob.DsTrans, prob.DsDataType, [](auto trans, auto dt) {
                return TensorDesc{dt, ToLayout(trans)};
            });
            x.a_elem_op   = prob.AElementOp;
            x.b_elem_op   = prob.BElementOp;
            x.cde_elem_op = prob.CDEElementOp;
            x.gemm_specialization =
                GetGemmSpec(prob.G, // TODO: check the input going into this is correct
                            prob.N,
                            prob.C,
                            x.tile_desc.m_per_block,
                            x.tile_desc.n_per_block,
                            x.tile_desc.k_per_block);
            return {x};
        },
        prob.ALayout,
        prob.BLayout,
        prologue,
        epilogue);
}

static const char* const Device_ConvTemplate =
    R"(
${Prologue}
${Epilogue}

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<${NumDim}, ${LayoutA}, ${LayoutB}, ${LayoutDs}, ${LayoutE}, ${ADataType}, ${BDataType}, ${AccDataType}, ${CShuffleDataType}, ${DsDataType}, ${EDataType}, ${AElementwiseOperation}, ${BElementwiseOperation}, ${CDEElementwiseOperation}, ${ConvSpecialization}, ${GemmSpecialization}, ${NumGemmkPrefetchStage}, ${BlockSize}, ${MPerBlock}, ${NPerBlock}, ${KPerBlock}, ${AK1}, ${BK1}, ${MPerXDL}, ${NPerXDL}, ${MXdlPerWave}, ${NXdlPerWave}, ${ABlockTransferThreadClusterLengths_AK0_M_AK1}, ${ABlockTransferThreadClusterArrangeOrder}, ${ABlockTransferSrcAccessOrder}, ${ABlockTransferSrcVectorDim}, ${ABlockTransferSrcScalarPerVector}, ${ABlockTransferDstScalarPerVector_AK1}, ${ABlockLdsExtraM}, ${BBlockTransferThreadClusterLengths_BK0_N_BK1}, ${BBlockTransferThreadClusterArrangeOrder}, ${BBlockTransferSrcAccessOrder}, ${BBlockTransferSrcVectorDim}, ${BBlockTransferSrcScalarPerVector}, ${BBlockTransferDstScalarPerVector_BK1}, ${BBlockLdsExtraN}, ${CShuffleMXdlPerWavePerShuffle}, ${CShuffleNXdlPerWavePerShuffle}, ${CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock}, ${CDEBlockTransferScalarPerVector_NPerBlock}>;

extern "C" __global__ void run_${name}(const ${ADataType}* a, const ${BDataType}* b, ${EDataType}* c,  const DeviceOp::Argument& arg)
{
    using CDEElementOp = Prologue;

    constexpr ck::LoopScheduler LoopSched = ck::make_default_loop_scheduler();

    // GridwiseGemm
    using GridwiseGemm = ck::GridwiseGemmMultipleABD_xdl_cshuffle<
        ${ADataType},
        ${BDataType},
        ${ComputeDataType},//double-check this assignment for correctness
        ${AccDataType},
        ${CShuffleDataType},
        ${DsDataType},
        ${EDataType},
        ${AElementwiseOperation},
        ${BElementwiseOperation},
        ${CDEElementwiseOperation},
        ck::InMemoryDataOperationEnum::Set,
        ${NumGemmkPrefetchStage},
        ${BlockSize},
        ${MPerBlock},
        ${NPerBlock},
        ${KPerBlock},
        ${AK1},
        ${BK1},
        ${MPerXDL},
        ${NPerXDL},
        ${MXdlPerWave},
        ${NXdlPerWave},
        ${ABlockTransferThreadClusterLengths_AK0_M_AK1},
        ${ABlockTransferThreadClusterArrangeOrder},
        ${ABlockTransferSrcAccessOrder},
        ${ABlockTransferSrcVectorDim},
        ${ABlockTransferSrcScalarPerVector},
        ${ABlockTransferDstScalarPerVector_AK1},
        false,
        ${ABlockLdsExtraM},
        ${BBlockTransferThreadClusterLengths_BK0_N_BK1},
        ${BBlockTransferThreadClusterArrangeOrder},
        ${BBlockTransferSrcAccessOrder},
        ${BBlockTransferSrcVectorDim},
        ${BBlockTransferSrcScalarPerVector},
        ${BBlockTransferDstScalarPerVector_BK1},
        false,
        ${BBlockLdsExtraN},
        ${CShuffleMXdlPerWavePerShuffle},
        ${CShuffleNXdlPerWavePerShuffle},
        ${CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock},
        ${CDEBlockTransferScalarPerVector_NPerBlock},
        LoopSched>;


    const auto kernel = kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
                    GridwiseGemm,
                    const ${ADataType}*,
                    const ${BDataType}*,
                    typename GridwiseGemm::DsGridPointer,
                    ${EDataType},
                    ${AElementwiseOperation},
                    ${BElementwiseOperation},
                    ${CDEElementwiseOperation},
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    Block2ETileMap,
		    ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, NumDTensor>,
                    bool,
                    bool,
                    bool>;
}
)";

Solution Operation_Conv::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"name",
         std::to_string(this->tile_desc.block_size) + "_" +
             std::to_string(this->tile_desc.m_per_block) + "_" +
             std::to_string(this->tile_desc.n_per_block) + "_" +
             std::to_string(this->tile_desc.k_per_block) + "_" +
             std::to_string(this->tile_desc.ak1)},
        {"NumDim", std::to_string(this->NumDim)},
        {"LayoutA", ToString(this->A.layout)},
        {"LayoutB", ToString(this->B.layout)},
        {"LayoutDs",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.layout); }))},
        {"LayoutE", ToString(this->E.layout)},
        {"ADataType", ToString(this->A.element)},
        {"BDataType", ToString(this->B.element)},
        {"AccDataType", ToString(this->acc)},
        {"ComputeDataType", ToString(this->A.element)},
        {"CShuffleDataType", ToString(this->cs_type)},
        {"DsDataType",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.element); }))},
        {"EDataType", ToString(this->E.element)},
        {"AElementwiseOperation", this->a_elem_op},
        {"BElementwiseOperation", this->b_elem_op},
        {"CDEElementwiseOperation", this->cde_elem_op},
        {"Prologue", this->prologue},
        {"Epilogue", this->epilogue},
        {"ConvSpecialization", this->conv_specialization},
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

    return Solution{InterpolateString(Device_ConvTemplate, values), std::move(values)};
}

} // namespace conv
} // namespace host
} // namespace ck
