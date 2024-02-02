#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "../parse/include/types_fe.hpp"
#include "ck/host/operation/gemm.hpp"

namespace ck {
namespace host {

struct CKGenOp_Xdl_CShuffle
{
    std::size_t M = 0;
    std::size_t N = 0;
    std::size_t K = 0;
    TensorDesc_fe A{};
    TensorDesc_fe B{};
    DataType_fe acc     = DataType_fe::Float;
    DataType_fe cs_type = DataType_fe::Half;
    TensorDesc_fe Ds{};
    TensorDesc_fe E{};

    std::string a_elem_op           = PassThrough;
    std::string b_elem_op           = PassThrough;
    std::string cde_elem_op         = Bilinear;
    std::string gemm_specialization = "ck::tensor_operation::device::GemmSpecialization::Default";
    operation::TileDesc tile_desc{};
    operation::BlockTransferDesc a_block_transfer{};
    operation::BlockTransferDesc b_block_transfer{};
    operation::CShuffleDesc cshuffle{};
    operation::CBlockTransferDesc c_block_transfer{};
};
// functions for ops
// TODO: replace long parameter list with some sort of Arg list
extern "C" {
std::string CKGenSetOp(CKGenOp_Xdl_CShuffle& op,
                       DataType_fe ADataType,
                       DataType_fe BDataType,
                       DataType_fe DsDataType,
                       DataType_fe EDataType,
                       Layout_fe ALayout,
                       Layout_fe BLayout,
                       Layout_fe DsLayout,
                       Layout_fe ELayout,
                       std::size_t M,
                       std::size_t N,
                       std::size_t K); // set up problem size for the operation
// nlohmann::json CKGenGetOpParams();     // get and parse the JSON file of requirements
void CKGenSetOpFusion(
    std::string Prologue); // override the prologue/epilogue TODO: fix the prologue type
char* CKGenGetBuffer(CKGenOp_Xdl_CShuffle& op,
                     std::string key,
                     char* buf); // get a kernel file from the JSON
}
} // namespace host
} // namespace ck
