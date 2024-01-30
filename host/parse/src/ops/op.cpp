#include <iostream>
#include <iomanip>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../parse/include/op.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "../parse/include/types_fe.hpp"
#include <cassert>

namespace ck {
namespace host {

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
                       std::size_t K)
{
    op.A.element  = ADataType;
    op.B.element  = BDataType;
    op.Ds.element = DsDataType;
    op.E.element  = EDataType;
    op.A.layout   = ALayout;
    op.B.layout   = BLayout;
    op.Ds.layout  = DsLayout;
    op.E.layout   = ELayout;
    op.M          = M;
    op.N          = N;
    op.K          = K;
    /**for(int i = 0; i < op.Ds.size(); i++)
    {
        op.Ds[i].element = DsDataType;
        op.Ds[i].layout  = DsLayout;
    }**/
    std::string inst_key = To_String(ADataType) + To_String(BDataType) + To_String(DsDataType) +
                           To_String(EDataType) + To_String(ALayout) + To_String(BLayout) +
                           To_String(DsLayout) + To_String(ELayout);
    return inst_key;
}

nlohmann::json CKGenGetOpParams()
{
    std::ifstream f;
    f.open("/root/workspace/composable_kernel/host/build/op_inst.json");
    // std::cout << "located file" << std::endl;
    if(!f)
    {
        std::cout << "cannot open file" << std::endl;
    }
    // std::cout << f.rdbuf();
    nlohmann::json data = nlohmann::json::parse(f);
    return data;
}

void CKGenSetOpFusion(std::string Prologue)
{
    nlohmann::json j = CKGenGetOpParams();
    nlohmann::json update;
    update                       = j;
    update["fusion"]["prologue"] = Prologue;
    std::ofstream out("/root/workspace/composable_kernel/host/build/op_inst.json");
    out << std::setw(4) << update;
}

char* CKGenGetBuffer(CKGenOp_Xdl_CShuffle& op, std::string key, char* buf)
{
    nlohmann::json data = CKGenGetOpParams();
    // const char *buf;
    std::string tmp = "";

    // retrieve specific instance
    std::string inst = data["instances"][key]["0"].get<std::string>();
    // std::cout << "specific inst: " << inst << std::endl;

    // run this in a loop?
    // write includes and prologue into file - TODO: needs to be ordered -> update JSON version
    for(const auto& item : data.items())
    {
        if(item.key() == "instances")
        {
            break;
        }
        // std::cout << data[item.key()].get<std::string>();
        // std::cout << item.key() << "\n";
        for(const auto& val : item.value().items())
        {
            tmp += (val.value().get<std::string>() + "\n");
        }
    }
    // write in instance +global function
    tmp += inst;
    buf = const_cast<char*>(tmp.c_str()); // better option for char return?
    printf("%s", buf);
    return buf;
    // return buf;
}
} // namespace host
} // namespace ck
