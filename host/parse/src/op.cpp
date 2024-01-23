#include <iostream>
#include <iomanip>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../parse/include/op.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
//#include "ck/host/types.hpp"
#include "../parse/include/types_fe.hpp"
#include <cassert>

namespace ck {
namespace host {

std::string CKGenOp_Xdl_CShuffle::CKGenSetOp(CKGenOp_Xdl_CShuffle& op,
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
    op.A.element = ADataType;
    op.B.element = BDataType;
    // Transform(op.Ds, [](auto tensor) { tensor.element = DsDataType; })
    op.E.element = EDataType;
    op.A.layout  = ALayout;
    op.B.layout  = BLayout;
    op.E.layout  = ELayout;
    op.M         = M;
    op.N         = N;
    op.K         = K;
    for(int i = 0; i < op.Ds.size(); i++)
    {
        op.Ds[i].element = DsDataType;
        op.Ds[i].layout  = DsLayout;
    }
    std::string inst_key = To_String(ADataType) + To_String(BDataType) + To_String(DsDataType) +
                           To_String(EDataType) + To_String(ALayout) + To_String(BLayout) +
                           To_String(DsLayout) + To_String(ELayout);
    return inst_key;
}

nlohmann::json CKGenOp_Xdl_CShuffle::CKGenGetOpParams()
{
    std::ifstream f;
    f.open("/root/workspace/composable_kernel/host/build/op_inst.json");
    std::cout << "located file" << std::endl;
    if(!f)
    {
        std::cout << "cannot open file" << std::endl;
    }
    // std::cout << f.rdbuf();
    nlohmann::json data = nlohmann::json::parse(f);
    return data;
}
void CKGenOp_Xdl_CShuffle::CKGenSetOpFusion(std::string Prologue)
{ // TODO: fix the argument type
    nlohmann::json j = CKGenGetOpParams();
    nlohmann::json update;
    update                       = j;
    update["fusion"]["prologue"] = Prologue;
    std::ofstream out("/root/workspace/composable_kernel/host/build/op_inst.json");
    out << std::setw(4) << update;
}

char* CKGenOp_Xdl_CShuffle::CKGenGetBuffer(CKGenOp_Xdl_CShuffle& op, std::string key)
{
    nlohmann::json data = CKGenGetOpParams();
    const char *buf;
    std::string tmp = "";

    std::cout << "key: " << key << std::endl;
    // retrieve specific instance
    std::cout << "got parsed data" << std::endl;
    std::string inst = data["instances"][key]["0"].get<std::string>();
    //std::cout << "specific inst: " << inst << std::endl;
    // run this in a loop?
    // write includes and prologue into file
    for(const auto& item : data.items())
    {
        if(item.key() == "instances")
        {
            break;
        }
	//std::cout << data[item.key()].get<std::string>();
        //std::cout << item.key() << "\n";
        for(const auto& val : item.value().items())
        {
            tmp += (val.value().get<std::string>() + "\n");
        }
    }
    // write in instance +global function
    tmp += inst;
    //std::cout << tmp << std::endl;
    buf = tmp.c_str();
    printf("%s",buf);
    //return buf;
}
} // namespace host
} // namespace ck
