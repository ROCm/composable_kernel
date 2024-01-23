#include "ck/host/types.hpp"
#include "../parse/include/types_fe.hpp"
#include "ck/host/stringutils.hpp"
#include <algorithm>
#include <stdexcept>

namespace ck {
namespace host {

std::string To_String(DataType_fe dt)
{
    switch(dt)
    {
    case DataType_fe::Float: return "fp32";
    case DataType_fe::Half:
        return "fp16";
        // case DataType::Int8: return "int8";
        // case DataType::Int32: return "int32";
    }
    throw std::runtime_error("Incorrect data type");
}

std::string To_String(Layout_fe dl)
{
    switch(dl)
    {
    case Layout_fe::Row: return "Row";
    case Layout_fe::Column: return "Col";
    }
    throw std::runtime_error("Incorrect layout");
}

/**std::string SequenceStr(const std::vector<int>& v)
{
    return "ck::Sequence<" +
           JoinStrings(Transform(v, [](int x) { return std::to_string(x); }), ", ") + ">";
}**/

/**std::string MakeTuple(const std::vector<std::string>& v)
{
    return "ck::Tuple<" + JoinStrings(v, ", ") + ">";
}**/

} // namespace host
} // namespace ck
