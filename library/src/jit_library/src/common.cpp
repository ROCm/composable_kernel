
#include "ck/host/common.hpp"
#include "ck_headers.hpp"
#include <stdexcept>
#include <algorithm>

namespace ck {
namespace host {

std::string ToString(DataType dt)
{
    switch(dt)
    {
    case DataType::Float: return "float";
    case DataType::Half: return "ck::half_t";
    case DataType::Int8: return "int8_t";
    case DataType::Int32: return "int32_t";
    }
    throw std::runtime_error("Incorrect data type");
}

const std::string config_header = "";

std::unordered_map<std::string_view, std::string_view> GetHeaders()
{
    auto headers = ck_headers();
    headers.insert(
        {"ck/config.h", config_header});
    return headers;
}

std::size_t integer_divide_ceil(std::size_t x, std::size_t y)
{
    return (x + y - std::size_t{1}) / y;
}

} // namespace host
} // namespace ck
