
#include "ck/host/common.hpp"
#include "ck_headers.hpp"

namespace ck {
namespace host {

std::unordered_map<std::string, std::pair<const char*,const char*>> GetHeaders()
{
    return ck_headers();
}

std::size_t integer_divide_ceil(std::size_t x, std::size_t y)
{
    return (x + y - std::size_t{1}) / y;
}

} // namespace host
} // namespace ck
