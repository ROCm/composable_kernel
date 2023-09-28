#include "ck/host/headers.hpp"
#include "ck_headers.hpp"

namespace ck {
namespace host {

const std::string config_header = "";

std::unordered_map<std::string, std::pair<const char*, const char*>> GetHeaders()
{
    auto headers = ck_headers();
    headers.insert(std::make_pair(
        "ck/config.h",
        std::make_pair(config_header.data(), config_header.data() + config_header.size())));
    return headers;
}

} // namespace host
} // namespace ck