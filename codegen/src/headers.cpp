// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/headers.hpp"
#include "ck/host/ck_tile_headers_preprocessor.hpp"
#include "ck_headers.hpp"
#include "ck_tile_headers.hpp"
#include "ck_codegen_headers.hpp"

namespace ck {
namespace host {

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif
const std::string config_header = "";
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif

std::unordered_map<std::string_view, std::string_view> GetHeaders()
{
    auto headers = ck_headers();
    headers.insert(std::make_pair("ck/config.h", config_header));
    return headers;
}

std::unordered_map<std::string, std::string> GetTileHeaders()
{
    auto tile_hdrs    = ck_tile_headers();
    auto codegen_hdrs = ck_codegen_headers();

    std::unordered_map<std::string, std::string> result;
    result.reserve(tile_hdrs.size() + codegen_hdrs.size());

    for(auto& [name, content] : tile_hdrs)
    {
        if(name == "ck_tile/core/utility/env.hpp")
        {
            result.emplace(std::string(name), "");
            continue;
        }
        result.emplace(std::string(name), strip_host_bodies(content));
    }

    for(auto& [name, content] : codegen_hdrs)
        result.emplace(std::string(name), std::string(content));

    return result;
}

} // namespace host
} // namespace ck
