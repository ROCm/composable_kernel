// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/headers.hpp"
#include "ck_headers.hpp"

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

} // namespace host
} // namespace ck
