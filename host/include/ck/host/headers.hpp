// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ck {
namespace host {

std::unordered_map<std::string, std::pair<const char*, const char*>> GetHeaders();

} // namespace host
} // namespace ck
