// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_fwd/problem.hpp"
#include "ck/host/device_fmha_fwd/operation.hpp"
#include <algorithm>

namespace ck {
namespace host {
namespace device_fmha_fwd {

// Based on factories defined in fmha_fwd.py
bool IsSupportedArch(const std::string& arch)
{
    if(arch.find("gfx950") == 0)
        return false; // WIP
    if(arch.find("gfx9") == 0)
        return true;
    if(arch.find("gfx12") == 0)
        return false; // WIP
    return false;
}

std::string Problem::GetIncludeHeader() const
{
    return "ck/host/device_fmha_fwd/fmha_fwd_wrapper.hpp";
}

std::vector<Solution> Problem::GetSolutions(const std::string& arch) const
{
    if(!IsSupportedArch(arch))
        return {};

    auto ops = Operation::CreateOperations(*this, arch);
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [](const auto& op) {
        return op.ToSolution();
    });
    return result;
}

} // namespace device_fmha_fwd
} // namespace host
} // namespace ck
