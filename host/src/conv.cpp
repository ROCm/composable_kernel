
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/conv/dev_conv.hpp"
#include "ck/host/conv/conv_op.hpp"
#include "ck/host/utils.hpp"
#include <algorithm>

namespace ck {
namespace host {
namespace conv {

std::string Problem_Conv::GetIncludeHeader() const
{
    return "ck/tensor_operation/gpu/device/impl/"
           "device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp";
}

std::vector<Solution> Problem_Conv::GetSolutions(const std::string& arch,
                                                 const std::string& prologue,
                                                 const std::string& epilogue) const
{
    if(get_xdlop_archs().count(arch) == 0)
        return {};
    auto ops = ck::host::conv::Operation_Conv::CreateOperations(*this, prologue, epilogue);
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [&](const auto& op) {
        return op.ToSolution();
    });
    return result;
}

} // namespace conv
} // namespace host
} // namespace ck
