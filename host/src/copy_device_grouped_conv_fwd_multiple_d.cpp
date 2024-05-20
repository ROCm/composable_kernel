
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/device_grouped_conv_fwd_multiple_d/copy_conv_fwd_problem.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/copy_conv_fwd_op.hpp"
#include "ck/host/utils.hpp"
#include <algorithm>
#include <iostream>

namespace ck {
namespace host {
namespace conv {

// return the relevant device op file based on the operation
// NOTE: this device op file is modified from the original CK Version to call the kernel from a
// device function
std::string Copy_Problem_Conv_Fwd::GetIncludeHeader() const
{
    return "ck/tensor_operation/gpu/device/impl/"
           "mod_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp";
}

// return vector of instances when provided with a problem instance
std::vector<Solution> Copy_Problem_Conv_Fwd::GetSolutions(const std::string& arch,
                                                          const std::string& prologue,
                                                          const std::string& epilogue) const
{
    if(get_xdlop_archs().count(arch) == 0)
        return {};
    auto ops = ck::host::conv::Copy_Operation_Conv_Fwd_Xdl_Cshuffle::CreateOperations(
        *this, prologue, epilogue);
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [&](const auto& op) {
        return op.ToSolution();
    });
    return result;
}

} // namespace conv
} // namespace host
} // namespace ck