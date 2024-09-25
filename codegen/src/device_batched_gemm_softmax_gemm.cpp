
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/device_batched_gemm_softmax_gemm/problem.hpp"
#include "ck/host/device_batched_gemm_softmax_gemm/operation.hpp"
#include "ck/host/utils.hpp"
#include <algorithm>

namespace ck {
namespace host {
namespace device_batched_gemm_softmax_gemm {

// return the relevant device op file based on the operation
std::string Problem::GetIncludeHeader() const
{
    return "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp";
}

// returns templated instances when provided with a problem specification
std::vector<Solution> Problem::GetSolutions(const std::string& arch,
                                            const std::string& prologue,
                                            const std::string& epilogue) const
{
    if(get_xdlop_archs().count(arch) == 0)
        return {};
    auto ops = ck::host::device_batched_gemm_softmax_gemm::Operation_Xdl_CShuffle::CreateOperations(
        *this, prologue, epilogue); // obtains vector of instances
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [&](const auto& op) {
        return op.ToSolution(); // template instance with correct values
    });
    return result;
}

} // namespace device_batched_gemm_softmax_gemm
} // namespace host
} // namespace ck
