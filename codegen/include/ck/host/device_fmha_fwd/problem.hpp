// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd {

struct Problem
{
    std::size_t M = 0; // seqlen_q
    std::size_t N = 0; // seqlen_k
    std::size_t K = 0; // hdim_q
    std::size_t O = 0; // hdim_v

    std::size_t batch = 0;
    std::size_t nhead = 0; // nhead_q == nhead_k

    DataType dtype = DataType::Half;

    bool is_v_rowmajor = true; // true=[N,O], false=[O,N]
    bool is_causal     = false;
    bool has_bias      = false;

    std::string GetIncludeHeader() const;

    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_fwd
} // namespace host
} // namespace ck
