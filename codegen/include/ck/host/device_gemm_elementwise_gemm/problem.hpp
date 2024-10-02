// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_gemm_elementwise_gemm {

// defines the problem specification for a GEMM operation
struct Problem
{
    std::size_t M             = 0;
    std::size_t N             = 0;
    std::size_t K             = 0;
    std::size_t O             = 0;
    bool TransA               = false;
    bool TransB0              = false;
    bool TransB1              = false;
    bool TransC               = false;
    DataType ADataType        = DataType::Half;
    DataType B0DataType       = DataType::Half;
    DataType B1DataType       = DataType::Half;
    DataType CDataType        = DataType::Half;
    std::string AElementOp    = PassThrough;
    std::string B0ElementOp   = PassThrough;
    std::string Acc0ElementOp = PassThrough;
    std::string B1ElementOp   = PassThrough;
    std::string CElementOp    = PassThrough;

    // returns the correct device op file for the operation
    std::string GetIncludeHeader() const;

    // returns a list of instances based on the problem spec and provided fusion operations
    std::vector<Solution> GetSolutions(const std::string& arch,
                                       const std::string& prologue,
                                       const std::string& epilogue) const;
};

} // namespace device_gemm_elementwise_gemm
} // namespace host
} // namespace ck
