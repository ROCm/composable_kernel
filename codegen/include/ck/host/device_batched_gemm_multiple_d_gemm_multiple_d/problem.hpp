// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_batched_gemm_multiple_d_gemm_multiple_d {

// defines the problem specification for a GEMM_ELEMENTWISE_GEMM operation
struct Problem
{
    std::size_t M                     = 0;
    std::size_t N                     = 0;
    std::size_t K                     = 0;
    std::size_t O                     = 0;
    bool TransA0                      = false;
    bool TransB0                      = false;
    std::vector<bool> D0sTrans        = {};
    bool TransB1                      = false;
    std::vector<bool> D1sTrans        = {};
    bool TransE1                      = false;
    DataType A0DataType               = DataType::Half;
    DataType B0DataType               = DataType::Half;
    std::vector<DataType> D0sDataType = {};
    DataType B1DataType               = DataType::Half;
    std::vector<DataType> D1sDataType = {};
    DataType E1DataType               = DataType::Half;
    std::string A0ElementOp           = PassThrough;
    std::string B0ElementOp           = PassThrough;
    std::string CDE0ElementOp         = PassThrough;
    std::string B1ElementOp           = PassThrough;
    std::string CDE1ElementOp         = PassThrough;

    // returns the correct device op file for the operation
    std::string GetIncludeHeader() const;

    // returns a list of instances based on the problem spec and provided fusion operations
    std::vector<Solution> GetSolutions(const std::string& arch,
                                       const std::string& prologue,
                                       const std::string& epilogue) const;
};

} // namespace device_batched_gemm_multiple_d_gemm_multiple_d
} // namespace host
} // namespace ck
