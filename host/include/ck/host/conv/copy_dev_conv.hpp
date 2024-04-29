// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>
#include <sstream>
#include <iterator>
#include <numeric>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace conv {

struct Copy_Problem_Conv
{
    std::size_t NumDim               = 0;
    std::size_t G                    = 0;
    std::size_t N                    = 0;
    std::size_t C                    = 0;
    std::size_t Hi                   = 0;
    std::size_t Wi                   = 0;
    std::size_t Ho                   = 0;
    std::size_t Wo                   = 0;
    std::size_t K                    = 0;
    std::size_t Y                    = 0;
    std::size_t X                    = 0;
    Layout ALayout                   = Layout::NHWGC;
    Layout BLayout                   = Layout::GKYXC;
    Layout ELayout                   = Layout::NHWGK;
    std::vector<Layout> DsLayout     = {};
    std::vector<bool> DsTrans        = {};
    DataType ADataType               = DataType::Half;
    DataType BDataType               = DataType::Half;
    DataType EDataType               = DataType::Half;
    std::vector<DataType> DsDataType = {};
    std::string AElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string BElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string CDEElementOp         = "ck::tensor_operation::element_wise::PassThrough";

    std::string GetIncludeHeader() const;

    std::vector<Solution> GetSolutions(const std::string& arch,
                                       const std::string& prologue,
                                       const std::string& epilogue) const;
};

} // namespace conv
} // namespace host
} // namespace ck
