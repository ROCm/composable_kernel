// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>
#include <sstream>
#include <iterator>
#include <numeric>
#include "ck/host/common.hpp"

namespace ck {
namespace host {
namespace device_gemm_softmax_gemm {

struct Problem
{
    std::size_t M                    = 0;
    std::size_t N                    = 0;
    std::size_t K                    = 0;
    std::size_t O                    = 0;
    bool TransA                      = false;
    bool TransB                      = false;
    bool TransB1                     = false;
    bool TransC                      = false;
    DataType ADataType               = DataType::Half;
    DataType BDataType               = DataType::Half;
    DataType B1DataType              = DataType::Half;
    DataType CDataType               = DataType::Half;
    DataType AccDataType             = DataType::Float;
    std::string AElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string BElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string AccElementOp         = "ck::tensor_operation::element_wise::Scale";
    std::string B1ElementOp          = "ck::tensor_operation::element_wise::PassThrough";

    static const std::size_t ds_layout_idx         = 3;
    static const std::size_t ds_data_type_idx      = 9;
    static const std::size_t e_data_type_idx       = 10;
    static const std::size_t a_elementwise_op_idx  = 11;
    static const std::size_t b_elementwise_op_idx  = 12;
    static const std::size_t ds_elementwise_op_idx = 13;
    static const std::size_t gemm_spec_idx         = 14;
    static const std::size_t block_size_idx        = 16;
    static const std::size_t m_per_block_idx       = 17;
    static const std::size_t n_per_block_idx       = 18;
    static const std::size_t k_per_block_idx       = 19;

    std::string GetIncludeHeader() const;

    std::vector<Solution> GetSolutions(const std::string& arch) const;

    private:
    std::vector<std::string> GetInstances(const std::string& arch) const;

    Solution MakeSolution(std::size_t idx, const std::string& arch) const;
};

} // namespace device_gemm_softmax_gemm
} // namespace host
} // namespace ck
