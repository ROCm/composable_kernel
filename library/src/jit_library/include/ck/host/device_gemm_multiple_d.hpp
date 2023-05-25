// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>
#include <sstream>
#include <iterator>
#include <numeric>


namespace ck {
namespace host {
namespace device_gemm_multiple_d {


struct Solution
{
    std::string template_str;
    std::size_t block_size;
    std::size_t grid_size;
};

struct Problem
{
    std::size_t M = 0;
    std::size_t N = 0;
    std::size_t K = 0;
    bool TransA = false;
    bool TransB = false;
    bool TransE = false;
    std::vector<bool> DsLayout = {};
    std::string ADataType = "ck::half_t";
    std::string BDataType = "ck::half_t";
    std::string EDataType = "ck::half_t";
    std::vector<std::string> DsDataType = {};
    std::string AElementOp = "ck::tensor_operation::element_wise::PassThrough";
    std::string BElementOp = "ck::tensor_operation::element_wise::PassThrough";
    std::string CDEElementOp = "ck::Tuple<>";

    static const std::size_t ds_layout_idx   = 3;
    static const std::size_t ds_data_type_idx = 9;
    static const std::size_t e_data_type_idx = 10;
    static const std::size_t a_elementwise_op_idx = 11;
    static const std::size_t b_elementwise_op_idx = 12;
    static const std::size_t ds_elementwise_op_idx = 13;
    static const std::size_t gemm_spec_idx = 14;
    static const std::size_t block_size_idx  = 16;
    static const std::size_t m_per_block_idx = 17;
    static const std::size_t n_per_block_idx = 18;
    static const std::size_t k_per_block_idx = 19;

private:
    std::vector<std::string> GetInstances(const std::string& arch) const;

    std::string MakeLayoutTuple(const std::vector<bool>& layouts) const;

    std::string MakeTypeTuple(const std::vector<std::string>& types) const;

    Solution MakeSolution(std::size_t idx, const std::string& arch) const
;

public:
    std::string GetIncludeHeader() const;

    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
