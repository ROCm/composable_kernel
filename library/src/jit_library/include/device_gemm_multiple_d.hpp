// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>
#include <sstream>
#include <iterator>
#include <numeric>
#include "ck/solution_instances/gemm_add_add_fastgelu_instances.hpp"
#include "ck/ck.hpp"
#include "ck/utility/math.hpp"
#include "ck_headers.hpp"


namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_multiple_d {


struct Solution
{
    std::string template_str;
    index_t block_size;
    index_t grid_size;
};

std::string GetGemmSpec(const index_t m, 
                        const index_t n, 
                        const index_t k,
                        const index_t m_per_block,
                        const index_t n_per_block,
                        const index_t k_per_block) 
{
    std::string spec = "";
    if(math::integer_divide_ceil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if(math::integer_divide_ceil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if(math::integer_divide_ceil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";
    if(spec == "")
        return "ck::tensor_operation::device::GemmSpecialization::Default";

    return "ck::tensor_operation::device::GemmSpecialization::" + spec + "Padding";
}

index_t GetGridSize(const index_t m, 
                    const index_t n,
                    const index_t m_per_block,
                    const index_t n_per_block)
{
    return math::integer_divide_ceil(m, m_per_block) *
            math::integer_divide_ceil(n, n_per_block);
}

const std::unordered_set<std::string>& get_xdlop_archs()
{
    static std::unordered_set<std::string> supported_archs{"gfx90a"};
    return supported_archs;
}

struct Problem
{
    index_t M = 0;
    index_t N = 0;
    index_t K = 0;
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

    static const index_t ds_layout_idx   = 3;
    static const index_t ds_data_type_idx = 9;
    static const index_t e_data_type_idx = 10;
    static const index_t a_elementwise_op_idx = 11;
    static const index_t b_elementwise_op_idx = 12;
    static const index_t ds_elementwise_op_idx = 13;
    static const index_t gemm_spec_idx = 14;
    static const index_t block_size_idx  = 16;
    static const index_t m_per_block_idx = 17;
    static const index_t n_per_block_idx = 18;
    static const index_t k_per_block_idx = 19;

private:
    auto GetInstances(const std::string& arch) const
    {
        std::vector<std::string> instances;
        const bool quantize = ADataType == "int8_t" and BDataType == "int8_t";
        if (get_xdlop_archs().find(arch) != get_xdlop_archs().end())
        {
            instance::gemm_add_add_fastgelu_instances all_instances{};
            if(TransA and TransB)
                instances = all_instances.get_col_col_instances(quantize);
            else if(TransA and not TransB)
                instances = all_instances.get_col_row_instances(quantize);
            else if(not TransA and not TransB)
                instances = all_instances.get_row_row_instances(quantize);
            else
                instances = all_instances.get_row_col_instances(quantize);
        }
        return instances;
    }

    auto MakeLayoutTuple(const std::vector<bool>& layouts) const
    {
        std::string layout_tuple = "ck::Tuple<";
        auto it = layouts.begin();
        while(it != layouts.end())
        {
            layout_tuple += *it ? "ck::tensor_layout::gemm::ColumnMajor" : "ck::tensor_layout::gemm::RowMajor";
            it = std::next(it);
            if (it != layouts.end())
                layout_tuple += ", ";
        }
            
        return layout_tuple + ">";
    }

    auto MakeTypeTuple(const std::vector<std::string>& types) const
    {
        std::string type_tuple = "ck::Tuple<";
        auto it = types.begin();
        while(it != types.end())
        {
            type_tuple += *it;
            it = std::next(it);
            if (it != types.end())
                type_tuple += ", ";
        }
        return type_tuple + ">";
    }

    auto MakeSolution(index_t idx, const std::string& arch) const
    {
        auto template_str = GetInstances(arch).at(idx);
        std::istringstream iss(template_str);
        std::vector<std::string> params(std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>());
        
        if (ADataType == "int8_t" and BDataType == "int8_t")
        {
            // Change CBlockTransfer ScalarPerVector if Ds contains other types
            if (std::any_of(DsDataType.begin(), DsDataType.end(), [](auto t) { return t == "ck::half_t"; }))
            {
                params[params.size() - 3] = "8";
            }
            if (std::any_of(DsDataType.begin(), DsDataType.end(), [](auto t) { return t == "float"; }))
            {
                params[params.size() - 3] = "4";
            }
        }

        params[a_elementwise_op_idx] = AElementOp;
        params[b_elementwise_op_idx] = BElementOp;
        params[ds_layout_idx] = MakeLayoutTuple(DsLayout);
        params[ds_data_type_idx] = MakeTypeTuple(DsDataType);
        params[ds_elementwise_op_idx] = CDEElementOp;
        params[e_data_type_idx] = EDataType;
        auto block_size_str = params[block_size_idx];
        auto m_per_block_str = params[m_per_block_idx];
        auto n_per_block_str = params[n_per_block_idx];
        auto k_per_block_str = params[k_per_block_idx];
        const auto block_size  = std::stoi(block_size_str);
        const auto m_per_block = std::stoi(m_per_block_str);
        const auto n_per_block = std::stoi(n_per_block_str);
        const auto k_per_block = std::stoi(k_per_block_str);
        const auto grid_size   = GetGridSize(M, N, m_per_block, n_per_block);
        params[gemm_spec_idx]  = GetGemmSpec(M, N, K, m_per_block, n_per_block, k_per_block);

        std::string str = std::accumulate(params.begin() + 1, params.end(), std::string{},
                                        [](const std::string& a, const std::string& b) {
                                            return a.empty() ? b : a + ", " + b;
                                        });
        str = params.front() + "< " + str + ">";
        
        return Solution{str, block_size, grid_size};
    }

public:
    auto GetHeaders() const
    {
        return ck_headers();
    }

    auto GetIncludeHeader() const
    {
        return instance::gemm_add_add_fastgelu_instances{}.get_include_header();
    }

    auto GetSolutions(const std::string& arch) const
    {
        std::vector<Solution> solutions;
        const auto num_instances = GetInstances(arch).size();
        for (auto i = 0; i < num_instances; ++i)
        {
            solutions.push_back(MakeSolution(i, arch));
        }

        return solutions;
    }
};

} // namespace device_gemm_multiple_d
} // namespace device
} // namespace tensor_operation
} // namespace ck
