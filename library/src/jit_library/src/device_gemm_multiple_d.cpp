#include "ck/host/device_gemm_multiple_d.hpp"
#include "ck/host/common.hpp"
#include "ck/solution_instances/gemm_add_add_fastgelu_instances.hpp"
#include <unordered_set>

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

std::string GetGemmSpec(const std::size_t m, 
                        const std::size_t n, 
                        const std::size_t k,
                        const std::size_t m_per_block,
                        const std::size_t n_per_block,
                        const std::size_t k_per_block) 
{
    std::string spec = "";
    if(integer_divide_ceil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if(integer_divide_ceil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if(integer_divide_ceil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";
    if(spec == "")
        return "ck::tensor_operation::device::GemmSpecialization::Default";

    return "ck::tensor_operation::device::GemmSpecialization::" + spec + "Padding";
}

std::size_t GetGridSize(const std::size_t m, 
                    const std::size_t n,
                    const std::size_t m_per_block,
                    const std::size_t n_per_block)
{
    return integer_divide_ceil(m, m_per_block) *
            integer_divide_ceil(n, n_per_block);
}

const std::unordered_set<std::string>& get_xdlop_archs()
{
    static std::unordered_set<std::string> supported_archs{"gfx90a"};
    return supported_archs;
}

std::vector<std::string> Problem::GetInstances(const std::string& arch) const
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

std::string Problem::MakeLayoutTuple(const std::vector<bool>& layouts) const
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

std::string Problem::MakeTypeTuple(const std::vector<std::string>& types) const
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

Solution Problem::MakeSolution(std::size_t idx, const std::string& arch) const
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


} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
