#include "ck/host/device_batched_gemm_softmax_gemm.hpp"
#include "ck/host/common.hpp"
#include "batched_gemm_softmax_gemm_instances.hpp"
#include <algorithm>
#include <unordered_set>

namespace ck {
namespace host {
namespace device_batched_gemm_softmax_gemm {

std::string GetGemmSpec(const std::size_t m,
                        const std::size_t n,
                        const std::size_t k,
                        const std::size_t n1,
                        const std::size_t m_per_block,
                        const std::size_t n_per_block,
                        const std::size_t k_per_block,
                        const std::size_t n1_per_block)
{
    std::string spec = "";
    if(integer_divide_ceil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if(integer_divide_ceil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if(integer_divide_ceil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";
    if(integer_divide_ceil(n1, n1_per_block) * n1_per_block - n1 != 0)
        spec += "O";
    if(spec == "")
        return "ck::tensor_operation::device::GemmSpecialization::Default";

    return "ck::tensor_operation::device::GemmSpecialization::" + spec + "Padding";
}

std::size_t GetGridSize(const std::size_t m,
                        const std::size_t n,
                        const std::size_t m_per_block,
                        const std::size_t n_per_block)
{
    return integer_divide_ceil(m, m_per_block) * integer_divide_ceil(n, n_per_block);
}

const std::unordered_set<std::string>& get_xdlop_archs()
{
    static std::unordered_set<std::string> supported_archs{"gfx90a", "gfx908", "gfx940"};
    return supported_archs;
}

std::vector<std::string> Problem::GetInstances(const std::string& arch) const
{
    std::vector<std::string> instances;
    if(get_xdlop_archs().find(arch) != get_xdlop_archs().end())
    {
        ck::host::instance::batched_gemm_softmax_gemm_instances all_instances{};
        instances = all_instances.get_instances();
    }
    return instances;
}

Solution Problem::MakeSolution(std::size_t idx, const std::string& arch) const
{
    auto template_str = GetInstances(arch).at(idx);
    std::istringstream iss(template_str);
    std::vector<std::string> params(std::istream_iterator<std::string>{iss},
                                    std::istream_iterator<std::string>());

    params[AElementwiseOperation_idx]   = AElementOp;
    params[B0ElementwiseOperation_idx]  = BElementOp;
    params[B1ElementwiseOperation_idx]  = BElementOp;
    params[CElementwiseOperation_idx]   = CElementOp;
    params[Acc0ElementwiseOperation_idx] = AccElementOp;
    auto block_size_str           = params[BlockSize_idx];
    auto m_per_block_str          = params[Gemm01MPerBlock_idx];
    auto n_per_block_str          = params[Gemm0NPerBlock_idx];
    auto k_per_block_str          = params[Gemm0KPerBlock_idx];
    auto n1_per_block_str         = params[Gemm1NPerBlock_idx];
    const std::size_t block_size  = std::stoi(block_size_str);
    const std::size_t m_per_block = std::stoi(m_per_block_str);
    const std::size_t n_per_block = std::stoi(n_per_block_str);
    const std::size_t k_per_block = std::stoi(k_per_block_str);
    const std::size_t n1_per_block = std::stoi(n1_per_block_str);
    const std::size_t grid_size    = GetGridSize(M, O, m_per_block, n1_per_block);
    params[GEMMSpecialization_idx] = GetGemmSpec(M, N, K, O, m_per_block, n_per_block, k_per_block, n1_per_block);

    std::string str = std::accumulate(
        params.begin() + 1,
        params.end(),
        std::string{},
        [](const std::string& a, const std::string& b) { return a.empty() ? b : a + ", " + b; });
    str = params.front() + "< " + str + ">";

    return Solution{str, block_size, grid_size};
}

std::string Problem::GetIncludeHeader() const
{
    return ck::host::instance::batched_gemm_softmax_gemm_instances{}.get_include_header();
}

std::vector<Solution> Problem::GetSolutions(const std::string& arch) const
{
    std::vector<Solution> solutions;
    const std::size_t num_instances = GetInstances(arch).size();
    for(std::size_t i = 0; i < num_instances; ++i)
    {
        solutions.push_back(MakeSolution(i, arch));
    }

    return solutions;
}

} // namespace device_batched_gemm_softmax_gemm
} // namespace host
} // namespace ck
