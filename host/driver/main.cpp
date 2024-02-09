
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/conv/conv_op.hpp"
#include "../parse/include/op.hpp"
#include "../parse/include/op_conv.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/types.hpp"
#include <iomanip>
#include <fstream>

struct Emitters
{
    std::unordered_map<std::string, std::function<std::vector<std::string>()>> m;

    template <class T>
    void Register(const std::string& name, const std::string& prologue, const std::string& epilogue)
    {

        m[name] = [&] {
            auto ops = T::CreateOperations(prologue, epilogue);
            return ck::host::Transform(
                ops, [&](const auto& op) { return op.ToSolution().ToTemplateString(); });
        };
        m.at(name)();
    }

    std::string Emit(const std::string& name) { return ck::host::JoinStrings(m.at(name)(), "\n"); }

    std::vector<std::string> List() const
    {
        return ck::host::Transform(m, [](auto&& p) { return p.first; });
    }

    template <class T>
    void Select(ck::host::device_gemm_multiple_d::Problem& prob,
                const std::string& name,
                const std::string& prologue,
                const std::string& epilogue)
    {
        auto M = std::to_string(prob.M);
        auto N = std::to_string(prob.N);
        auto K = std::to_string(prob.K);
        std::cout << "M: " << M << std::endl;
        std::cout << "N: " << N << std::endl;
        std::cout << "K: " << K << std::endl;

        // TODO: add argument check here
        // generate all instances
        auto ops = T::CreateOperations(prologue, epilogue);
        std::vector<std::string> match;
        for(auto op : ops)
        {
            // check that user's prob desc matches the instances
            if(prob.ADataType == op.A.element || prob.BDataType == op.B.element ||
               prob.EDataType == op.E.element || ck::host::ToLayout(prob.TransA) == op.A.layout ||
               ck::host::ToLayout(prob.TransB) == op.B.layout ||
               ck::host::ToLayout(prob.TransE) == op.E.layout || prob.AElementOp == op.a_elem_op ||
               prob.BElementOp == op.b_elem_op || prob.CDEElementOp == op.cde_elem_op)
            {
                match.push_back(op.ToSolution().ToTemplateString());
                std::cout << op.ToSolution().ToTemplateString() << std::endl;
            }
        }
    }
};

int main(int argc, const char* argv[])
{
    std::string prog = argv[0];
    std::vector<std::string> args(argv + 1, argv + argc);

    std::string prologue = R"(struct AlphaBetaAdd
{
    AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, ck::half_t>(
        ck::half_t& e, const float& c, const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * c + beta_ * ck::type_convert<float>(d));
    };

    float alpha_;
    float beta_;
};
using Prologue = AlphaBetaAdd;)";
    std::string epilogue = "";

    Emitters e;
    e.Register<ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle>(
        "DeviceGemmMultipleD_Xdl_CShuffle", prologue, epilogue);
    e.Register<ck::host::conv::Operation_Conv>("DeviceConv", prologue, epilogue);

    if(args.empty() or std::any_of(args.begin(), args.end(), [](auto arg) {
           return arg == "-h" or arg == "--help";
       }))
    {
        std::cout << "USAGE:" << std::endl;
        std::cout << "    " << prog << " [TEMPLATE]" << std::endl;
        std::cout << std::endl;
        std::cout << "FLAGS:" << std::endl;
        std::cout << "    -h, --help                     Show help" << std::endl;
        std::cout << std::endl;
        std::cout << "TEMPLATES:" << std::endl;
        for(auto x : e.List())
            std::cout << "    " << x << std::endl;
        std::cout << std::endl;
        return 0;
    }

    ck::host::device_gemm_multiple_d::Problem prob;
    prob.M = 1024;
    prob.N = 1024;
    prob.K = 1024;
    e.Select<ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle>(
        prob, "DeviceGemmMultipleD_Xdl_CShuffle", prologue, epilogue);

    // for(auto name : args)
    //  std::cout << e.Emit(name) << std::endl;

    return 0;
}
