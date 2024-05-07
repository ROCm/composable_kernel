#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/copy_conv_fwd_op.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/types.hpp"
#include <iomanip>
#include <fstream>

struct Emitters
{
    std::unordered_map<std::string, std::function<std::vector<std::string>()>> m;

    // retrieve the hard-coded instances provided and template them > store in a map
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

    // function to retrieve all instances for a certain problem size
    // NOTE: this is specifically for convolutions
    template <class T>
    void Select(ck::host::conv::Copy_Problem_Conv_Fwd& prob,
                const std::string& name,
                const std::string& prologue,
                const std::string& epilogue)
    {
        auto G  = std::to_string(prob.G);
        auto N  = std::to_string(prob.N);
        auto C  = std::to_string(prob.C);
        auto K  = std::to_string(prob.K);
        auto Y  = std::to_string(prob.Y);
        auto X  = std::to_string(prob.X);
        auto Hi = std::to_string(prob.Hi);
        auto Wi = std::to_string(prob.Wi);
        auto Ho = std::to_string(prob.Ho);
        auto Wo = std::to_string(prob.Wo);
        // TODO: add argument check here
        // generate all instances
        auto ops = T::CreateOperations(prologue, epilogue);
        std::vector<std::string> match;
        for(auto op : ops)
        {
            // check that user's prob desc matches the instances
            if(prob.ADataType == op.A.element || prob.BDataType == op.B.element ||
               prob.EDataType == op.E.element || prob.ALayout == op.A.layout ||
               prob.BLayout == op.B.layout || prob.ELayout == op.E.layout ||
               prob.AElementOp == op.a_elem_op || prob.BElementOp == op.b_elem_op ||
               prob.CDEElementOp == op.cde_elem_op)
            {
                match.push_back(op.ToSolution().ToTemplateString());
            }
        }
    }
};

int main(int argc, const char* argv[])
{
    std::string prog = argv[0];
    std::vector<std::string> args(argv + 1, argv + argc);

    // user provided fusion
    std::string prologue = R"(struct Prologue
{
    Prologue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

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
)";
    std::string epilogue = "";

    // Load in operations into the Register
    Emitters e;
    e.Register<ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle>(
        "DeviceGemmMultipleD_Xdl_CShuffle", prologue, epilogue);
    e.Register<ck::host::conv::Copy_Operation_Conv_Fwd_Xdl_Cshuffle>(
        "DeviceConv", prologue, epilogue);

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

    // can specify problem size to view specific instances for
    ck::host::conv::Copy_Problem_Conv_Fwd prob;
    prob.G       = 1024;
    prob.N       = 1024;
    prob.C       = 1024;
    prob.K       = 1024;
    prob.X       = 1024;
    prob.Y       = 1024;
    prob.Hi      = 1024;
    prob.Wi      = 1024;
    prob.Ho      = 1024;
    prob.Wo      = 1024;
    prob.ALayout = ck::host::Layout::GNHWC;
    prob.BLayout = ck::host::Layout::GKYXC;
    prob.ELayout = ck::host::Layout::GNHWK;
    e.Select<ck::host::conv::Copy_Operation_Conv_Fwd_Xdl_Cshuffle>(
        prob, "Device_Conv", prologue, epilogue);

    // print out all the instances for the operation that was chosen at the command line
    for(auto name : args)
        std::cout << e.Emit(name) << std::endl;

    return 0;
}
