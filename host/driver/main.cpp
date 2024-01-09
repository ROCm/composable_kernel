
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/stringutils.hpp"
#include <iomanip>
#include <fstream>
#include <nlohmann/json.hpp>

struct Emitters
{
    std::unordered_map<std::string, std::function<std::vector<std::string>()>> m;

    template <class T>
    void Register(const std::string& name)
    {
        m[name] = [] {
            auto ops = T::CreateOperations();
            return ck::host::Transform(
                ops, [](const auto& op) { return op.ToSolution().ToTemplateString(); });
        };

        std::ofstream out("./op_inst.json");
        // populate json
        nlohmann::json data;

        // include section
        data["include"] = "#include <string>";

        // prologue and epilogue
        data["fusion"] = {{"prologue", "using Prologue = BaseOp;"},
                          {"epilogue", "using Epilogue = BaseOp;"}};

        // add instances
        // TODO: separate problem and tuning parameters to nest further
        data["instances"] = nlohmann::json::object();
        for(int x = 0; x < m[name]().size(); x++)
        {
            std::string tmp        = std::to_string(x);
            data["instances"][tmp] = m[name]()[x];
        }

        // the run function

        // traits (other information)

        out << std::setw(4) << data;
    }

    std::string Emit(const std::string& name)
    {
        return "std::tuple<\n" + ck::host::JoinStrings(m.at(name)(), ",\n") + ">";
    }

    std::vector<std::string> List() const
    {
        return ck::host::Transform(m, [](auto&& p) { return p.first; });
    }
};

int main(int argc, const char* argv[])
{
    std::string prog = argv[0];
    std::vector<std::string> args(argv + 1, argv + argc);
    Emitters e;
    e.Register<ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle>(
        "DeviceGemmMultipleD_Xdl_CShuffle");

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

    for(auto name : args)
        std::cout << e.Emit(name) << std::endl;

    return 0;
}
