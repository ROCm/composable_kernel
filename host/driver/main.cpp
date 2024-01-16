
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
    nlohmann::json inst;

    std::string get_includes()
    {
        static const char* const include_files = R"(
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/literals.hpp"

#include <sstream>
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
)";
        return include_files;
    }

    template <class T>
    void Register(const std::string& name)
    {
        std::ofstream out("./op_inst.json");
        // populate json

        // include section
        std::string inc = get_includes();
        inst["include"] = inc;

        // prologue and epilogue
        inst["fusion"] = {{"prologue", "using Prologue = BaseOp;"},
                          {"epilogue", "using Epilogue = BaseOp;"}};

        // std::cout << "made it here" << std::endl;
        // std::cout << inst.type_name() << std::endl;
        // std::cout << "size before function: " << inst.size() << std::endl;
        // std::cout << "Address before: " << &inst << std::endl;
        // std::cout << inst.dump() << std::endl;

        m[name] = [&] {
            auto ops = T::CreateOperations();
            // std::cout << "added" << std::endl;
            return ck::host::Transform(
                ops, [](const auto& op) { return op.ToSolution().ToTemplateString(); });
        };
        m.at(name)();
        // std::cout << "left lambda" << std::endl;
        // std::cout << "size after function: " << inst.size() << std::endl;
        // std::cout << "Address after: " << &inst << std::endl;
        std::string prob = "";
        for(const auto& item : inst.items())
        {
            std::cout << item.key() << "\n";
            for(const auto& val : item.value().items())
            {
                if(val.key() == "instances")
                {
                    prob = val.key();
                }
                std::cout << "  " << val.key() << ": " << val.value() << "\n";
            }
        }
        std::cout << prob << std::endl;

        // add instances
        // TODO: separate problem and tuning parameters to nest further
        std::string prob_spec = "fp16";
        std::cout << "starting" << std::endl;
        for(int x = 0; x < m[name]().size(); x++)
        {
            std::string tmp                   = std::to_string(x);
            inst["instances"][prob_spec][tmp] = m[name]()[x];
        }

        // traits (other information)

        out << std::setw(4) << inst;
    }

    // add instance
    // TODO: separate problem and tuning parameters to nest further
    /**data["instances"] = nlohmann::json::object();
    for(int x = 0; x < m[name]().size(); x++)
    {
           std::string tmp        = std::to_string(x);
           data["instances"][tmp] = m[name]()[x];
    }**/

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

    // for(auto name : args)
    //  std::cout << e.Emit(name) << std::endl;

    return 0;
}
