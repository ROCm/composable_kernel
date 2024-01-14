
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

    std::string run_function()
    {
        std::string fcn = R"(
run(int argc, char* argv[])
{
    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<${ADataType}> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ${ALayout}{}));
    Tensor<${BDataType}> b_k_n(f_host_tensor_descriptor(K, N, StrideB, ${BLayout}{}));
    Tensor<${DDataType}> d_m_n(f_host_tensor_descriptor(M, N, StrideD, ${DLayout}{}));
    Tensor<${EDataType}> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ${ELayout}{}));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<${ADataType}>{0.0, 1.0});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<${BDataType}>{-0.5, 0.5});
    d_m_n.GenerateTensorValue(GeneratorTensor_3<${DDataType}>{-0.5, 0.5});

    DeviceMem a_device_buf(sizeof(${ADataType}) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(${BDataType}) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(${DDataType}) * d_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(${EDataType}) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    d_device_buf.ToDevice(d_m_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{alpha, beta};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                            b_device_buf.GetDeviceBuffer(),
                            std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                            e_device_buf.GetDeviceBuffer(),
                            M,
                            N,
                            K,
                            StrideA,
                            StrideB,
                            std::array<ck::index_t, 1>{StrideD},
                            StrideE,
                            a_element_op,
                            b_element_op,
                            cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());
    return 0;
}
extern "C" {
    bool run_op(int argc, char* argv[])
	        {
			        ProblemSize problem_size;
				        ExecutionConfig config;

					        return !parse_cmd_args(argc, argv, problem_size, config) || run(problem_size, config);
						    }
    })";
        return fcn;
    }

    template <class T>
    void Register(const std::string& name)
    {
        std::ofstream out("./op_inst.json");
        // populate json

        // include section
        inst["include"] = "#include <string>";

        // prologue and epilogue
        inst["fusion"]      = {{"prologue", "using Prologue = BaseOp;"},
                          {"epilogue", "using Epilogue = BaseOp;"}};
        std::string run_fcn = run_function();
        inst["run"]         = run_fcn;

        //std::cout << "made it here" << std::endl;
        std::cout << inst.type_name() << std::endl;
        std::cout << "size before function: " << inst.size() << std::endl;
        //std::cout << "Address before: " << &inst << std::endl;
        //std::cout << inst.dump() << std::endl;

        m[name] = [&] {
            auto ops = T::CreateOperations(inst);
            std::cout << "added" << std::endl;
            return ck::host::Transform(
                ops, [](const auto& op) { return op.ToSolution().ToTemplateString(); });
        };
        m.at(name)();
        std::cout << "left lambda" << std::endl;
        std::cout << "size after function: " << inst.size() << std::endl;
        //std::cout << "Address after: " << &inst << std::endl;
	std::string prob = "";
        for(const auto& item : inst.items())
        {
            std::cout << item.key() << "\n";
            for(const auto& val : item.value().items())
            {
		    if(val.key() == "instances"){
			    prob = val.key();
	            }
                std::cout << "  " << val.key() << ": " << val.value() << "\n";
            }
        }
	std::cout << prob << std::endl;

        // add instances
        // TODO: separate problem and tuning parameters to nest further
        // std::string prob_spec = inst["instances"].get<std::string>();
        // std::cout << prob_spec << std::endl;
        std::string prob_spec = "fp16";
        std::cout << "starting" << std::endl;
        // inst["instance"][prob_spec] = nlohmann::json::object();
        for(int x = 0; x < m[name]().size(); x++)
        {
            std::string tmp                   = std::to_string(x);
            inst["instances"][prob_spec][tmp] = m[name]()[x];
        }
        /**for(int x = 0; x < m[name]().size(); x++)
        {
            std::string tmp        = std::to_string(x);
            inst["instances"][prob_spec][tmp] = m[name]()[x];
        }**/

        // the run function

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
