
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "../parse/include/op.hpp"
#include "../parse/include/op_conv.hpp"
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
placeholder - #include <iostream>
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
        inst["headers"] = inc;

        // prologue and epilogue TODO: change names to CK symbols and make specific
        inst["fusion"] = {{"base", R"(struct BaseOperator
{
    BaseOperator()                    = default;
    BaseOperator(const BaseOperator&) = default;
    BaseOperator& operator=(const BaseOperator&) = default;

    virtual bool IsSupportedArgument(const BaseArgument*) { return false; }
    virtual std::string GetTypeString() const { return ""; }

    virtual std::string GetTypeIdName() const { return typeid(*this).name(); }

    virtual std::string GetTypeIdHashCode() const
    {
        std::ostringstream oss;

        oss << std::hex << typeid(*this).hash_code();

        return oss.str();
    };

    virtual size_t GetWorkSpaceSize(const BaseArgument*) const { return 0; }

    virtual void SetWorkSpacePointer(BaseArgument* p_arg,
                                     void* p_workspace,
                                     const StreamConfig& = StreamConfig{}) const
    {
        assert(p_arg);
        p_arg->p_workspace_ = p_workspace;
    }

    virtual ~BaseOperator() {}
};)"},
                          {"prologue", "using CDEElementOp = BaseOperator;"},
                          {"epilogue", "using Epilogue = BaseOperator;"}};

        m[name] = [&] {
            auto ops = T::CreateOperations();
            return ck::host::Transform(
                ops, [](const auto& op) { return op.ToSolution().ToTemplateString(); });
        };
        m.at(name)();
        std::string prob = "";
        /**for(const auto& item : inst.items())
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
        }**/

        // add instances
        std::string prob_spec =
            "fp16fp16fp16fp16RowRowRowRow"; // TODO: find a good way to hand in keys for instances
        for(int x = 0; x < m[name]().size(); x++)
        {
            std::string tmp                   = std::to_string(x);
            inst["instances"][prob_spec][tmp] = m[name]()[x];
        }

        // traits (other information)

        out << std::setw(4) << inst;
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
    // e.Register<ck::host::conv::Operation_Conv>("DeviceConv");

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
    //

    ck::host::CKGenOp_Xdl_CShuffle op;
    char buf[10000];
    std::string op_key = CKGenSetOp(op,
                                    ck::host::DataType_fe::Half,
                                    ck::host::DataType_fe::Half,
                                    ck::host::DataType_fe::Half,
                                    ck::host::DataType_fe::Half,
                                    ck::host::Layout_fe::Row,
                                    ck::host::Layout_fe::Row,
                                    ck::host::Layout_fe::Row,
                                    ck::host::Layout_fe::Row,
                                    8,
                                    8,
                                    8);
    // std::cout << op_key << std::endl;
    nlohmann::json data;
    data = ck::host::CKGenGetOpParams();
    // std::cout << "got data" << std::endl;
    // std::cout << "check 1 - retrieving original JSON: "
    //          << data["fusion"]["prologue"].get<std::string>() << std::endl;
    std::string tmp = R"(struct AlphaBetaAdd
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
using CDEElementOp = AlphaBetaAdd;)";
    ck::host::CKGenSetOpFusion(tmp);
    // data = ck::host::CKGenGetOpParams();
    // std::cout << "check 2 - retrieving updated JSON: "
    //          << data["fusion"]["prologue"].get<std::string>() << std::endl;
    CKGenGetBuffer(op, op_key, buf);

    return 0;
}
