#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_reduce_blockwise.hpp"
#include "host_generic_reduction.hpp"

#include "reduction_enums.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

using inType   = ck::half_t;
using outType  = ck::half_t;
using compType = float;

constexpr int rank  = 4;
using toReduceDims_ = ck::Sequence<0, 1, 2>;

constexpr ReduceTensorOp_t reduceOp        = ReduceTensorOp_t::AVG;
constexpr NanPropagation_t nanOpt          = NanPropagation_t::NOT_PROPAGATE_NAN;
constexpr ReduceTensorIndices_t indicesOpt = ReduceTensorIndices_t::NO_INDICES;

using DeviceReduceInstance = DeviceReduceBlockWise<inType,
                                                   compType,
                                                   outType,
                                                   rank,
                                                   toReduceDims_,
                                                   reduceOp,
                                                   nanOpt,
                                                   indicesOpt,
                                                   256,
                                                   4,
                                                   64,
                                                   true,
                                                   1,
                                                   1>;

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"scales", required_argument, nullptr, 'S'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class SimpleAppArgs
{
    template <typename T>
    static T getSingleValueFromString(const std::string& valueStr)
    {
        std::istringstream iss(valueStr);

        T ret;

        iss >> ret;

        return (ret);
    };

    template <typename T>
    static std::vector<T> getTypeValuesFromString(const char* cstr_values)
    {
        std::string valuesStr(cstr_values);

        std::vector<T> values;
        std::size_t pos = 0;
        std::size_t new_pos;

        new_pos = valuesStr.find(',', pos);
        while(new_pos != std::string::npos)
        {
            const std::string sliceStr = valuesStr.substr(pos, new_pos - pos);

            T val = getSingleValueFromString<T>(sliceStr);

            values.push_back(val);

            pos     = new_pos + 1;
            new_pos = valuesStr.find(',', pos);
        };

        std::string sliceStr = valuesStr.substr(pos);
        T val                = getSingleValueFromString<T>(sliceStr);

        values.push_back(val);

        return (values);
    };

    private:
    int option_index = 0;

    public:
    std::vector<size_t> inLengths;
    std::vector<float> scales;

    bool do_verification = false;

    int init_method = 1;
    int nrepeat     = 5;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
                  << std::endl;
        std::cout << "--scales or -S, comma separated two float values for alpha and beta"
                  << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the reduction result by "
                     "comparing with the host-based reduction"
                  << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        unsigned int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:S:v:l:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inLengths = getTypeValuesFromString<size_t>(optarg);
                break;
            case 'S':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                scales = getTypeValuesFromString<float>(optarg);
                break;
            case 'v':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_verification = static_cast<bool>(std::atoi(optarg));
                break;
            case '?':
                if(std::string(long_options[option_index].name) == "help")
                {
                    show_usage(argv[0]);
                    return (-1);
                };
                break;
            default: show_usage(argv[0]); return (-1);
            };
        };

        if(optind + 2 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        init_method = std::atoi(argv[optind++]);
        nrepeat     = std::atoi(argv[optind]);

        if(scales.empty())
        {
            scales.push_back(1.0f);
            scales.push_back(0.0f);
        };

        return (0);
    };
};

template <int rank, typename toReduceDims>
static std::vector<int> get_toReduce_dims()
{
    std::vector<int> resDims;

    static_for<0, toReduceDims::Size(), 1>{}(
        [&](auto i) { resDims.push_back(toReduceDims::At(i)); });

    return (resDims);
};

template <int rank, typename toReduceDims>
static std::vector<int> get_invariant_dims()
{
    std::vector<int> resDims;
    unsigned int incFlag = 0;

    static_for<0, toReduceDims::Size(), 1>{}(
        [&](auto i) { incFlag = incFlag | (0x1 << toReduceDims::At(i)); });

    for(int dim = 0; dim < rank; dim++)
    {
        if(incFlag & (0x1 << dim))
            continue;
        resDims.push_back(dim);
    };

    return (resDims);
};

static std::vector<int> to_int_vector(const std::vector<size_t>& inData)
{
    std::vector<int> outData;

    for(auto elem : inData)
        outData.push_back(static_cast<int>(elem));

    return (outData);
};

static void check_indices(const Tensor<int>& ref, const Tensor<int>& result)
{
    bool has_error  = false;
    int error_count = 0;

    for(int i = 0; i < ref.mData.size(); ++i)
    {
        if(ref.mData[i] != result.mData[i])
        {
            std::cerr << std::endl
                      << "Indices different at position " << i << " (ref: " << ref.mData[i]
                      << ", result: " << result.mData[i] << ")" << std::endl;
            has_error = true;
            error_count++;
            if(error_count == 20)
                break;
        };
    }

    if(!has_error)
        std::cout << std::endl << "Indices result is completely acccurate!" << std::endl;
};

int main(int argc, char* argv[])
{
    SimpleAppArgs args;

    if(args.processArgs(argc, argv) < 0)
        return (-1);

    constexpr bool op_support_indices =
        (reduceOp == ReduceTensorOp_t::MIN || reduceOp == ReduceTensorOp_t::MAX ||
         reduceOp == ReduceTensorOp_t::AMAX);

    constexpr bool need_indices =
        (op_support_indices && (indicesOpt != ReduceTensorIndices_t::NO_INDICES));

    // if input is half type, no reason to use float for indiced reduction operation and must use
    // float for non-indiced reduction operation for accuracy
    constexpr bool invalid_reduce_1 =
        std::is_same<inType, ck::half_t>::value &&
        ((!op_support_indices && !std::is_same<compType, float>::value) ||
         (op_support_indices && !std::is_same<compType, ck::half_t>::value));

    // if input is float type, no reason to use double for indiced reduction operation
    constexpr bool invalid_reduce_2 = std::is_same<inType, float>::value &&
                                      (op_support_indices && !std::is_same<compType, float>::value);

    // indices option can only be used when it is really needed
    constexpr bool invalid_reduce_3 =
        (!op_support_indices && indicesOpt != ReduceTensorIndices_t::NO_INDICES);

    constexpr bool invalid_reduce = (invalid_reduce_1 || invalid_reduce_2 || invalid_reduce_3);

    if constexpr(invalid_reduce)
        std::cout << "Reduction setting is not supported, exiting!" << std::endl;

    Tensor<inType> in(args.inLengths);

    const std::vector<int> invariantDims = get_invariant_dims<rank, toReduceDims_>();
    const std::vector<int> toReduceDims  = get_toReduce_dims<rank, toReduceDims_>();

    std::vector<size_t> outLengths;

    if(invariantDims.empty())
        outLengths.push_back(1);
    else
        for(auto dim : invariantDims)
            outLengths.push_back(args.inLengths[dim]);

    Tensor<outType> out_ref(outLengths);
    Tensor<outType> out(outLengths);
    Tensor<int> out_indices_ref(outLengths);
    Tensor<int> out_indices(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    size_t dim0_total_length = out.mDesc.GetElementSize();
    size_t dim1_total_length = in.mDesc.GetElementSize() / dim0_total_length;

    float alpha = args.scales[0];
    float beta  = args.scales[1];

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(args.do_verification)
    {
        switch(args.init_method)
        {
        case 0:
            in.GenerateTensorValue(GeneratorTensor_1<inType>{}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<inType>{}, num_thread);
            break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_2<inType>{-99, 99}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<inType>{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_2<inType>{1, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<inType>{1, 5}, num_thread);
        }

        if(beta != 0.0f)
            for(size_t i = 0; i < out_ref.mDesc.GetElementSpace(); i++)
                out.mData[i] = out_ref.mData[i];
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(inType) * in.mDesc.GetElementSpace());
    DeviceMem out_dev(sizeof(outType) * out.mDesc.GetElementSpace());

    in_dev.ToDevice(in.mData.data());

    if(beta != 0.0f)
        out_dev.ToDevice(out.mData.data());

    size_t indicesSizeInBytes = need_indices ? out.mDesc.GetElementSize() * sizeof(int) : 0;

    DeviceMem out_indices_dev(indicesSizeInBytes);

    if(args.do_verification)
    {
        ReductionHost<inType, compType, outType> hostReduce(
            reduceOp, nanOpt, indicesOpt, in.mDesc, out_ref.mDesc, invariantDims, toReduceDims);

        hostReduce.Run(
            alpha, in.mData.data(), beta, out_ref.mData.data(), out_indices_ref.mData.data());
    };

    const auto i_inLengths  = to_int_vector(args.inLengths);
    const auto i_inStrides  = to_int_vector(inStrides);
    const auto i_outLengths = to_int_vector(outLengths);
    const auto i_outStrides = to_int_vector(outStrides);

    auto reduce = DeviceReduceInstance{};

    auto wsSizeInBytes = reduce.getWorkspaceSize(i_inLengths);

    DeviceMem ws_dev(wsSizeInBytes);

    auto argument_ptr = reduce.MakeArgumentPointer(i_inLengths,
                                                   i_inStrides,
                                                   i_outLengths,
                                                   i_outStrides,
                                                   alpha,
                                                   beta,
                                                   in_dev.GetDeviceBuffer(),
                                                   out_dev.GetDeviceBuffer(),
                                                   out_indices_dev.GetDeviceBuffer(),
                                                   ws_dev.GetDeviceBuffer());

    if(!reduce.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
    };

    std::string reduce_name = reduce.GetTypeString();

    auto invoker_ptr = reduce.MakeInvokerPointer();

    float avg_time = invoker_ptr->Run(argument_ptr.get(), args.nrepeat);

    std::size_t num_bytes = dim0_total_length * dim1_total_length * sizeof(inType) +
                            dim0_total_length * sizeof(outType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
              << std::endl;

    if(args.do_verification)
    {
        out_dev.FromDevice(out.mData.data());
        check_error(out_ref, out);

        if(need_indices)
        {
            out_indices_dev.FromDevice(out_indices.mData.data());
            check_indices(out_indices_ref, out_indices);
        };
    };
}
