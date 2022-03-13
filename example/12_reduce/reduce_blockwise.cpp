#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_reduce_blockwise.hpp"
#include "host_reduce_util.hpp"
#include "host_generic_reduction.hpp"

#include "reduction_enums.hpp"
#include "reduction_operator_mapping.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

using InDataType  = half_float::half;
using OutDataType = half_float::half;
using AccDataType = float;

using kInDataType  = ck::half_t;
using kOutDataType = ck::half_t;
using kAccDataType = float;

constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;

constexpr ReduceTensorOp_t ReduceOpId = ReduceTensorOp_t::NORM2;
constexpr NanPropagation_t NanOpt     = NanPropagation_t::PROPAGATE_NAN;
constexpr bool PropagateNan = (NanOpt == NanPropagation_t::NOT_PROPAGATE_NAN) ? false : true;
constexpr ReduceTensorIndices_t IndicesOpt = ReduceTensorIndices_t::NO_INDICES;

using ReduceOperation = typename reduce_binary_operator<AccDataType, ReduceOpId>::opType;
using InElementwiseOperation =
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::InElementwiseOperation;
using AccElementwiseOperation =
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::AccElementwiseOperation;

using DeviceReduceInstance = DeviceReduceBlockWise<kInDataType,
                                                   kAccDataType,
                                                   kOutDataType,
                                                   Rank,
                                                   NumReduceDim,
                                                   ReduceOperation,
                                                   InElementwiseOperation,
                                                   AccElementwiseOperation,
                                                   PropagateNan,
                                                   false,
                                                   256,
                                                   4,
                                                   64,
                                                   1,
                                                   1,
                                                   0,
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

int main(int argc, char* argv[])
{
    using namespace ck::host_reduce;

    const std::vector<int> reduceDims{0, 1, 2};
    const std::vector<int> invariantDims{3};

    SimpleAppArgs args;

    if(args.processArgs(argc, argv) < 0)
        return (-1);

    constexpr bool op_support_indices =
        (ReduceOpId == ReduceTensorOp_t::MIN || ReduceOpId == ReduceTensorOp_t::MAX ||
         ReduceOpId == ReduceTensorOp_t::AMAX);

    constexpr bool NeedIndices =
        (op_support_indices && (IndicesOpt != ReduceTensorIndices_t::NO_INDICES));

    // if input is half type, no reason to use float for indiced reduction operation and must use
    // float for non-indiced reduction operation for accuracy
    constexpr bool invalid_reduce_1 =
        std::is_same<InDataType, ck::half_t>::value &&
        ((!op_support_indices && !std::is_same<AccDataType, float>::value) ||
         (op_support_indices && !std::is_same<AccDataType, ck::half_t>::value));

    // if input is float type, no reason to use double for indiced reduction operation
    constexpr bool invalid_reduce_2 =
        std::is_same<InDataType, float>::value &&
        (op_support_indices && !std::is_same<AccDataType, float>::value);

    // indices option can only be used when it is really needed
    constexpr bool invalid_reduce_3 =
        (!op_support_indices && IndicesOpt != ReduceTensorIndices_t::NO_INDICES);

    constexpr bool invalid_reduce = (invalid_reduce_1 || invalid_reduce_2 || invalid_reduce_3);

    if constexpr(invalid_reduce)
        std::cout << "Reduction setting is not supported, exiting!" << std::endl;

    Tensor<InDataType> in(args.inLengths);

    std::vector<size_t> outLengths;

    if(invariantDims.empty())
        outLengths.push_back(1);
    else
        for(auto dim : invariantDims)
            outLengths.push_back(args.inLengths[dim]);

    Tensor<OutDataType> out_ref(outLengths);
    Tensor<OutDataType> out(outLengths);
    Tensor<int> out_indices_ref(outLengths);
    Tensor<int> out_indices(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    size_t invariant_total_length = out.mDesc.GetElementSize();
    size_t reduce_total_length    = in.mDesc.GetElementSize() / invariant_total_length;

    float alpha = args.scales[0];
    float beta  = args.scales[1];

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(args.do_verification)
    {
        switch(args.init_method)
        {
        case 0:
            in.GenerateTensorValue(GeneratorTensor_1<InDataType>{}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<InDataType>{}, num_thread);
            break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_2<InDataType>{1, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<InDataType>{1, 5}, num_thread);
        }

        if(beta != 0.0f)
            for(size_t i = 0; i < out_ref.mDesc.GetElementSpace(); i++)
                out.mData[i] = out_ref.mData[i];
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpace());
    DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpace());

    in_dev.ToDevice(in.mData.data());

    if(beta != 0.0f)
        out_dev.ToDevice(out.mData.data());

    size_t indicesSizeInBytes = NeedIndices ? out.mDesc.GetElementSize() * sizeof(int) : 0;

    DeviceMem out_indices_dev(indicesSizeInBytes);

    if(args.do_verification)
    {
        ReductionHost<InDataType, AccDataType, OutDataType, ReduceOpId, PropagateNan, NeedIndices>
            hostReduce(in.mDesc, out_ref.mDesc, invariantDims, reduceDims);

        hostReduce.Run(
            alpha, in.mData.data(), beta, out_ref.mData.data(), out_indices_ref.mData.data());
    };

    const auto i_inLengths  = to_int_vector(args.inLengths);
    const auto i_inStrides  = to_int_vector(inStrides);
    const auto i_outLengths = to_int_vector(outLengths);
    const auto i_outStrides = to_int_vector(outStrides);

    auto reduce = DeviceReduceInstance{};

    auto wsSizeInBytes = reduce.GetWorkspaceSizeInBytes(i_inLengths, reduceDims);

    DeviceMem ws_dev(wsSizeInBytes);

    auto argument_ptr =
        reduce.MakeArgumentPointer(i_inLengths,
                                   i_inStrides,
                                   i_outLengths,
                                   i_outStrides,
                                   reduceDims,
                                   alpha,
                                   beta,
                                   in_dev.GetDeviceBuffer(),
                                   out_dev.GetDeviceBuffer(),
                                   out_indices_dev.GetDeviceBuffer(),
                                   ws_dev.GetDeviceBuffer(),
                                   InElementwiseOperation{static_cast<int>(reduce_total_length)},
                                   AccElementwiseOperation{static_cast<int>(reduce_total_length)});

    if(!reduce.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
    };

    std::string reduce_name = reduce.GetTypeString();

    auto invoker_ptr = reduce.MakeInvokerPointer();

    float avg_time = invoker_ptr->Run(argument_ptr.get(), args.nrepeat);

    std::size_t num_bytes = invariant_total_length * reduce_total_length * sizeof(InDataType) +
                            invariant_total_length * sizeof(OutDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
              << std::endl;

    if(args.do_verification)
    {
        out_dev.FromDevice(out.mData.data());
        check_error(out_ref, out);

        if(NeedIndices)
        {
            out_indices_dev.FromDevice(out_indices.mData.data());
            check_indices(out_indices_ref, out_indices);
        };
    };
}
