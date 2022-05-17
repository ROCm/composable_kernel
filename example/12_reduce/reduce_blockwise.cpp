#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "check_err.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_reduce_blockwise.hpp"
#include "host_common_util.hpp"
#include "host_reduction.hpp"

#include "reduction_enums.hpp"
#include "reduction_operator_mapping.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

using InDataType  = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;

constexpr ReduceTensorOp ReduceOpId = ReduceTensorOp::NORM2;
constexpr bool PropagateNan         = true;
constexpr bool OutputIndex          = false;

using ReduceOperation = typename reduce_binary_operator<AccDataType, ReduceOpId>::opType;
using InElementwiseOperation =
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::InElementwiseOperation;
using AccElementwiseOperation =
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::AccElementwiseOperation;

using DeviceReduceInstance = DeviceReduceBlockWise<InDataType,
                                                   AccDataType,
                                                   OutDataType,
                                                   Rank,
                                                   NumReduceDim,
                                                   ReduceOperation,
                                                   InElementwiseOperation,
                                                   AccElementwiseOperation,
                                                   PropagateNan,
                                                   OutputIndex,
                                                   false, // HaveIndexInputIfOutputIndex
                                                   256,
                                                   4,
                                                   64,
                                                   1,
                                                   1,
                                                   0,
                                                   1,
                                                   1>;

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class SimpleAppArgs
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inLengths = {16, 64, 32, 960};
    std::vector<float> scales     = {1.0f, 0.0f};

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
                  << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the reduction result by "
                     "comparing with the host-based reduction"
                  << std::endl;
        std::cout << "Arg1 -- init method (0=no init, 1=single integer value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
        std::cout << "Arg2 -- time kernel (0=no, 1=yes)" << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:v:l:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inLengths = getTypeValuesFromString<size_t>(optarg);
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
        time_kernel = static_cast<bool>(std::atoi(argv[optind]));

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
    using ck::host_common::to_int_vector;

    const std::vector<int> reduceDims{0, 1, 2};
    const std::vector<int> invariantDims{3};

    SimpleAppArgs args;

    if(argc > 1)
    {
        if(args.processArgs(argc, argv) < 0)
            return (-1);
    };

    constexpr bool op_support_indices =
        (ReduceOpId == ReduceTensorOp::MIN || ReduceOpId == ReduceTensorOp::MAX ||
         ReduceOpId == ReduceTensorOp::AMAX);

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
    constexpr bool invalid_reduce_3 = (!op_support_indices && OutputIndex);

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

    std::size_t num_thread = 1;

    if(args.do_verification)
    {
        switch(args.init_method)
        {
        case 0: break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
            break;
        case 2:
            in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
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

    size_t indicesSizeInBytes = OutputIndex ? out.mDesc.GetElementSize() * sizeof(int32_t) : 0;

    DeviceMem out_index_dev(indicesSizeInBytes);

    if(args.do_verification)
    {
        ReductionHost<InDataType,
                      AccDataType,
                      OutDataType,
                      ReduceOpId,
                      Rank,
                      NumReduceDim,
                      PropagateNan,
                      OutputIndex>
            hostReduce(in.mDesc, out_ref.mDesc, invariantDims, reduceDims);

        hostReduce.Run(
            alpha, in.mData.data(), beta, out_ref.mData.data(), out_indices_ref.mData.data());
    };

    const auto i_inLengths  = to_int_vector(args.inLengths);
    const auto i_inStrides  = to_int_vector(inStrides);
    const auto i_outLengths = to_int_vector(outLengths);
    const auto i_outStrides = to_int_vector(outStrides);

    auto reduce = DeviceReduceInstance{};

    auto argument_ptr = reduce.MakeArgumentPointer(
        i_inLengths,
        i_inStrides,
        i_outLengths,
        i_outStrides,
        reduceDims,
        alpha,
        beta,
        in_dev.GetDeviceBuffer(),
        nullptr,
        out_dev.GetDeviceBuffer(),
        out_index_dev.GetDeviceBuffer(),
        InElementwiseOperation{static_cast<int32_t>(reduce_total_length)},
        AccElementwiseOperation{static_cast<int32_t>(reduce_total_length)});

    if(!reduce.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
    };

    std::string reduce_name = reduce.GetTypeString();

    auto invoker_ptr = reduce.MakeInvokerPointer();

    float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, args.time_kernel});

    std::size_t num_bytes = invariant_total_length * reduce_total_length * sizeof(InDataType) +
                            invariant_total_length * sizeof(OutDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
              << std::endl;

    if(args.do_verification)
    {
        out_dev.FromDevice(out.mData.data());
        ck::utils::check_err(out.mData, out_ref.mData);

        if(OutputIndex)
        {
            out_index_dev.FromDevice(out_indices.mData.data());
            ck::utils::check_err(out_indices.mData, out_indices_ref.mData);
        };
    };
}
