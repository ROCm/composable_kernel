#include <iostream>
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
#include "device_reduce_multiblock.hpp"
#include "host_common_util.hpp"

#include "reduction_enums.hpp"
#include "reduction_operator_mapping.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

using InDataType  = ck::half_t;
using OutDataType = float;
using AccDataType = float;

// for NHWC layer-norm calculation of mean and meansquare
constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;

constexpr bool PropagateNan = false;
constexpr bool OutputIndex  = false;

constexpr InMemoryDataOperationEnum OutMemoryDataOperation = InMemoryDataOperationEnum::AtomicAdd;

using ReduceOperation_Mean = ck::reduce::Add<AccDataType>;
using InElementwiseOperation_Mean =
    ck::tensor_operation::element_wise::UnaryIdentic<AccDataType, AccDataType, false>;
using AccElementwiseOperation_Mean =
    ck::tensor_operation::element_wise::UnaryIdentic<AccDataType, AccDataType, true>;

using ReduceOperation_Meansquare = ck::reduce::Add<AccDataType>;
using InElementwiseOperation_Meansquare =
    ck::tensor_operation::element_wise::UnarySquare<AccDataType, AccDataType, false>;
using AccElementwiseOperation_Meansquare =
    ck::tensor_operation::element_wise::UnaryIdentic<AccDataType, AccDataType, true>;

using DeviceReduceInstance_Mean = DeviceReduceMultiBlock<InDataType,
                                                         AccDataType,
                                                         OutDataType,
                                                         Rank,
                                                         NumReduceDim,
                                                         ReduceOperation_Mean,
                                                         InElementwiseOperation_Mean,
                                                         AccElementwiseOperation_Mean,
                                                         OutMemoryDataOperation,
                                                         PropagateNan,
                                                         OutputIndex,
                                                         false, // HaveIndexInputIfOutputIndex
                                                         256,
                                                         16,
                                                         16,
                                                         1,
                                                         1,
                                                         1, // InSrcVectorDim
                                                         1,
                                                         1>;

using DeviceReduceInstance_Meansquare = DeviceReduceMultiBlock<InDataType,
                                                               AccDataType,
                                                               OutDataType,
                                                               Rank,
                                                               NumReduceDim,
                                                               ReduceOperation_Meansquare,
                                                               InElementwiseOperation_Meansquare,
                                                               AccElementwiseOperation_Meansquare,
                                                               OutMemoryDataOperation,
                                                               PropagateNan,
                                                               OutputIndex,
                                                               false, // HaveIndexInputIfOutputIndex
                                                               256,
                                                               16,
                                                               16,
                                                               1,
                                                               1,
                                                               1, // InSrcVectorDim
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
    std::vector<size_t> inLengths = {512, 28, 28, 256};
    size_t n, h, w, c;

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    public:
    SimpleAppArgs()
    {
        n = inLengths[0];
        h = inLengths[1];
        w = inLengths[2];
        c = inLengths[3];
    };

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
                if(inLengths.size() != Rank)
                    throw std::runtime_error(
                        "Invalid option format! The number of integers is incorrect!");

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

        n = inLengths[0];
        h = inLengths[1];
        w = inLengths[2];
        c = inLengths[3];

        return (0);
    };
};

template <typename InDataType, typename OutDataType, typename AccDataType>
static void mean_meansquare_host(const Tensor<InDataType>& in,
                                 Tensor<OutDataType>& mean_ref,
                                 Tensor<OutDataType>& meansquare_ref,
                                 size_t n,
                                 size_t h,
                                 size_t w,
                                 size_t c)

{
    auto thread_reduce_func = [&](auto iN) {
        AccDataType mean       = type_convert<AccDataType>(0.0f);
        AccDataType meansquare = type_convert<AccDataType>(0.0f);

        // compute mean, meanquare, variance, invVariance
        for(std::size_t iH = 0; iH < h; iH++)
        {
            for(std::size_t iW = 0; iW < w; iW++)
            {
                for(std::size_t iC = 0; iC < c; iC++)
                {
                    AccDataType curr_value = type_convert<AccDataType>(in(iN, iH, iW, iC));

                    mean += curr_value;
                    meansquare += curr_value * curr_value;
                };
            }
        };

        mean       = mean / (h * w * c);
        meansquare = meansquare / (h * w * c);

        mean_ref(iN)       = mean;
        meansquare_ref(iN) = meansquare;
    };

    std::size_t num_thread      = std::thread::hardware_concurrency();
    std::size_t work_per_thread = (n + num_thread - 1) / num_thread;

    std::vector<joinable_thread> threads(num_thread);

    for(std::size_t it = 0; it < num_thread; it++)
    {
        std::size_t iN_begin = it * work_per_thread;
        std::size_t iN_end   = std::min(static_cast<size_t>((it + 1) * work_per_thread), n);

        auto f = [=] {
            for(std::size_t iN = iN_begin; iN < iN_end; iN++)
            {
                thread_reduce_func(iN);
            }
        };

        threads[it] = joinable_thread(f);
    }
};

int main(int argc, char* argv[])
{
    // layer-norm calculates mean and meansquare by reducing [N, H, W, C] to [N]
    const std::vector<int> reduceDims{1, 2, 3};
    const std::vector<int> invariantDims{0};

    SimpleAppArgs args;

    if(argc > 1)
    {
        if(args.processArgs(argc, argv) < 0)
            return (-1);
    };

    Tensor<InDataType> in(args.inLengths);

    std::vector<size_t> outLengths{args.n};

    Tensor<OutDataType> mean_ref(outLengths);
    Tensor<OutDataType> mean(outLengths);
    Tensor<OutDataType> meansquare_ref(outLengths);
    Tensor<OutDataType> meansquare(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = mean.mDesc.GetStrides();

    size_t invariant_total_length = args.n;
    size_t reduce_total_length    = args.h * args.w * args.c;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    std::size_t num_thread = 1;

    if(args.do_verification)
    {
        switch(args.init_method)
        {
        case 0: break;
        case 1: in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread); break;
        case 2: in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread); break;
        default: in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
        }
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpace());
    DeviceMem mean_dev(sizeof(OutDataType) * mean.mDesc.GetElementSpace());
    DeviceMem meansquare_dev(sizeof(OutDataType) * meansquare.mDesc.GetElementSpace());

    in_dev.ToDevice(in.mData.data());

    if(args.do_verification)
    {
        mean_meansquare_host<InDataType, AccDataType, OutDataType>(
            in, mean_ref, meansquare_ref, args.n, args.h, args.w, args.c);
    };

    std::vector<ck::index_t> i_inLengths;
    std::vector<ck::index_t> i_inStrides;
    std::vector<ck::index_t> i_outLengths;
    std::vector<ck::index_t> i_outStrides;

    i_inLengths.assign(args.inLengths.begin(), args.inLengths.end());
    i_inStrides.assign(inStrides.begin(), inStrides.end());
    i_outLengths.assign(outLengths.begin(), outLengths.end());
    i_outStrides.assign(outStrides.begin(), outStrides.end());

    auto reduce_1 = DeviceReduceInstance_Mean{};
    auto reduce_2 = DeviceReduceInstance_Meansquare{};

    auto argument_ptr_1 = reduce_1.MakeArgumentPointer(
        i_inLengths,
        i_inStrides,
        i_outLengths,
        i_outStrides,
        reduceDims,
        alpha,
        beta,
        in_dev.GetDeviceBuffer(),
        nullptr,
        mean_dev.GetDeviceBuffer(),
        nullptr,
        InElementwiseOperation_Mean{},
        AccElementwiseOperation_Mean{static_cast<int32_t>(reduce_total_length)});

    auto argument_ptr_2 = reduce_2.MakeArgumentPointer(
        i_inLengths,
        i_inStrides,
        i_outLengths,
        i_outStrides,
        reduceDims,
        alpha,
        beta,
        in_dev.GetDeviceBuffer(),
        nullptr,
        meansquare_dev.GetDeviceBuffer(),
        nullptr,
        InElementwiseOperation_Meansquare{},
        AccElementwiseOperation_Meansquare{static_cast<int32_t>(reduce_total_length)});

    if(!reduce_1.IsSupportedArgument(argument_ptr_1.get()) ||
       !reduce_2.IsSupportedArgument(argument_ptr_2.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
        return (-1);
    };

    std::string reduce_name_1 = reduce_1.GetTypeString();
    std::string reduce_name_2 = reduce_2.GetTypeString();

    auto invoker_ptr_1 = reduce_1.MakeInvokerPointer();
    auto invoker_ptr_2 = reduce_2.MakeInvokerPointer();

    float avg_time = 0.0f;

    avg_time += invoker_ptr_1->Run(argument_ptr_1.get(), StreamConfig{nullptr, args.time_kernel});
    avg_time += invoker_ptr_2->Run(argument_ptr_2.get(), StreamConfig{nullptr, args.time_kernel});

    std::size_t num_bytes = invariant_total_length * reduce_total_length * sizeof(InDataType) +
                            invariant_total_length * sizeof(OutDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name_1
              << " + " << reduce_name_2 << std::endl;

    bool pass = true;

    if(args.do_verification)
    {
        mean_dev.FromDevice(mean.mData.data());
        meansquare_dev.FromDevice(meansquare.mData.data());
        pass = pass && ck::utils::check_err(mean.mData, mean_ref.mData);
        pass = pass && ck::utils::check_err(meansquare.mData, meansquare_ref.mData);
    };

    return (pass ? 0 : 1);
}
