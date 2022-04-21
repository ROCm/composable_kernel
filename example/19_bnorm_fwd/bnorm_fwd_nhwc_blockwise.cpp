#include <iostream>
#include <fstream>
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
#include "host_reduce_util.hpp"
#include "device_tensor.hpp"
#include "reference_bnorm_fwd_nhwc_c.hpp"
#include "device_bnorm_fwd_nhwc_c_with_reduce_blockwise.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;
using namespace ck::tensor_operation::host;

using InOutDataType = ck::half_t;
using AccDataType   = float;

static const double exponentialAverageFactor = 0.2;
static const double epsilon                  = std::numeric_limits<AccDataType>::epsilon();

using ReferenceBatchNormFwdInstance =
    ReferenceBatchNormFwd_Input_N_H_W_C_Output_C<InOutDataType, AccDataType>;

using DeviceBatchNormFwdInstance = DeviceBatchNormFwd_Input_N_H_W_C_Output_C_With_Reduce_Blockwise<
    InOutDataType,
    AccDataType,
    256, // BlockSize
    8,   // MThreadClusterSize,
    32,  // KThreadClusterSize,
    1,   // MThreadSliceSize,
    1,   // KThreadSliceSize,
    1,   // InOutVectorSize,
    1>;  // ScaleBiasMeanVarVectorSize

static struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class SimpleAppArg
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
    std::vector<size_t> inOutLengths;

    bool do_verification = false;

    bool saveMeanAndInvVariance;
    bool updateMovingAverage;

    int init_method = 1;
    int nrepeat     = 5;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inOutLengths or -D, comma separated list of input tensor dimension "
                     "lengths, must have 4 integers for nhwc"
                  << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the batch-normalization "
                     "result by "
                     "comparing with the host-based batch-normalization"
                  << std::endl;
        std::cout << "Arg1 -- 1/0 to indicate whether to save the calculated mean and invVariance"
                  << std::endl;
        std::cout << "Arg2 -- 1/0 to indicate whether to update the moving average of the mean and "
                     "variance"
                  << std::endl;
        std::cout << "Arg3 -- init method used for bnScale and bnBias (0=no init, 1=single integer "
                     "value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
        std::cout << "Arg4 -- number of repeats to run the kernel" << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        unsigned int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:v:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inOutLengths = getTypeValuesFromString<size_t>(optarg);

                if(inOutLengths.size() != 4)
                    throw std::runtime_error(
                        "NHWC tensor layout should have 4 length values specified!");

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

        if(optind + 4 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        saveMeanAndInvVariance = std::atoi(argv[optind++]);
        updateMovingAverage    = std::atoi(argv[optind++]);
        init_method            = std::atoi(argv[optind++]);
        nrepeat                = std::atoi(argv[optind]);

        return (0);
    };
};

int main(int argc, char* argv[])
{
    using ck::to_int_vector;

    SimpleAppArg arg;

    if(arg.processArgs(argc, argv) < 0)
        return (-1);

    std::vector<size_t> scaleBiasMeanVarLengths = {arg.inOutLengths[3]};

    Tensor<InOutDataType> in(arg.inOutLengths);

    Tensor<InOutDataType> out_ref(arg.inOutLengths);
    Tensor<InOutDataType> out(arg.inOutLengths);

    Tensor<AccDataType> bnScale(scaleBiasMeanVarLengths);
    Tensor<AccDataType> bnBias(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultSaveMean_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultSaveInvVariance_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultRunningMean(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultRunningVariance(scaleBiasMeanVarLengths);

    auto inOutStrides            = in.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = bnScale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(arg.do_verification)
    {
        switch(arg.init_method)
        {
        case 0:
            bnScale.GenerateTensorValue(GeneratorTensor_0<AccDataType>{}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_0<AccDataType>{}, num_thread);
            break;
        case 1:
            bnScale.GenerateTensorValue(GeneratorTensor_1<AccDataType>{1}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_1<AccDataType>{0}, num_thread);
            break;
        case 2:
            bnScale.GenerateTensorValue(GeneratorTensor_2<AccDataType>{-5, 5}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_2<AccDataType>{-5, 5}, num_thread);
            break;
        default:
            bnScale.GenerateTensorValue(GeneratorTensor_3<AccDataType>{-5.0f, 5.0f}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_3<AccDataType>{-5.0f, 5.0f}, num_thread);
        }

        const float data_mean    = 0.0f;
        const float data_stddev  = 1.0f;
        const float noise_stddev = 0.0001f;

        // input data in normal distribution
        in.GenerateTensorValue(GeneratorTensor_4<AccDataType>{data_mean, data_stddev}, num_thread);

        if(arg.updateMovingAverage)
        {
            // initial moving mean set to be the mean of the input data with small normal noise
            resultRunningMean.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{data_mean, noise_stddev}, num_thread);

            // initial moving variance set to be the square of the stddev of the input data with
            // small normal noise
            resultRunningVariance.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{data_stddev * data_stddev, noise_stddev},
                num_thread);
        };
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InOutDataType) * in.mDesc.GetElementSpace());
    DeviceMem out_dev(sizeof(InOutDataType) * out.mDesc.GetElementSpace());
    DeviceMem bnScale_dev(sizeof(AccDataType) * bnScale.mDesc.GetElementSpace());
    DeviceMem bnBias_dev(sizeof(AccDataType) * bnBias.mDesc.GetElementSpace());
    DeviceMem resultRunningMean_dev(sizeof(AccDataType) *
                                    resultRunningMean.mDesc.GetElementSpace());
    DeviceMem resultRunningVariance_dev(sizeof(AccDataType) *
                                        resultRunningVariance.mDesc.GetElementSpace());
    DeviceMem resultSaveMean_dev(sizeof(AccDataType) * resultSaveMean_ref.mDesc.GetElementSpace());
    DeviceMem resultSaveInvVariance_dev(sizeof(AccDataType) *
                                        resultSaveInvVariance_ref.mDesc.GetElementSpace());

    in_dev.ToDevice(in.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());
    bnBias_dev.ToDevice(bnBias.mData.data());

    if(arg.updateMovingAverage)
    {
        resultRunningMean_dev.ToDevice(resultRunningMean.mData.data());
        resultRunningVariance_dev.ToDevice(resultRunningMean.mData.data());
    };

    const auto i_inOutLengths            = to_int_vector(arg.inOutLengths);
    const auto i_inOutStrides            = to_int_vector(inOutStrides);
    const auto i_scaleBiasMeanVarLengths = to_int_vector(scaleBiasMeanVarLengths);
    const auto i_scaleBiasMeanVarStrides = to_int_vector(scaleBiasMeanVarStrides);

    auto batchNormFwd = DeviceBatchNormFwdInstance{};

    auto wsSizeInBytes =
        batchNormFwd.GetWorkspaceSizeInBytes(i_inOutLengths[3], arg.saveMeanAndInvVariance);

    DeviceMem ws_dev(wsSizeInBytes);

    auto argument_ptr = batchNormFwd.MakeArgumentPointer(
        i_inOutLengths,
        i_inOutStrides,
        i_inOutLengths,
        i_inOutStrides,
        i_scaleBiasMeanVarLengths,
        i_scaleBiasMeanVarStrides,
        1.0f, // alpha
        0.0f, // beta
        in_dev.GetDeviceBuffer(),
        out_dev.GetDeviceBuffer(),
        arg.saveMeanAndInvVariance ? nullptr : ws_dev.GetDeviceBuffer(),
        bnScale_dev.GetDeviceBuffer(),
        bnBias_dev.GetDeviceBuffer(),
        exponentialAverageFactor,
        arg.updateMovingAverage ? resultRunningMean_dev.GetDeviceBuffer() : nullptr,
        arg.updateMovingAverage ? resultRunningVariance_dev.GetDeviceBuffer() : nullptr,
        epsilon,
        arg.saveMeanAndInvVariance ? resultSaveMean_dev.GetDeviceBuffer() : nullptr,
        arg.saveMeanAndInvVariance ? resultSaveInvVariance_dev.GetDeviceBuffer() : nullptr);

    if(!batchNormFwd.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
    };

    std::string batchNormFwd_name = batchNormFwd.GetTypeString();

    auto invoker_ptr = batchNormFwd.MakeInvokerPointer();

    float avg_time = invoker_ptr->Run(argument_ptr.get(), arg.nrepeat);

    std::size_t num_bytes = arg.inOutLengths[0] * arg.inOutLengths[1] * arg.inOutLengths[2] *
                            arg.inOutLengths[3] * sizeof(InOutDataType);

    if(arg.saveMeanAndInvVariance)
        num_bytes += 2 * arg.inOutLengths[3] * sizeof(AccDataType);

    if(arg.updateMovingAverage)
        num_bytes += 2 * arg.inOutLengths[3] * 2 * sizeof(AccDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << batchNormFwd_name
              << std::endl;

    if(arg.do_verification)
    {
        auto batchNormFwd_ref = ReferenceBatchNormFwdInstance{};

        auto argument_ptr_ref = batchNormFwd_ref.MakeArgumentPointer(
            i_inOutLengths,
            i_inOutStrides,
            i_inOutLengths,
            i_inOutStrides,
            i_scaleBiasMeanVarLengths,
            i_scaleBiasMeanVarStrides,
            1.0f, // alpha
            0.0f, // beta
            in.mData.data(),
            out_ref.mData.data(),
            nullptr,
            bnScale.mData.data(),
            bnBias.mData.data(),
            exponentialAverageFactor,
            arg.updateMovingAverage ? resultRunningMean.mData.data() : nullptr,
            arg.updateMovingAverage ? resultRunningVariance.mData.data() : nullptr,
            epsilon,
            arg.saveMeanAndInvVariance ? resultSaveMean_ref.mData.data() : nullptr,
            arg.saveMeanAndInvVariance ? resultSaveInvVariance_ref.mData.data() : nullptr);

        if(!batchNormFwd_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout
                << "The runtime parameters seems not supported by the BatchNorm instance, exiting!"
                << std::endl;
        };

        auto invoker_ptr_ref = batchNormFwd_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get(), 1);

        out_dev.FromDevice(out.mData.data());
        ck::utils::check_err(out_ref.mData, out.mData);
    };
}
