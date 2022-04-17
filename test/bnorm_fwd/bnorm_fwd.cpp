#include <iostream>
#include <fstream>
#include <getopt.h>
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "check_err.hpp"
#include "device_bnorm_fwd.hpp"
#include "device_bnorm_fwd_instance.hpp"
#include "reference_bnorm_fwd_nhwc_c.hpp"
#include "host_common_util.hpp"

using namespace ck;

namespace {

template <typename InOutDataType, typename AccDataType>
bool test_bnorm_fwd_impl(int init_method,
                         const std::vector<size_t> inOutLengths,
                         const std::vector<size_t> scaleBiasMeanVarLengths,
                         bool saveMeanAndInvVariance,
                         bool updateMovingAverage,
                         double epsilon,
                         double exponentialAverageFactor,
                         float alpha,
                         float beta)
{
    using namespace ck::tensor_operation::device;
    using namespace ck::tensor_operation::device::device_bnorm_fwd_instance;
    using namespace ck::tensor_operation::host;
    using ck::host_common::to_int_vector;

    Tensor<InOutDataType> in(inOutLengths);

    Tensor<InOutDataType> out_ref(inOutLengths);
    Tensor<InOutDataType> out(inOutLengths);

    Tensor<AccDataType> bnScale(scaleBiasMeanVarLengths);
    Tensor<AccDataType> bnBias(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultSaveMean_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultSaveInvVariance_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultRunningMean(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultRunningVariance(scaleBiasMeanVarLengths);

    auto inOutStrides            = in.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = bnScale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    switch(init_method)
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
    };

    const float data_mean    = 0.0f;
    const float data_stddev  = 1.0f;
    const float noise_stddev = 0.0001f;

    // input data in normal distribution
    in.GenerateTensorValue(GeneratorTensor_4<InOutDataType>{data_mean, data_stddev}, num_thread);

    if(updateMovingAverage)
    {
        // initial moving mean set to be the mean of the input data with small normal noise
        resultRunningMean.GenerateTensorValue(
            GeneratorTensor_4<AccDataType>{data_mean, noise_stddev}, num_thread);

        // initial moving variance set to be the square of the stddev of the input data with
        // small normal noise
        resultRunningVariance.GenerateTensorValue(
            GeneratorTensor_4<AccDataType>{data_stddev * data_stddev, noise_stddev}, num_thread);
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

    if(beta != 0.0f)
        out_dev.ToDevice(out.mData.data());

    if(updateMovingAverage)
    {
        resultRunningMean_dev.ToDevice(resultRunningMean.mData.data());
        resultRunningVariance_dev.ToDevice(resultRunningMean.mData.data());
    };

    std::vector<DeviceBatchNormFwdPtr> bnorm_fwd_ptrs;

    add_device_bnorm_fwd_with_reduce_blockwise_instance<InOutDataType, AccDataType>(bnorm_fwd_ptrs);
    add_device_bnorm_fwd_with_reduce_multiblock_instance<InOutDataType, AccDataType>(
        bnorm_fwd_ptrs);

    const auto i_inOutLengths            = to_int_vector(inOutLengths);
    const auto i_inOutStrides            = to_int_vector(inOutStrides);
    const auto i_scaleBiasMeanVarLengths = to_int_vector(scaleBiasMeanVarLengths);
    const auto i_scaleBiasMeanVarStrides = to_int_vector(scaleBiasMeanVarStrides);

    bool result = true;

    for(auto& bnorm_fwd_ptr : bnorm_fwd_ptrs)
    {
        auto wsSizeInBytes =
            bnorm_fwd_ptr->GetWorkspaceSizeInBytes(i_inOutLengths[3], saveMeanAndInvVariance);

        DeviceMem ws_dev(wsSizeInBytes);

        auto argument_ptr = bnorm_fwd_ptr->MakeArgumentPointer(
            i_inOutLengths,
            i_inOutStrides,
            i_inOutLengths,
            i_inOutStrides,
            i_scaleBiasMeanVarLengths,
            i_scaleBiasMeanVarStrides,
            alpha,
            beta,
            in_dev.GetDeviceBuffer(),
            out_dev.GetDeviceBuffer(),
            saveMeanAndInvVariance ? nullptr : ws_dev.GetDeviceBuffer(),
            bnScale_dev.GetDeviceBuffer(),
            bnBias_dev.GetDeviceBuffer(),
            exponentialAverageFactor,
            updateMovingAverage ? resultRunningMean_dev.GetDeviceBuffer() : nullptr,
            updateMovingAverage ? resultRunningVariance_dev.GetDeviceBuffer() : nullptr,
            epsilon,
            saveMeanAndInvVariance ? resultSaveMean_dev.GetDeviceBuffer() : nullptr,
            saveMeanAndInvVariance ? resultSaveInvVariance_dev.GetDeviceBuffer() : nullptr);

        if(!bnorm_fwd_ptr->IsSupportedArgument(argument_ptr.get()))
            continue;

        std::string bnorm_fwd_name = bnorm_fwd_ptr->GetTypeString();

        auto invoker_ptr = bnorm_fwd_ptr->MakeInvokerPointer();

        (void)invoker_ptr->Run(argument_ptr.get());

        using ReferenceBatchNormFwdInstance =
            ReferenceBatchNormFwd_Input_N_H_W_C_Output_C<InOutDataType, AccDataType>;

        auto bnorm_fwd_ref = ReferenceBatchNormFwdInstance{};

        auto argument_ptr_ref = bnorm_fwd_ref.MakeArgumentPointer(
            i_inOutLengths,
            i_inOutStrides,
            i_inOutLengths,
            i_inOutStrides,
            i_scaleBiasMeanVarLengths,
            i_scaleBiasMeanVarStrides,
            alpha,
            beta,
            in.mData.data(),
            out_ref.mData.data(),
            nullptr,
            bnScale.mData.data(),
            bnBias.mData.data(),
            exponentialAverageFactor,
            updateMovingAverage ? resultRunningMean.mData.data() : nullptr,
            updateMovingAverage ? resultRunningVariance.mData.data() : nullptr,
            epsilon,
            saveMeanAndInvVariance ? resultSaveMean_ref.mData.data() : nullptr,
            saveMeanAndInvVariance ? resultSaveInvVariance_ref.mData.data() : nullptr);

        if(!bnorm_fwd_ref.IsSupportedArgument(argument_ptr_ref.get()))
            continue;

        auto invoker_ptr_ref = bnorm_fwd_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get(), 1);

        out_dev.FromDevice(out.mData.data());

        bool single_result = true;

        if constexpr(std::is_same<InOutDataType, ck::half_t>::value)
        {
            single_result = ck::utils::check_err(
                out.mData, out_ref.mData, "Error: Incorrect results!", 1e-5, 4.0e-3);
        }
        else if constexpr(std::is_same<InOutDataType, float>::value)
        {
            single_result = ck::utils::check_err(
                out.mData, out_ref.mData, "Error: Incorrect results!", 1e-5, 1.4e-3);
        }
        else if constexpr(std::is_same<InOutDataType, ck::bhalf_t>::value)
        {
            single_result = ck::utils::check_err(
                out.mData, out_ref.mData, "Error: Incorrect results!", 1e-5, 3.2e-2);
        }
        else
            single_result = ck::utils::check_err(out.mData, out_ref.mData);

        if(!single_result)
        {
            std::cout << "Fail Info: " << bnorm_fwd_ptr->GetTypeString() << std::endl;
            result = false;
        };
    };

    return (result);
};

} // anonymous namespace

static struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
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
    std::vector<size_t> inOutLengths;

    int data_type;
    int init_method = 1;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inOutLengths or -D, comma separated list of input/output tensor dimension "
                     "lengths "
                     "(only 4-d tensor supported)"
                  << std::endl;
        std::cout << "Arg1 -- data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)" << std::endl;
        std::cout << "Arg2 -- init method used for bnScale and bnBias (0=no init, 1=single integer "
                     "value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        unsigned int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inOutLengths = getTypeValuesFromString<size_t>(optarg);
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

        data_type   = std::atoi(argv[optind++]);
        init_method = std::atoi(argv[optind]);

        if(inOutLengths.size() != 4)
            return (-1);

        if(data_type != 0 && data_type != 1 && data_type != 3 && data_type != 5 && data_type != 6)
            return (-1);

        return (0);
    };
};

bool test_bnorm_fwd(int data_type,
                    int init_method,
                    const std::vector<size_t> inOutLengths,
                    const std::vector<size_t> scaleBiasMeanVarLengths,
                    bool saveMeanAndInvVariance,
                    bool updateMovingAverage,
                    float alpha,
                    float beta)
{
    bool result                           = true;
    const double exponentialAverageFactor = 0.2;

    if(data_type == 0)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        result = test_bnorm_fwd_impl<ck::half_t, float>(init_method,
                                                        inOutLengths,
                                                        scaleBiasMeanVarLengths,
                                                        saveMeanAndInvVariance,
                                                        updateMovingAverage,
                                                        epsilon,
                                                        exponentialAverageFactor,
                                                        alpha,
                                                        beta);
    }
    else if(data_type == 1)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        result = test_bnorm_fwd_impl<float, float>(init_method,
                                                   inOutLengths,
                                                   scaleBiasMeanVarLengths,
                                                   saveMeanAndInvVariance,
                                                   updateMovingAverage,
                                                   epsilon,
                                                   exponentialAverageFactor,
                                                   alpha,
                                                   beta);
    }
    else if(data_type == 3)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        result = test_bnorm_fwd_impl<int8_t, float>(init_method,
                                                    inOutLengths,
                                                    scaleBiasMeanVarLengths,
                                                    saveMeanAndInvVariance,
                                                    updateMovingAverage,
                                                    epsilon,
                                                    exponentialAverageFactor,
                                                    alpha,
                                                    beta);
    }
    else if(data_type == 5)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        result = test_bnorm_fwd_impl<ck::bhalf_t, float>(init_method,
                                                         inOutLengths,
                                                         scaleBiasMeanVarLengths,
                                                         saveMeanAndInvVariance,
                                                         updateMovingAverage,
                                                         epsilon,
                                                         exponentialAverageFactor,
                                                         alpha,
                                                         beta);
    }
    else if(data_type == 6)
    {
        const double epsilon = std::numeric_limits<double>::epsilon();

        result = test_bnorm_fwd_impl<double, double>(init_method,
                                                     inOutLengths,
                                                     scaleBiasMeanVarLengths,
                                                     saveMeanAndInvVariance,
                                                     updateMovingAverage,
                                                     epsilon,
                                                     exponentialAverageFactor,
                                                     alpha,
                                                     beta);
    };

    return result;
};

int main(int argc, char* argv[])
{
    SimpleAppArgs args;

    bool result = true;

    if(argc == 1)
    {
        std::vector<int> data_types = {0, 1};
        int init_method             = 1;

        std::vector<std::vector<size_t>> v_inOutLengths{{256, 14, 14, 1024},
                                                        {256, 28, 28, 128},
                                                        {256, 58, 58, 128},
                                                        {256, 7, 7, 2048},
                                                        {256, 14, 14, 256},
                                                        {256, 30, 30, 256},
                                                        {256, 56, 56, 256},
                                                        {256, 16, 16, 512}};
        for(auto data_type : data_types)
        {
            for(auto inOutLengths : v_inOutLengths)
            {
                std::vector<size_t> scaleBiasMeanVarLengths = {inOutLengths[3]};
                result                                      = result && test_bnorm_fwd(data_type,
                                                  init_method,
                                                  inOutLengths,
                                                  scaleBiasMeanVarLengths,
                                                  true,
                                                  false,
                                                  1.0f,
                                                  0.0f);
            };
        };
    }
    else
    {
        if(args.processArgs(argc, argv) < 0)
        {
            throw std::runtime_error(
                "Invalid input arguments, test_bnorm_fwd could not be executed!");
        };

        std::vector<size_t> scaleBiasMeanVarLengths = {args.inOutLengths[3]};

        result = test_bnorm_fwd(args.data_type,
                                args.init_method,
                                args.inOutLengths,
                                scaleBiasMeanVarLengths,
                                true,
                                false,
                                1.0f,
                                0.0f);
    }

    std::cout << "test_bnorm_fwd ..... " << (result ? "SUCCESS" : "FAILURE") << std::endl;

    return (result ? 0 : -1);
}
