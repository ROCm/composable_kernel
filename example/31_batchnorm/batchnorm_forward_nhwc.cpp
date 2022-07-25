// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <limits>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/host_tensor/host_common_util.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_forward_nhwc_c.hpp"

#include "batchnorm_forward_impl.hpp"

template <typename InOutDataType, typename AccDataType>
using ReferenceBatchNormFwdInstance =
    ck::tensor_operation::host::ReferenceBatchNormFwd_Input_N_H_W_C_Output_C<InOutDataType,
                                                                             AccDataType>;

static struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class BatchNormFwdArg
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inOutLengths;

    bool do_verification = false;

    bool updateMovingAverage;
    bool saveMeanAndInvVariance;

    int data_type    = 0;
    int init_method  = 2;
    bool time_kernel = false;

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
        std::cout << "Arg1: data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)" << std::endl;
        std::cout << "Arg2: 1/0 to indicate whether to update the moving average and variance "
                     "(0=no, 1=yes)"
                  << std::endl;
        std::cout << "Arg3: 1/0 to indicate whether to save the calculated mean and invVariance "
                     "(0=no, 1=yes)"
                  << std::endl;
        std::cout << "Arg4: init method used for bnScale and bnBias (0=no init, 1=single integer "
                     "value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
        std::cout << "Arg5: time kernel (0=no, 1=yes)" << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

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

        if(optind + 5 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        data_type              = std::atoi(argv[optind++]);
        updateMovingAverage    = std::atoi(argv[optind++]);
        saveMeanAndInvVariance = std::atoi(argv[optind++]);
        init_method            = std::atoi(argv[optind++]);
        time_kernel            = static_cast<bool>(std::atoi(argv[optind]));

        if(data_type != 0 && data_type != 1 && data_type != 3 && data_type != 5 && data_type != 6)
            return (-1);

        return (0);
    };
};

using namespace ck;

template <typename InOutDataType, typename AccDataType>
bool bnorm_fwd_nhwc_test(bool do_verification,
                         int init_method,
                         bool time_kernel,
                         const std::vector<size_t> inOutLengths,
                         bool updateMovingAverage,
                         bool saveMeanAndInvVariance,
                         double averageFactor,
                         double epsilon)
{
    // for NHWC BatchNorm calculation of mean and meansquare
    constexpr int Rank         = 4;
    constexpr int NumReduceDim = 3;

    const std::vector<size_t> scaleBiasMeanVarLengths = {inOutLengths[3]};

    // input data of the batchnorm forward algorithm
    Tensor<InOutDataType> x(inOutLengths);
    Tensor<AccDataType> bnScale(scaleBiasMeanVarLengths);
    Tensor<AccDataType> bnBias(scaleBiasMeanVarLengths);

    // output data of the batchnorm forward algorithm
    Tensor<InOutDataType> y_ref(inOutLengths);
    Tensor<InOutDataType> y(inOutLengths);

    Tensor<AccDataType> resultSaveMean_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultSaveInvVariance_ref(scaleBiasMeanVarLengths);

    Tensor<AccDataType> resultRunningMean_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultRunningVariance_ref(scaleBiasMeanVarLengths);

    auto inOutStrides            = x.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = bnScale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(updateMovingAverage)
    {
        if constexpr(std::is_same<InOutDataType, int8_t>::value)
        {
            x.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);

            const float x_mean       = 0.0f;
            const float x_stddev     = 2.5f;
            const float noise_stddev = 0.04f;

            resultRunningMean_ref.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{x_mean, noise_stddev}, num_thread);

            resultRunningVariance_ref.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);
        }
        else
        {
            const float x_mean       = 0.0f;
            const float x_stddev     = 1.0f;
            const float noise_stddev = 0.04f;

            // input data in normal distribution
            x.GenerateTensorValue(GeneratorTensor_4<InOutDataType>{x_mean, x_stddev}, num_thread);

            // initialize the runningMean to be values with tiny variation to the mean of the x
            // values
            resultRunningMean_ref.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{x_mean, noise_stddev}, num_thread);

            // initialize the runningVariance to be values with tiny variation to the variance of
            // the x values
            resultRunningVariance_ref.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);
        };
    }
    else
    {
        if constexpr(std::is_same<InOutDataType, int8_t>::value)
            x.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
        else
            x.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0f, 5.0f}, num_thread);
    };

    if(do_verification)
    {
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
        }
    };

    // these buffers are usually provided by the user application
    DeviceMem x_dev(sizeof(InOutDataType) * x.mDesc.GetElementSpace());
    DeviceMem y_dev(sizeof(InOutDataType) * y.mDesc.GetElementSpace());
    DeviceMem bnScale_dev(sizeof(AccDataType) * bnScale.mDesc.GetElementSpace());
    DeviceMem bnBias_dev(sizeof(AccDataType) * bnBias.mDesc.GetElementSpace());

    // mean_dev or resultSaveMean_dev
    DeviceMem resultSaveMean_dev(sizeof(AccDataType) * resultSaveMean_ref.mDesc.GetElementSpace());
    // meansquare_dev or resultSaveInvVariance_dev
    DeviceMem resultSaveInvVariance_dev(sizeof(AccDataType) *
                                        resultSaveInvVariance_ref.mDesc.GetElementSpace());
    // resultRunningMean_dev
    DeviceMem resultRunningMean_dev(sizeof(AccDataType) *
                                    resultRunningMean_ref.mDesc.GetElementSpace());
    // resultRunningVariance_dev
    DeviceMem resultRunningVariance_dev(sizeof(AccDataType) *
                                        resultRunningVariance_ref.mDesc.GetElementSpace());

    x_dev.ToDevice(x.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());
    bnBias_dev.ToDevice(bnBias.mData.data());

    if(updateMovingAverage)
    {
        resultRunningMean_dev.ToDevice(resultRunningMean_ref.mData.data());
        resultRunningVariance_dev.ToDevice(resultRunningVariance_ref.mData.data());
    };

    std::array<index_t, Rank> i_inOutLengths;
    std::array<index_t, Rank> i_inOutStrides;
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarStrides;

    std::copy(inOutLengths.begin(), inOutLengths.end(), i_inOutLengths.begin());
    std::copy(inOutStrides.begin(), inOutStrides.end(), i_inOutStrides.begin());
    std::copy(scaleBiasMeanVarLengths.begin(),
              scaleBiasMeanVarLengths.end(),
              i_scaleBiasMeanVarLengths.begin());
    std::copy(scaleBiasMeanVarStrides.begin(),
              scaleBiasMeanVarStrides.end(),
              i_scaleBiasMeanVarStrides.begin());

    int result = 0;

    // used for saving meansquare
    DeviceMem workspace(sizeof(AccDataType) * 2 * resultSaveMean_ref.mDesc.GetElementSpace() + 128);

    void* p_tmp_mean = workspace.GetDeviceBuffer();
    void* p_tmp_meansquare =
        static_cast<char*>(p_tmp_mean) +
        (sizeof(AccDataType) * resultSaveMean_ref.mDesc.GetElementSpace() + 63) / 64 * 64;

    result = bnorm_fwd<InOutDataType, AccDataType, Rank, NumReduceDim, false>(
        time_kernel,
        updateMovingAverage,
        saveMeanAndInvVariance,
        {0, 1, 2},
        i_inOutLengths,
        i_inOutStrides,
        i_inOutStrides,
        i_scaleBiasMeanVarLengths,
        i_scaleBiasMeanVarStrides,
        x_dev.GetDeviceBuffer(),
        bnScale_dev.GetDeviceBuffer(),
        bnBias_dev.GetDeviceBuffer(),
        y_dev.GetDeviceBuffer(),
        averageFactor,
        updateMovingAverage ? resultRunningMean_dev.GetDeviceBuffer() : nullptr,
        updateMovingAverage ? resultRunningVariance_dev.GetDeviceBuffer() : nullptr,
        epsilon,
        saveMeanAndInvVariance ? resultSaveMean_dev.GetDeviceBuffer() : nullptr,
        saveMeanAndInvVariance ? resultSaveInvVariance_dev.GetDeviceBuffer() : nullptr,
        p_tmp_mean,
        p_tmp_meansquare);

    if(result < 0)
        return (false);

    bool pass = true;

    if(do_verification)
    {
        auto batchNormFwd_ref = ReferenceBatchNormFwdInstance<InOutDataType, AccDataType>{};

        auto argument_ptr_ref = batchNormFwd_ref.MakeArgumentPointer(
            i_inOutLengths,
            i_inOutStrides,
            i_inOutStrides,
            i_scaleBiasMeanVarLengths,
            i_scaleBiasMeanVarStrides,
            x.mData.data(),
            bnScale.mData.data(),
            bnBias.mData.data(),
            y_ref.mData.data(),
            0.1, // exponentialAverageFactor
            updateMovingAverage ? resultRunningMean_ref.mData.data() : nullptr, // resultRunningMean
            updateMovingAverage ? resultRunningVariance_ref.mData.data()
                                : nullptr, // resultRunningVariance
            epsilon,
            saveMeanAndInvVariance ? resultSaveMean_ref.mData.data() : nullptr,
            saveMeanAndInvVariance ? resultSaveInvVariance_ref.mData.data() : nullptr);

        if(!batchNormFwd_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout
                << "The runtime parameters seems not supported by the BatchNorm instance, exiting!"
                << std::endl;
            return (-2);
        };

        auto invoker_ptr_ref = batchNormFwd_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get());

        y_dev.FromDevice(y.mData.data());
        pass = pass && ck::utils::check_err(y.mData, y_ref.mData);

        if(updateMovingAverage)
        {
            Tensor<AccDataType> resultRunningMean(scaleBiasMeanVarLengths);
            Tensor<AccDataType> resultRunningVariance(scaleBiasMeanVarLengths);

            resultRunningMean_dev.FromDevice(resultRunningMean.mData.data());
            resultRunningVariance_dev.FromDevice(resultRunningVariance.mData.data());

            pass =
                pass && ck::utils::check_err(resultRunningMean.mData, resultRunningMean_ref.mData);
            pass = pass && ck::utils::check_err(resultRunningVariance.mData,
                                                resultRunningVariance_ref.mData);
        };

        if(saveMeanAndInvVariance)
        {
            Tensor<AccDataType> resultSaveMean(scaleBiasMeanVarLengths);
            Tensor<AccDataType> resultSaveInvVariance(scaleBiasMeanVarLengths);

            resultSaveMean_dev.FromDevice(resultSaveMean.mData.data());
            resultSaveInvVariance_dev.FromDevice(resultSaveInvVariance.mData.data());

            pass = pass && ck::utils::check_err(resultSaveMean.mData, resultSaveMean_ref.mData);
            pass = pass && ck::utils::check_err(resultSaveInvVariance.mData,
                                                resultSaveInvVariance_ref.mData);
        };
    };

    return (pass);
};

const double epsilon              = std::numeric_limits<float>::epsilon();
static const double averageFactor = 0.1;

int main(int argc, char* argv[])
{
    bool pass = true;

    if(argc > 1)
    {
        BatchNormFwdArg arg;

        if(arg.processArgs(argc, argv) < 0)
            return (-1);

        if(arg.data_type == 0)
        {
            pass = bnorm_fwd_nhwc_test<ck::half_t, float>(arg.do_verification,
                                                          arg.init_method,
                                                          arg.time_kernel,
                                                          arg.inOutLengths,
                                                          arg.updateMovingAverage,
                                                          arg.saveMeanAndInvVariance,
                                                          averageFactor,
                                                          epsilon);
        }
        else if(arg.data_type == 1)
        {
            pass = bnorm_fwd_nhwc_test<float, float>(arg.do_verification,
                                                     arg.init_method,
                                                     arg.time_kernel,
                                                     arg.inOutLengths,
                                                     arg.updateMovingAverage,
                                                     arg.saveMeanAndInvVariance,
                                                     averageFactor,
                                                     epsilon);
        }
        else if(arg.data_type == 3)
        {
            pass = bnorm_fwd_nhwc_test<int8_t, float>(arg.do_verification,
                                                      arg.init_method,
                                                      arg.time_kernel,
                                                      arg.inOutLengths,
                                                      arg.updateMovingAverage,
                                                      arg.saveMeanAndInvVariance,
                                                      averageFactor,
                                                      epsilon);
        }
        else if(arg.data_type == 5)
        {
            pass = bnorm_fwd_nhwc_test<ck::bhalf_t, float>(arg.do_verification,
                                                           arg.init_method,
                                                           arg.time_kernel,
                                                           arg.inOutLengths,
                                                           arg.updateMovingAverage,
                                                           arg.saveMeanAndInvVariance,
                                                           averageFactor,
                                                           epsilon);
        }
        else if(arg.data_type == 6)
        {
            pass = bnorm_fwd_nhwc_test<double, double>(arg.do_verification,
                                                       arg.init_method,
                                                       arg.time_kernel,
                                                       arg.inOutLengths,
                                                       arg.updateMovingAverage,
                                                       arg.saveMeanAndInvVariance,
                                                       averageFactor,
                                                       epsilon);
        }
    }
    else
    {
        pass = bnorm_fwd_nhwc_test<ck::half_t, float>(true,
                                                      2,
                                                      false, // don't time kernel
                                                      {128, 16, 16, 1024},
                                                      true,
                                                      false,
                                                      averageFactor,
                                                      epsilon);
    };

    return (pass ? 0 : 1);
}
