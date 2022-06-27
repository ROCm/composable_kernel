// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <limits>
#include <iostream>
#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/host_tensor/host_common_util.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_backward_nhwc_c.hpp"

#include "batchnorm_backward_impl.hpp"

template <typename InOutDataType, typename AccDataType>
using ReferenceBatchNormBwdInstance =
    ck::tensor_operation::host::ReferenceBatchNormBwd_Input_N_H_W_C_Output_C<InOutDataType,
                                                                             AccDataType>;

static struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class BatchNormBwdArg
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inOutLengths;

    bool do_verification = false;

    bool use_savedMeanAndInvVariance;

    int init_method  = 3;
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
        std::cout << "Arg1 -- 1/0 to indicate whether to use saved mean and invVariance"
                  << std::endl;
        std::cout << "Arg2 -- init method used for dy and bnScale (0=no init, 1=single integer "
                     "value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
        std::cout << "Arg3 -- time kernel (0=no, 1=yes)" << std::endl;
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

        if(optind + 3 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        use_savedMeanAndInvVariance = std::atoi(argv[optind++]);
        init_method                 = std::atoi(argv[optind++]);
        time_kernel                 = static_cast<bool>(std::atoi(argv[optind]));

        return (0);
    };
};

using namespace ck;

template <typename InOutDataType, typename AccDataType>
bool bnorm_bwd_nhwc_test(bool do_verification,
                         int init_method,
                         bool time_kernel,
                         const std::vector<size_t> inOutLengths,
                         bool use_savedMeanAndInvVariance,
                         double epsilon)
{
    // for NHWC BatchNorm calculation of mean and meansquare
    constexpr index_t Rank         = 4;
    constexpr index_t NumReduceDim = 3;

    const std::vector<size_t> scaleBiasDiffLengths = {inOutLengths[3]};

    // input data of the batchnorm backward algorithm
    Tensor<InOutDataType> x(inOutLengths);
    Tensor<InOutDataType> dy(inOutLengths);

    Tensor<AccDataType> bnScale(scaleBiasDiffLengths);

    Tensor<AccDataType> savedMean(scaleBiasDiffLengths);
    Tensor<AccDataType> savedInvVariance(scaleBiasDiffLengths);
    // savedVariance is only used for initializing savedInvVariance
    Tensor<AccDataType> savedVariance(scaleBiasDiffLengths);

    // output data of the batchnorm backward algorithm
    Tensor<InOutDataType> dx_ref(inOutLengths);
    Tensor<InOutDataType> dx(inOutLengths);

    Tensor<AccDataType> bnScaleDiff(scaleBiasDiffLengths);
    Tensor<AccDataType> bnBiasDiff(scaleBiasDiffLengths);

    Tensor<AccDataType> bnScaleDiff_ref(scaleBiasDiffLengths);
    Tensor<AccDataType> bnBiasDiff_ref(scaleBiasDiffLengths);

    auto inOutStrides         = dy.mDesc.GetStrides();
    auto scaleBiasDiffStrides = bnScaleDiff.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(use_savedMeanAndInvVariance)
    {
        const float x_mean       = 0.0f;
        const float x_stddev     = 1.0f;
        const float noise_stddev = 0.0001f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<InOutDataType>{x_mean, x_stddev}, num_thread);

        // initialize the savedMean to be values with tiny variation to the mean of the x values
        savedMean.GenerateTensorValue(GeneratorTensor_4<AccDataType>{x_mean, noise_stddev},
                                      num_thread);

        // initialize the variance to be values with tiny variation to the variance of the x values
        savedVariance.GenerateTensorValue(
            GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);

        auto it_src       = savedVariance.mData.begin();
        auto it_dst       = savedInvVariance.mData.begin();
        float tmp_epsilon = std::numeric_limits<float>::epsilon();

        while(it_src != savedVariance.mData.end())
        {
            *it_dst = type_convert<AccDataType>(
                1.0f / std::sqrtf(type_convert<float>(*it_src) + tmp_epsilon));

            it_src++;
            it_dst++;
        };
    }
    else
    {
        const float x_mean   = 0.0f;
        const float x_stddev = 1.0f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<InOutDataType>{x_mean, x_stddev}, num_thread);
    };

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            dy.GenerateTensorValue(GeneratorTensor_0<InOutDataType>{}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_0<InOutDataType>{}, num_thread);
            break;
        case 1:
            dy.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            break;
        case 2:
            dy.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            break;
        default:
            dy.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-1.0f, 1.0f}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-1.0f, 1.0f}, num_thread);
        }
    };

    // input data of the batchnorm backward algorithm
    DeviceMem x_dev(sizeof(InOutDataType) * x.mDesc.GetElementSpace());
    DeviceMem dy_dev(sizeof(InOutDataType) * dy.mDesc.GetElementSpace());

    DeviceMem bnScale_dev(sizeof(AccDataType) * bnScale.mDesc.GetElementSpace());

    DeviceMem savedMean_dev(sizeof(AccDataType) * savedMean.mDesc.GetElementSpace());
    DeviceMem savedInvVariance_dev(sizeof(AccDataType) * savedInvVariance.mDesc.GetElementSpace());

    // output data of the batchnorm backward algorithm
    DeviceMem dx_dev(sizeof(InOutDataType) * dx.mDesc.GetElementSpace());

    DeviceMem bnScaleDiff_dev(sizeof(AccDataType) * bnScaleDiff.mDesc.GetElementSpace());
    DeviceMem bnBiasDiff_dev(sizeof(AccDataType) * bnBiasDiff.mDesc.GetElementSpace());

    x_dev.ToDevice(x.mData.data());
    dy_dev.ToDevice(dy.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());

    if(use_savedMeanAndInvVariance)
    {
        savedMean_dev.ToDevice(savedMean.mData.data());
        savedInvVariance_dev.ToDevice(savedInvVariance.mData.data());
    };

    std::vector<ck::index_t> i_inOutLengths;
    std::vector<ck::index_t> i_inOutStrides;
    std::vector<ck::index_t> i_scaleBiasDiffLengths;
    std::vector<ck::index_t> i_scaleBiasDiffStrides;

    i_inOutLengths.assign(inOutLengths.begin(), inOutLengths.end());
    i_inOutStrides.assign(inOutStrides.begin(), inOutStrides.end());
    i_scaleBiasDiffLengths.assign(scaleBiasDiffLengths.begin(), scaleBiasDiffLengths.end());
    i_scaleBiasDiffStrides.assign(scaleBiasDiffStrides.begin(), scaleBiasDiffStrides.end());
    int result = 0;

    if(use_savedMeanAndInvVariance)
    {
        result = batchnorm::bnorm_bwd_use_saved_mean_inv_variance<InOutDataType,
                                                                  AccDataType,
                                                                  Rank,
                                                                  NumReduceDim,
                                                                  false>(
            time_kernel,
            {0, 1, 2},
            i_inOutLengths,
            i_inOutStrides, // xStrides
            i_inOutStrides, // dyStrides
            i_inOutStrides, // dxStrides
            i_scaleBiasDiffLengths,
            i_scaleBiasDiffStrides,
            x_dev.GetDeviceBuffer(),
            dy_dev.GetDeviceBuffer(),
            bnScale_dev.GetDeviceBuffer(),
            savedMean_dev.GetDeviceBuffer(),
            savedInvVariance_dev.GetDeviceBuffer(),
            dx_dev.GetDeviceBuffer(),
            bnScaleDiff_dev.GetDeviceBuffer(),
            bnBiasDiff_dev.GetDeviceBuffer());
    }
    else
    {
        DeviceMem workspace(sizeof(AccDataType) * bnScale.mDesc.GetElementSpace());

        result = batchnorm::bnorm_bwd_without_saved_mean_inv_variance<InOutDataType,
                                                                      AccDataType,
                                                                      Rank,
                                                                      NumReduceDim,
                                                                      false>(
            time_kernel,
            {0, 1, 2},
            i_inOutLengths,
            i_inOutStrides, // xStrides
            i_inOutStrides, // dyStrides
            i_inOutStrides, // dxStrides
            i_scaleBiasDiffLengths,
            i_scaleBiasDiffStrides,
            x_dev.GetDeviceBuffer(),
            dy_dev.GetDeviceBuffer(),
            bnScale_dev.GetDeviceBuffer(),
            epsilon,
            dx_dev.GetDeviceBuffer(),
            bnScaleDiff_dev.GetDeviceBuffer(),
            bnBiasDiff_dev.GetDeviceBuffer(),
            workspace.GetDeviceBuffer());
    };

    if(result < 0)
        return (false);

    bool pass = true;

    if(do_verification)
    {
        auto batchNormBwd_ref = ReferenceBatchNormBwdInstance<InOutDataType, AccDataType>{};

        auto argument_ptr_ref = batchNormBwd_ref.MakeArgumentPointer(
            i_inOutLengths,
            i_inOutStrides,
            i_inOutStrides,
            i_inOutStrides,
            i_scaleBiasDiffLengths,
            i_scaleBiasDiffStrides,
            x.mData.data(),
            dy.mData.data(),
            bnScale.mData.data(),
            use_savedMeanAndInvVariance ? savedMean.mData.data() : nullptr,
            use_savedMeanAndInvVariance ? savedInvVariance.mData.data() : nullptr,
            epsilon,
            dx_ref.mData.data(),
            bnScaleDiff_ref.mData.data(),
            bnBiasDiff_ref.mData.data());

        if(!batchNormBwd_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout
                << "The runtime parameters seems not supported by the device instance, exiting!"
                << std::endl;
            return (-2);
        };

        auto invoker_ptr_ref = batchNormBwd_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get());

        dx_dev.FromDevice(dx.mData.data());
        pass = pass && ck::utils::check_err(dx.mData, dx_ref.mData);
    };

    return (pass);
};

using InOutDataType = ck::half_t;
using AccDataType   = float;

static const double epsilon = std::numeric_limits<AccDataType>::epsilon();

int main(int argc, char* argv[])
{
    bool pass = true;

    if(argc > 1)
    {
        BatchNormBwdArg arg;

        if(arg.processArgs(argc, argv) < 0)
            return (-1);

        pass = bnorm_bwd_nhwc_test<InOutDataType, AccDataType>(arg.do_verification,
                                                               arg.init_method,
                                                               arg.time_kernel,
                                                               arg.inOutLengths,
                                                               arg.use_savedMeanAndInvVariance,
                                                               epsilon);
    }
    else
    {
        pass = bnorm_bwd_nhwc_test<InOutDataType, AccDataType>(true,
                                                               3,
                                                               false, // don't time kernel
                                                               {128, 16, 16, 1024},
                                                               true,
                                                               epsilon);
    };

    return (pass ? 0 : 1);
}
