#pragma once

#include <iostream>
#include <fstream>
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "check_err.hpp"
#include "device_bnorm_fwd.hpp"
#include "device_bnorm_fwd_instance.hpp"
#include "reference_bnorm_fwd_nhwc_c.hpp"
#include "host_common_util.hpp"

namespace ck {
namespace profiler {

template <typename InOutDataType, typename AccDataType>
bool profile_bnorm_fwd_impl(bool do_verification,
                            int init_method,
                            bool do_dumpout,
                            int nrepeat,
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
    using ck::host_common::dumpBufferToFile;
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

    bool pass = true;

    std::size_t num_thread = std::thread::hardware_concurrency();

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
        };

        const float data_mean    = 0.0f;
        const float data_stddev  = 1.0f;
        const float noise_stddev = 0.0001f;

        // input data in normal distribution
        in.GenerateTensorValue(GeneratorTensor_4<InOutDataType>{data_mean, data_stddev},
                               num_thread);

        if(updateMovingAverage)
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

    if(beta != 0.0f)
        out_dev.ToDevice(out.mData.data());

    if(updateMovingAverage)
    {
        resultRunningMean_dev.ToDevice(resultRunningMean.mData.data());
        resultRunningVariance_dev.ToDevice(resultRunningMean.mData.data());
    };

    float best_avg_time   = 0;
    float best_gb_per_sec = 0;

    std::vector<DeviceBatchNormFwdPtr> bnorm_fwd_ptrs;

    add_device_bnorm_fwd_with_reduce_blockwise_instance<InOutDataType, AccDataType>(bnorm_fwd_ptrs);
    add_device_bnorm_fwd_with_reduce_multiblock_instance<InOutDataType, AccDataType>(
        bnorm_fwd_ptrs);

    const auto i_inOutLengths            = to_int_vector(inOutLengths);
    const auto i_inOutStrides            = to_int_vector(inOutStrides);
    const auto i_scaleBiasMeanVarLengths = to_int_vector(scaleBiasMeanVarLengths);
    const auto i_scaleBiasMeanVarStrides = to_int_vector(scaleBiasMeanVarStrides);

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

        float avg_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

        // currently, input data is accessed from device memory separately for computing the mean
        // and meansquare, later the implementation will be fused as well as accessing the input
        // data from device memory
        std::size_t num_bytes = 2 * inOutLengths[0] * inOutLengths[1] * inOutLengths[2] *
                                inOutLengths[3] * sizeof(InOutDataType);

        if(saveMeanAndInvVariance)
            num_bytes += 2 * inOutLengths[3] * sizeof(AccDataType);

        if(updateMovingAverage)
            num_bytes += 2 * inOutLengths[3] * 2 * sizeof(AccDataType);

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        if(nrepeat > 0)
            std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << bnorm_fwd_name << std::endl;

        if(gb_per_sec > best_gb_per_sec)
        {
            best_avg_time   = avg_time;
            best_gb_per_sec = gb_per_sec;
        }

        if(do_verification)
        {
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
            {
                std::cout << "The runtime parameters seems not supported by the BatchNorm "
                             "instance, exiting!"
                          << std::endl;
            };

            auto invoker_ptr_ref = bnorm_fwd_ref.MakeInvokerPointer();

            (void)invoker_ptr_ref->Run(argument_ptr_ref.get(), 1);

            out_dev.FromDevice(out.mData.data());

            bool single_pass;

            if constexpr(std::is_same<InOutDataType, ck::half_t>::value)
            {
                single_pass = ck::utils::check_err(
                    out.mData, out_ref.mData, "Error: Incorrect results!", 1e-5, 4.0e-3);
            }
            else if constexpr(std::is_same<InOutDataType, float>::value)
            {
                single_pass = ck::utils::check_err(
                    out.mData, out_ref.mData, "Error: Incorrect results!", 1e-5, 1.4e-3);
            }
            else if constexpr(std::is_same<InOutDataType, ck::bhalf_t>::value)
            {
                single_pass = ck::utils::check_err(
                    out.mData, out_ref.mData, "Error: Incorrect results!", 1e-5, 3.2e-2);
            }
            else
                single_pass = ck::utils::check_err(out.mData, out_ref.mData);

            pass = pass && single_pass;
        };

        if(do_dumpout)
        {
            dumpBufferToFile("dump_in.bin", in.mData.data(), in.mDesc.GetElementSize());
            dumpBufferToFile("dump_out.bin", out.mData.data(), out.mDesc.GetElementSize());
            dumpBufferToFile(
                "dump_out_host.bin", out_ref.mData.data(), out_ref.mDesc.GetElementSize());

            if(saveMeanAndInvVariance)
            {
                Tensor<AccDataType> resultSaveMean(scaleBiasMeanVarLengths);
                Tensor<AccDataType> resultSaveInvVariance(scaleBiasMeanVarLengths);

                resultSaveMean_dev.FromDevice(resultSaveMean.mData.data());
                resultSaveInvVariance_dev.FromDevice(resultSaveInvVariance.mData.data());

                dumpBufferToFile("dump_mean.bin",
                                 resultSaveMean.mData.data(),
                                 resultSaveMean.mDesc.GetElementSize());
                dumpBufferToFile("dump_mean_host.bin",
                                 resultSaveMean_ref.mData.data(),
                                 resultSaveMean_ref.mDesc.GetElementSize());

                dumpBufferToFile("dump_inv_variance.bin",
                                 resultSaveInvVariance.mData.data(),
                                 resultSaveInvVariance.mDesc.GetElementSize());
                dumpBufferToFile("dump_inv_variance_host.bin",
                                 resultSaveInvVariance_ref.mData.data(),
                                 resultSaveInvVariance_ref.mDesc.GetElementSize());
            };
        };
    };

    if(nrepeat > 0)
        std::cout << "Best Perf: " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s"
                  << std::endl;

    return pass;
};

} // namespace profiler
} // namespace ck
