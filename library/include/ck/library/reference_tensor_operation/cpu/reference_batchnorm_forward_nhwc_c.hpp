// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <thread>

#include "ck/tensor_operation/gpu/device/device_batchnorm_forward.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename InOutDataType, typename AccDataType>
struct ReferenceBatchNormFwd_Input_N_H_W_C_Output_C : public device::DeviceBatchNormFwd<4, 3>
{
    struct Argument : public device::BaseArgument
    {
        Argument(const std::array<index_t, 4> xyLengths,
                 const std::array<index_t, 4> xStrides,
                 const std::array<index_t, 4> yStrides,
                 const std::array<index_t, 1> bnScaleBiasMeanVarLengths,
                 const std::array<index_t, 1> bnScaleBiasMeanVarStrides,
                 const InOutDataType* p_x,
                 const AccDataType* bnScale,
                 const AccDataType* bnBias,
                 InOutDataType* p_y,
                 double exponentialAverageFactor,
                 AccDataType* resultRunningMean,
                 AccDataType* resultRunningVariance,
                 double epsilon,
                 AccDataType* resultSaveMean,
                 AccDataType* resultSaveInvVariance)
            : p_x_(p_x),
              bnScale_(bnScale),
              bnBias_(bnBias),
              p_y_(p_y),
              resultRunningMean_(resultRunningMean),
              resultRunningVariance_(resultRunningVariance),
              resultSaveMean_(resultSaveMean),
              resultSaveInvVariance_(resultSaveInvVariance),
              exponentialAverageFactor_(exponentialAverageFactor),
              epsilon_(epsilon)
        {
            (void)xStrides;
            (void)yStrides;
            (void)bnScaleBiasMeanVarStrides;

            if(xyLengths.size() != 4 || bnScaleBiasMeanVarLengths.size() != 1 ||
               bnScaleBiasMeanVarLengths[0] != xyLengths[3])
                throw std::runtime_error("Invalid tensor dimensions!");

            n = xyLengths[0];
            h = xyLengths[1];
            w = xyLengths[2];
            c = xyLengths[3];

            resultSave    = (resultSaveMean != nullptr && resultSaveInvVariance != nullptr);
            resultRunning = (resultRunningMean != nullptr && resultRunningVariance != nullptr);
        }

        const InOutDataType* p_x_;
        const AccDataType* bnScale_;
        const AccDataType* bnBias_;
        InOutDataType* p_y_;

        AccDataType* resultRunningMean_;
        AccDataType* resultRunningVariance_;
        AccDataType* resultSaveMean_;
        AccDataType* resultSaveInvVariance_;

        bool resultSave, resultRunning;

        index_t n, h, w, c;

        double exponentialAverageFactor_;
        double epsilon_;
    };

    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            auto thread_reduce_func = [&](auto iC) {
                AccDataType reduceSize = type_convert<AccDataType>(arg.n) *
                                         type_convert<AccDataType>(arg.h) *
                                         type_convert<AccDataType>(arg.w);
                index_t offset_C       = iC;
                AccDataType mean       = type_convert<AccDataType>(0.0f);
                AccDataType meansquare = type_convert<AccDataType>(0.0f);

                // compute mean, meanquare, variance, invVariance
                for(index_t iN = 0; iN < arg.n; iN++)
                {
                    index_t offset_N = iN * arg.h * arg.w * arg.c;
                    for(index_t iH = 0; iH < arg.h; iH++)
                    {
                        index_t offset_H = iH * arg.w * arg.c;
                        for(index_t iW = 0; iW < arg.w; iW++)
                        {
                            index_t offset_W = iW * arg.c;

                            auto offset = offset_N + offset_H + offset_W + offset_C;

                            AccDataType x = type_convert<AccDataType>(arg.p_x_[offset]);

                            mean += x;
                            meansquare += x * x;
                        };
                    }
                };

                mean       = mean / reduceSize;
                meansquare = meansquare / reduceSize;

                AccDataType variance = meansquare - mean * mean;
                AccDataType invVariance =
                    type_convert<AccDataType>(1.0f) /
                    std::sqrt(type_convert<AccDataType>(arg.epsilon_) + variance);

                // save the mean/invVariance if required
                if(arg.resultSave)
                {
                    arg.resultSaveMean_[iC]        = mean;
                    arg.resultSaveInvVariance_[iC] = invVariance;
                };

                // update the moving average if required
                if(arg.resultRunning)
                {
                    arg.resultRunningMean_[iC] =
                        arg.resultRunningMean_[iC] *
                            type_convert<AccDataType>(1.0 - arg.exponentialAverageFactor_) +
                        mean * arg.exponentialAverageFactor_;
                    arg.resultRunningVariance_[iC] =
                        arg.resultRunningVariance_[iC] *
                            type_convert<AccDataType>(1.0 - arg.exponentialAverageFactor_) +
                        variance * arg.exponentialAverageFactor_;
                };

                // Normalization
                for(index_t iN = 0; iN < arg.n; iN++)
                {
                    index_t offset_N = iN * arg.h * arg.w * arg.c;
                    for(index_t iH = 0; iH < arg.h; iH++)
                    {
                        index_t offset_H = iH * arg.w * arg.c;
                        for(index_t iW = 0; iW < arg.w; iW++)
                        {
                            index_t offset_W = iW * arg.c;

                            auto offset = offset_N + offset_H + offset_W + offset_C;

                            AccDataType x = type_convert<AccDataType>(arg.p_x_[offset]);

                            AccDataType norm_x =
                                arg.bnScale_[iC] * (x - mean) * invVariance + arg.bnBias_[iC];

                            arg.p_y_[offset] = type_convert<InOutDataType>(norm_x);
                        };
                    }
                };
            };

            std::size_t num_thread      = std::thread::hardware_concurrency();
            std::size_t work_per_thread = (arg.c + num_thread - 1) / num_thread;

            std::vector<joinable_thread> threads(num_thread);

            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t ic_begin = it * work_per_thread;
                std::size_t ic_end = std::min(static_cast<int>((it + 1) * work_per_thread), arg.c);

                auto f = [=] {
                    for(std::size_t ic = ic_begin; ic < ic_end; ++ic)
                    {
                        thread_reduce_func(ic);
                    }
                };

                threads[it] = joinable_thread(f);
            }

            return (0.0f);
        };

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /*stream_config*/ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        };
    };

    bool IsSupportedArgument(const device::BaseArgument* p_arg) override
    {
        (void)p_arg;

        return (true);
    };

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const std::array<index_t, 4> xyLengths,
                        const std::array<index_t, 4> xStrides,
                        const std::array<index_t, 4> yStrides,
                        const std::array<index_t, 1> bnScaleBiasMeanVarLengths,
                        const std::array<index_t, 1> bnScaleBiasMeanVarStrides,
                        const void* p_x,
                        const void* bnScale,
                        const void* bnBias,
                        void* p_y,
                        double exponentialAverageFactor,
                        void* resultRunningMean,
                        void* resultRunningVariance,
                        double epsilon,
                        void* resultSaveMean,
                        void* resultSaveInvVariance) override
    {
        return std::make_unique<Argument>(xyLengths,
                                          xStrides,
                                          yStrides,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleBiasMeanVarStrides,
                                          static_cast<const InOutDataType*>(p_x),
                                          static_cast<const AccDataType*>(bnScale),
                                          static_cast<const AccDataType*>(bnBias),
                                          static_cast<InOutDataType*>(p_y),
                                          exponentialAverageFactor,
                                          static_cast<AccDataType*>(resultRunningMean),
                                          static_cast<AccDataType*>(resultRunningVariance),
                                          epsilon,
                                          static_cast<AccDataType*>(resultSaveMean),
                                          static_cast<AccDataType*>(resultSaveInvVariance));
    };

    std::unique_ptr<device::BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "Reference_BatchNorm_Forward_NHWC_C<" << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
