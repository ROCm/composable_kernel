// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <thread>

#include "ck/utility/math_v2.hpp"
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
                 const std::array<int, 3> reduceDims,
                 const std::array<index_t, 1> bnScaleBiasMeanVarLengths,
                 const std::array<index_t, 1> bnScaleBiasStrides,
                 const std::array<index_t, 1> bnMeanVarStrides,
                 const InOutDataType* p_x,
                 const AccDataType* bnScale,
                 const AccDataType* bnBias,
                 double epsilon,
                 InOutDataType* p_y,
                 AccDataType* resultSaveMean,
                 AccDataType* resultSaveInvVariance,
                 double averageFactor,
                 AccDataType* resultRunningMean,
                 AccDataType* resultRunningVariance)
            : p_x_(p_x),
              bnScale_(bnScale),
              bnBias_(bnBias),
              p_y_(p_y),
              resultSaveMean_(resultSaveMean),
              resultSaveInvVariance_(resultSaveInvVariance),
              resultRunningMean_(resultRunningMean),
              resultRunningVariance_(resultRunningVariance)
        {
            (void)xStrides;
            (void)yStrides;
            (void)bnScaleBiasStrides;
            (void)bnMeanVarStrides;
            (void)reduceDims;

            if(xyLengths.size() != 4 || bnScaleBiasMeanVarLengths.size() != 1 ||
               bnScaleBiasMeanVarLengths[0] != xyLengths[3])
                throw std::runtime_error("Invalid tensor dimensions!");

            n = xyLengths[0];
            h = xyLengths[1];
            w = xyLengths[2];
            c = xyLengths[3];

            epsilon_       = type_convert<AccDataType>(epsilon);
            averageFactor_ = type_convert<AccDataType>(averageFactor);

            resultSave    = (resultSaveMean != nullptr && resultSaveInvVariance != nullptr);
            resultRunning = (resultRunningMean != nullptr && resultRunningVariance != nullptr);
        }

        const InOutDataType* p_x_;
        const AccDataType* bnScale_;
        const AccDataType* bnBias_;
        InOutDataType* p_y_;

        AccDataType* resultSaveMean_;
        AccDataType* resultSaveInvVariance_;
        AccDataType* resultRunningMean_;
        AccDataType* resultRunningVariance_;

        bool resultSave, resultRunning;

        index_t n, h, w, c;

        AccDataType averageFactor_;
        AccDataType epsilon_;
    };

    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            auto thread_reduce_func = [&](auto iC) {
                index_t offset_C     = iC;
                AccDataType mean     = type_convert<AccDataType>(0.0f);
                AccDataType variance = type_convert<AccDataType>(0.0f);
                int32_t curr_count   = 0;

                // compute mean, variance using welford method
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

                            curr_count++;

                            AccDataType x = type_convert<AccDataType>(arg.p_x_[offset]);

                            AccDataType delta = x - mean;

                            mean += delta / curr_count;

                            AccDataType delta2 = x - mean;

                            variance += delta * delta2;
                        };
                    }
                };

                // actual variance
                variance = variance / curr_count;

                AccDataType invVariance =
                    type_convert<AccDataType>(1.0f) / ck::math::sqrt(arg.epsilon_ + variance);

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
                            (type_convert<AccDataType>(1.0) - arg.averageFactor_) +
                        mean * arg.averageFactor_;
                    arg.resultRunningVariance_[iC] =
                        arg.resultRunningVariance_[iC] *
                            (type_convert<AccDataType>(1.0) - arg.averageFactor_) +
                        variance * arg.averageFactor_;
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
                        const std::array<int, 3> reduceDims,
                        const std::array<index_t, 1> bnScaleBiasMeanVarLengths,
                        const std::array<index_t, 1> bnScaleBiasStrides,
                        const std::array<index_t, 1> bnMeanVarStrides,
                        const void* p_x,
                        const void* bnScale,
                        const void* bnBias,
                        double epsilon,
                        void* p_y,
                        void* resultSaveMean,
                        void* resultSaveInvVariance,
                        double averageFactor,
                        void* resultRunningMean,
                        void* resultRunningVariance) override
    {
        return std::make_unique<Argument>(xyLengths,
                                          xStrides,
                                          yStrides,
                                          reduceDims,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleBiasStrides,
                                          bnMeanVarStrides,
                                          static_cast<const InOutDataType*>(p_x),
                                          static_cast<const AccDataType*>(bnScale),
                                          static_cast<const AccDataType*>(bnBias),
                                          epsilon,
                                          static_cast<InOutDataType*>(p_y),
                                          static_cast<AccDataType*>(resultSaveMean),
                                          static_cast<AccDataType*>(resultSaveInvVariance),
                                          averageFactor,
                                          static_cast<AccDataType*>(resultRunningMean),
                                          static_cast<AccDataType*>(resultRunningVariance));
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
