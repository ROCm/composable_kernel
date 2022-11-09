// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <thread>

#include "ck/utility/math_v2.hpp"
#include "ck/utility/ignore.hpp"
#include "ck/tensor_operation/gpu/device/device_batchnorm_forward.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename YElementwiseOp>
struct ReferenceBatchNormFwd_Input_N_H_W_C_Output_C
    : public device::DeviceBatchNormFwd<XDataType,
                                        YDataType,
                                        AccDataType,
                                        ScaleDataType,
                                        BiasDataType,
                                        MeanVarDataType,
                                        YElementwiseOp,
                                        4,
                                        3>
{
    struct Argument : public device::BaseArgument
    {
        Argument(const std::array<index_t, 4> xyLengths,
                 const std::array<index_t, 4> xStrides,
                 const std::array<index_t, 4> yStrides,
                 const std::array<int, 3> reduceDims,
                 const std::array<index_t, 1> bnScaleBiasMeanVarLengths,
                 const std::array<index_t, 1> bnScaleStrides,
                 const std::array<index_t, 1> bnBiasStrides,
                 const std::array<index_t, 1> bnMeanVarStrides,
                 const XDataType* p_x,
                 const ScaleDataType* bnScale,
                 const BiasDataType* bnBias,
                 double epsilon,
                 const YElementwiseOp y_elementwise_op,
                 YDataType* p_y,
                 MeanVarDataType* resultSaveMean,
                 MeanVarDataType* resultSaveInvVariance,
                 double averageFactor,
                 MeanVarDataType* resultRunningMean,
                 MeanVarDataType* resultRunningVariance)
            : p_x_(p_x),
              bnScale_(bnScale),
              bnBias_(bnBias),
              y_elementwise_op_(y_elementwise_op),
              p_y_(p_y),
              resultSaveMean_(resultSaveMean),
              resultSaveInvVariance_(resultSaveInvVariance),
              resultRunningMean_(resultRunningMean),
              resultRunningVariance_(resultRunningVariance)
        {
            ignore = xStrides;
            ignore = yStrides;
            ignore = bnScaleStrides;
            ignore = bnBiasStrides;
            ignore = bnMeanVarStrides;
            ignore = reduceDims;

            if(bnScaleBiasMeanVarLengths[0] != xyLengths[3])
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

        const XDataType* p_x_;
        const ScaleDataType* bnScale_;
        const BiasDataType* bnBias_;
        const YElementwiseOp y_elementwise_op_;
        YDataType* p_y_;

        MeanVarDataType* resultSaveMean_;
        MeanVarDataType* resultSaveInvVariance_;
        MeanVarDataType* resultRunningMean_;
        MeanVarDataType* resultRunningVariance_;

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
                    arg.resultSaveMean_[iC]        = type_convert<MeanVarDataType>(mean);
                    arg.resultSaveInvVariance_[iC] = type_convert<MeanVarDataType>(invVariance);
                };

                // update the moving average if required
                if(arg.resultRunning)
                {
                    AccDataType oneMinusAverageFactor =
                        type_convert<AccDataType>(1.0) - arg.averageFactor_;
                    arg.resultRunningMean_[iC] = type_convert<MeanVarDataType>(
                        type_convert<AccDataType>(arg.resultRunningMean_[iC]) *
                            oneMinusAverageFactor +
                        mean * arg.averageFactor_);
                    arg.resultRunningVariance_[iC] = type_convert<MeanVarDataType>(
                        arg.resultRunningVariance_[iC] * oneMinusAverageFactor +
                        variance * arg.averageFactor_);
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

                            arg.p_y_[offset] = type_convert<YDataType>(norm_x);
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
                        const std::array<index_t, 1> bnScaleStrides,
                        const std::array<index_t, 1> bnBiasStrides,
                        const std::array<index_t, 1> bnMeanVarStrides,
                        const void* p_x,
                        const void* bnScale,
                        const void* bnBias,
                        double epsilon,
                        const YElementwiseOp y_elementwise_op,
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
                                          bnScaleStrides,
                                          bnBiasStrides,
                                          bnMeanVarStrides,
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const ScaleDataType*>(bnScale),
                                          static_cast<const BiasDataType*>(bnBias),
                                          epsilon,
                                          y_elementwise_op,
                                          static_cast<YDataType*>(p_y),
                                          static_cast<MeanVarDataType*>(resultSaveMean),
                                          static_cast<MeanVarDataType*>(resultSaveInvVariance),
                                          averageFactor,
                                          static_cast<MeanVarDataType*>(resultRunningMean),
                                          static_cast<MeanVarDataType*>(resultRunningVariance));
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
