// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>

#include "ck/tensor_operation/gpu/device/device_batchnorm_infer.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType>
struct ReferenceBatchNormInfer_Input_N_H_W_C_Output_C : public device::DeviceBatchNormInfer<4, 3>
{
    struct Argument : public device::BaseArgument
    {
        Argument(const std::array<index_t, 4> xyLengths,
                 const std::array<index_t, 4> xStrides,
                 const std::array<index_t, 4> yStrides,
                 const std::array<index_t, 1> bnScaleBiasMeanVarLengths,
                 const std::array<index_t, 1> bnScaleStrides,
                 const std::array<index_t, 1> bnBiasStrides,
                 const std::array<index_t, 1> bnMeanVarStrides,
                 const XDataType* p_x,
                 const ScaleDataType* bnScale,
                 const BiasDataType* bnBias,
                 double epsilon,
                 const MeanVarDataType* estimatedMean,
                 const MeanVarDataType* estimatedVariance,
                 YDataType* p_y)
            : p_x_(p_x),
              bnScale_(bnScale),
              bnBias_(bnBias),
              epsilon_(epsilon),
              estimatedMean_(estimatedMean),
              estimatedVariance_(estimatedVariance),
              p_y_(p_y)
        {
            ignore = xStrides;
            ignore = yStrides;
            ignore = bnScaleStrides;
            ignore = bnBiasStrides;
            ignore = bnMeanVarStrides;

            if(xyLengths.size() != 4 || bnScaleBiasMeanVarLengths.size() != 1 ||
               bnScaleBiasMeanVarLengths[0] != xyLengths[3])
                throw std::runtime_error("Invalid tensor dimensions!");

            n_ = xyLengths[0];
            h_ = xyLengths[1];
            w_ = xyLengths[2];
            c_ = xyLengths[3];
        }

        const XDataType* p_x_;
        const ScaleDataType* bnScale_;
        const BiasDataType* bnBias_;

        double epsilon_;

        const MeanVarDataType* estimatedMean_;
        const MeanVarDataType* estimatedVariance_;

        YDataType* p_y_;

        index_t n_, h_, w_, c_;
    };

    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            auto thread_reduce_func = [&](auto iC) {
                index_t offset_C     = iC;
                AccDataType mean     = arg.estimatedMean_[offset_C];
                AccDataType variance = arg.estimatedVariance_[offset_C];

                AccDataType invVariance =
                    type_convert<AccDataType>(1.0f) /
                    std::sqrt(type_convert<AccDataType>(arg.epsilon_) + variance);

                // Normalization
                for(index_t iN = 0; iN < arg.n_; iN++)
                {
                    index_t offset_N = iN * arg.h_ * arg.w_ * arg.c_;
                    for(index_t iH = 0; iH < arg.h_; iH++)
                    {
                        index_t offset_H = iH * arg.w_ * arg.c_;
                        for(index_t iW = 0; iW < arg.w_; iW++)
                        {
                            index_t offset_W = iW * arg.c_;

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
            std::size_t work_per_thread = (arg.c_ + num_thread - 1) / num_thread;

            std::vector<joinable_thread> threads(num_thread);

            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t ic_begin = it * work_per_thread;
                std::size_t ic_end = std::min(static_cast<int>((it + 1) * work_per_thread), arg.c_);

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
                        const std::array<index_t, 1> bnScaleStrides,
                        const std::array<index_t, 1> bnBiasStrides,
                        const std::array<index_t, 1> bnMeanVarStrides,
                        const void* p_x,
                        const void* bnScale,
                        const void* bnBias,
                        double epsilon,
                        const void* estimatedMean,
                        const void* estimatedVariance,
                        void* p_y) override
    {
        return std::make_unique<Argument>(xyLengths,
                                          xStrides,
                                          yStrides,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleStrides,
                                          bnBiasStrides,
                                          bnMeanVarStrides,
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const ScaleDataType*>(bnScale),
                                          static_cast<const BiasDataType*>(bnBias),
                                          epsilon,
                                          static_cast<const MeanVarDataType*>(estimatedMean),
                                          static_cast<const MeanVarDataType*>(estimatedVariance),
                                          static_cast<YDataType*>(p_y));
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
