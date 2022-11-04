// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <algorithm>

#include "ck/tensor_operation/gpu/device/device_batchnorm_backward.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename XDataType,
          typename DyDataType,
          typename DxDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp>
struct ReferenceBatchNormBwd_Input_N_H_W_C_Output_C
    : public device::DeviceBatchNormBwd<4, 3, DyElementwiseOp>
{
    struct Argument : public device::BaseArgument
    {
        Argument(const std::array<index_t, 4> xyLengths,
                 const std::array<index_t, 4> xStrides,
                 const std::array<index_t, 4> dyStrides,
                 const std::array<index_t, 4> dxStrides,
                 const std::array<int, 3> reduceDims,
                 const std::array<ck::index_t, 1> bnScaleBiasMeanVarLengths,
                 const std::array<ck::index_t, 1> bnScaleStrides,
                 const std::array<ck::index_t, 1> bnBiasStrides,
                 const std::array<ck::index_t, 1> bnMeanVarStrides,
                 const XDataType* p_x,
                 const DyDataType* p_dy,
                 const ScaleDataType* p_scale,
                 const MeanVarDataType* p_savedMean,
                 const MeanVarDataType* p_savedInvVar,
                 double epsilon,
                 const DyElementwiseOp dy_elementwise_op,
                 DxDataType* p_dx,
                 ScaleDataType* p_dscale,
                 BiasDataType* p_dbias)
            : p_x_(p_x),
              p_dy_(p_dy),
              p_scale_(p_scale),
              p_savedMean_(p_savedMean),
              p_savedInvVar_(p_savedInvVar),
              epsilon_(epsilon),
              dy_elementwise_op_(dy_elementwise_op),
              p_dx_(p_dx),
              p_dscale_(p_dscale),
              p_dbias_(p_dbias)
        {
            ignore = xStrides;
            ignore = dyStrides;
            ignore = dxStrides;
            ignore = bnScaleStrides;
            ignore = bnBiasStrides;
            ignore = bnMeanVarStrides;

            if(xyLengths.size() != 4 || bnScaleBiasMeanVarLengths.size() != 1 ||
               bnScaleBiasMeanVarLengths[0] != xyLengths[3])
                throw std::runtime_error("Invalid tensor dimensions!");

            if(reduceDims[0] != 0 || reduceDims[1] != 1 || reduceDims[2] != 2)
                throw std::runtime_error("Invalid reduce dimensions!");

            n_ = xyLengths[0];
            h_ = xyLengths[1];
            w_ = xyLengths[2];
            c_ = xyLengths[3];

            haveSavedMeanInvVar_ = (p_savedMean != nullptr && p_savedInvVar != nullptr);
        }

        const XDataType* p_x_;
        const DyDataType* p_dy_;
        const ScaleDataType* p_scale_;
        const MeanVarDataType* p_savedMean_;
        const MeanVarDataType* p_savedInvVar_;

        double epsilon_;
        const DyElementwiseOp dy_elementwise_op_;

        DxDataType* p_dx_;
        ScaleDataType* p_dscale_;
        BiasDataType* p_dbias_;

        bool haveSavedMeanInvVar_;

        index_t n_, h_, w_, c_;
    };

    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            auto thread_reduce_func = [&](auto iC) {
                AccDataType reduceSize = type_convert<AccDataType>(arg.n_) *
                                         type_convert<AccDataType>(arg.h_) *
                                         type_convert<AccDataType>(arg.w_);
                index_t offset_C = iC;
                AccDataType mean;
                AccDataType invVar;

                if(arg.haveSavedMeanInvVar_)
                {
                    mean   = arg.p_savedMean_[offset_C];
                    invVar = arg.p_savedInvVar_[offset_C];
                }
                else
                {
                    AccDataType meansquare;

                    meansquare = type_convert<AccDataType>(0.0f);
                    mean       = type_convert<AccDataType>(0.0f);

                    // compute mean, meanquare, variance, inv-variance
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

                                mean += x;
                                meansquare += x * x;
                            };
                        }
                    };

                    mean       = mean / reduceSize;
                    meansquare = meansquare / reduceSize;

                    AccDataType variance = meansquare - mean * mean;
                    invVar               = type_convert<AccDataType>(1.0f) /
                             std::sqrt(type_convert<AccDataType>(arg.epsilon_) + variance);
                };

                AccDataType dbias  = type_convert<AccDataType>(0.0f); // Sum on NHW of dy
                AccDataType dscale = type_convert<AccDataType>(0.0f); // Sum on NHW of dy * norm_x

                // 1) calculate dy * (x - mean) * inv-variance
                // 2) calculate sum(dy) on NHW dimensions
                // 3) calculate sum(dy * norm_x) on NHW dimensions
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

                            AccDataType norm_x = (x - mean) * invVar;
                            AccDataType dy     = type_convert<AccDataType>(arg.p_dy_[offset]);

                            arg.dy_elementwise_op_(dy, dy);

                            dbias += dy;
                            dscale += norm_x * dy;
                        };
                    }
                };

                arg.p_dscale_[offset_C] = type_convert<ScaleDataType>(dscale);
                arg.p_dbias_[offset_C]  = type_convert<BiasDataType>(dbias);

                // 1) calculate tmp = dscale * (x - mean) * inv-variance
                // 2) calculate dx = 1/nhw * inv-variance * scale * (nhw * dy - dbias - tmp)
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

                            AccDataType norm_x = (x - mean) * invVar;
                            AccDataType dy     = type_convert<AccDataType>(arg.p_dy_[offset]);
                            AccDataType scale  = type_convert<AccDataType>(arg.p_scale_[offset_C]);

                            arg.dy_elementwise_op_(dy, dy);

                            AccDataType tmpVal = norm_x * dscale;

                            AccDataType dx = type_convert<AccDataType>(1.0f) / reduceSize * invVar *
                                             scale * (reduceSize * dy - dbias - tmpVal);

                            arg.p_dx_[offset] = type_convert<XDataType>(dx);
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
                        const std::array<index_t, 4> dyStrides,
                        const std::array<index_t, 4> dxStrides,
                        const std::array<int, 3> reduceDims,
                        const std::array<ck::index_t, 1> bnScaleBiasMeanVarLengths,
                        const std::array<ck::index_t, 1> bnScaleStrides,
                        const std::array<ck::index_t, 1> bnBiasStrides,
                        const std::array<ck::index_t, 1> bnMeanVarStrides,
                        const void* p_x,
                        const void* p_dy,
                        const void* p_scale,
                        const void* p_savedMean,
                        const void* p_savedInvVar,
                        double epsilon,
                        const DyElementwiseOp dy_elementwise_op,
                        void* p_dx,
                        void* p_dscale,
                        void* p_dbias) override
    {
        return std::make_unique<Argument>(xyLengths,
                                          xStrides,
                                          dyStrides,
                                          dxStrides,
                                          reduceDims,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleStrides,
                                          bnBiasStrides,
                                          bnMeanVarStrides,
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const DyDataType*>(p_dy),
                                          static_cast<const ScaleDataType*>(p_scale),
                                          static_cast<const MeanVarDataType*>(p_savedMean),
                                          static_cast<const MeanVarDataType*>(p_savedInvVar),
                                          epsilon,
                                          dy_elementwise_op,
                                          static_cast<DxDataType*>(p_dx),
                                          static_cast<ScaleDataType*>(p_dscale),
                                          static_cast<BiasDataType*>(p_dbias));
    };

    std::unique_ptr<device::BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "Reference_BatchNorm_Backward_NHWC_C<" << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
