#ifndef REFERENCE_BATCHNORM_BACKWARD_NHWC_C_HPP
#define REFERENCE_BATCHNORM_BACKWARD_NHWC_C_HPP

#include <iostream>
#include <sstream>
#include <algorithm>
#include "device_batchnorm_backward.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename InOutDataType, typename AccDataType>
struct ReferenceBatchNormBwd_Input_N_H_W_C_Output_C : public device::DeviceBatchNormBwd
{
    struct Argument : public device::BaseArgument
    {
        Argument(const std::vector<ck::index_t> xyLengths,
                 const std::vector<ck::index_t> xStrides,
                 const std::vector<ck::index_t> dyStrides,
                 const std::vector<ck::index_t> dxStrides,
                 const std::vector<ck::index_t> bnScaleBiasDiffLengths,
                 const std::vector<ck::index_t> bnScaleBiasDiffStrides,
                 const InOutDataType* p_x,
                 const InOutDataType* p_dy,
                 const AccDataType* bnScale,
                 const AccDataType* savedMean,
                 const AccDataType* savedInvVariance,
                 double epsilon,
                 InOutDataType* p_dx,
                 AccDataType* resultBnScaleDiff,
                 AccDataType* resultBnBiasDiff)
            : p_x_(p_x),
              p_dy_(p_dy),
              bnScale_(bnScale),
              savedMean_(savedMean),
              savedInvVariance_(savedInvVariance),
              epsilon_(epsilon),
              p_dx_(p_dx),
              resultBnScaleDiff_(resultBnScaleDiff),
              resultBnBiasDiff_(resultBnBiasDiff)
        {
            (void)xStrides;
            (void)dyStrides;
            (void)dxStrides;
            (void)bnScaleBiasDiffStrides;

            if(xyLengths.size() != 4 || bnScaleBiasDiffLengths.size() != 1 ||
               bnScaleBiasDiffLengths[0] != xyLengths[3])
                throw std::runtime_error("Invalid tensor dimensions!");

            n = xyLengths[0];
            h = xyLengths[1];
            w = xyLengths[2];
            c = xyLengths[3];

            use_savedMeanAndInvVariance = (savedMean != nullptr && savedInvVariance != nullptr);
        }

        const InOutDataType* p_x_;
        const InOutDataType* p_dy_;
        const AccDataType* bnScale_;
        const AccDataType* savedMean_;
        const AccDataType* savedInvVariance_;

        double epsilon_;

        InOutDataType* p_dx_;
        AccDataType* resultBnScaleDiff_;
        AccDataType* resultBnBiasDiff_;

        bool use_savedMeanAndInvVariance;

        index_t n, h, w, c;
    };

    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            auto thread_reduce_func = [&](auto iC) {
                AccDataType reduceSize = type_convert<AccDataType>(arg.n) *
                                         type_convert<AccDataType>(arg.h) *
                                         type_convert<AccDataType>(arg.w);
                index_t offset_C = iC;
                AccDataType mean;
                AccDataType invVariance;

                if(arg.use_savedMeanAndInvVariance)
                {
                    mean        = arg.savedMean_[offset_C];
                    invVariance = arg.savedInvVariance_[offset_C];
                }
                else
                {
                    AccDataType meansquare;

                    meansquare = type_convert<AccDataType>(0.0f);
                    mean       = type_convert<AccDataType>(0.0f);

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
                    invVariance          = type_convert<AccDataType>(1.0f) /
                                  std::sqrt(type_convert<AccDataType>(arg.epsilon_) + variance);
                };

                AccDataType bnBiasDiff = type_convert<AccDataType>(0.0f); // Sum on NHW of dy
                AccDataType bnScaleDiff =
                    type_convert<AccDataType>(0.0f); // Sum on NHW of dy * norm_x

                // 1) calculate dy * (x - mean) * invVariance
                // 2) calculate Sum on NHWC of dy
                // 3) calculate Sum on NHWC of dy * norm_x
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

                            AccDataType norm_x = (x - mean) * invVariance;
                            AccDataType dy     = type_convert<AccDataType>(arg.p_dy_[offset]);

                            bnBiasDiff += dy;
                            bnScaleDiff += type_convert<AccDataType>(norm_x * dy);
                        };
                    }
                };

                // 1) calculate tmp = scaleDiff * (x - mean) * invVariance
                // 2) calculate dx = 1/nhw * invVariance * scale * (nhw * dy - biasDiff - tmp)
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

                            AccDataType norm_x = (x - mean) * invVariance;
                            AccDataType dy     = type_convert<AccDataType>(arg.p_dy_[offset]);

                            AccDataType tmpVal = norm_x * bnScaleDiff;

                            AccDataType dx = type_convert<AccDataType>(1.0f) / reduceSize *
                                             invVariance * arg.bnScale_[offset_C] *
                                             (reduceSize * dy - bnBiasDiff - tmpVal);

                            arg.p_dx_[offset] = dx;
                        };
                    }
                };

                arg.resultBnBiasDiff_[offset_C]  = type_convert<InOutDataType>(bnBiasDiff);
                arg.resultBnScaleDiff_[offset_C] = type_convert<InOutDataType>(bnScaleDiff);
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
    MakeArgumentPointer(const std::vector<ck::index_t> xyLengths,
                        const std::vector<ck::index_t> xStrides,
                        const std::vector<ck::index_t> dyStrides,
                        const std::vector<ck::index_t> dxStrides,
                        const std::vector<ck::index_t> bnScaleBiasDiffLengths,
                        const std::vector<ck::index_t> bnScaleBiasDiffStrides,
                        const void* p_x,
                        const void* p_dy,
                        const void* bnScale,
                        const void* savedMean,
                        const void* savedInvVariance,
                        double epsilon,
                        void* p_dx,
                        void* resultBnScaleDiff,
                        void* resultBnBiasDiff) override
    {
        return std::make_unique<Argument>(xyLengths,
                                          xStrides,
                                          dyStrides,
                                          dxStrides,
                                          bnScaleBiasDiffLengths,
                                          bnScaleBiasDiffStrides,
                                          static_cast<const InOutDataType*>(p_x),
                                          static_cast<const InOutDataType*>(p_dy),
                                          static_cast<const AccDataType*>(bnScale),
                                          static_cast<const AccDataType*>(savedMean),
                                          static_cast<const AccDataType*>(savedInvVariance),
                                          epsilon,
                                          static_cast<InOutDataType*>(p_dx),
                                          static_cast<AccDataType*>(resultBnScaleDiff),
                                          static_cast<AccDataType*>(resultBnBiasDiff));
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
#endif
