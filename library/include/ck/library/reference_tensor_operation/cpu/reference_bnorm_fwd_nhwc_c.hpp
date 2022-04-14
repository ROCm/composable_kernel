#ifndef REFERENCE_BNORM_FWD_NHWC_C_HPP
#define REFERENCE_BNORM_FWD_NHWC_C_HPP

#include <iostream>
#include <sstream>
#include <algorithm>
#include "device_bnorm_fwd.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename InOutDataType, typename AccDataType>
struct ReferenceBatchNormFwd_Input_N_H_W_C_Output_C : public device::DeviceBatchNormFwd
{
    struct Argument : public device::BaseArgument
    {
        Argument(const std::vector<index_t> inLengths,
                 const std::vector<index_t> inStrides,
                 const std::vector<index_t> outLengths,
                 const std::vector<index_t> outStrides,
                 const std::vector<index_t> bnScaleBiasMeanVarLengths,
                 const std::vector<index_t> bnScaleBiasMeanVarStrides,
                 float alpha,
                 float beta,
                 const InOutDataType* in_dev,
                 InOutDataType* out_dev,
                 AccDataType* workspace_dev,
                 const AccDataType* bnScale,
                 const AccDataType* bnBias,
                 double exponentialAverageFactor,
                 AccDataType* resultRunningMean,
                 AccDataType* resultRunningVariance,
                 double epsilon,
                 AccDataType* resultSaveMean,
                 AccDataType* resultSaveInvVariance)
            : in_dev_(in_dev),
              out_dev_(out_dev),
              bnScale_(bnScale),
              bnBias_(bnBias),
              resultRunningMean_(resultRunningMean),
              resultRunningVariance_(resultRunningVariance),
              resultSaveMean_(resultSaveMean),
              resultSaveInvVariance_(resultSaveInvVariance),
              exponentialAverageFactor_(exponentialAverageFactor),
              epsilon_(epsilon)
        {
            (void)inStrides;
            (void)outStrides;
            (void)bnScaleBiasMeanVarStrides;
            (void)workspace_dev;

            if(inLengths.size() != 4 || outLengths.size() != 4 ||
               bnScaleBiasMeanVarLengths.size() != 1)
                throw std::runtime_error("Invalid tensor dimensions!");

            n = inLengths[0];
            h = inLengths[1];
            w = inLengths[2];
            c = inLengths[3];

            if(outLengths[3] != c || bnScaleBiasMeanVarLengths[0] != c)
                throw std::runtime_error("Inconsistent tensor lengths!");

            if(alpha != 1.0f || beta != 0.0f)
                throw std::runtime_error(
                    "Only alpha of value 1.0f and beta of value 0.0f is supported!");

            resultSave    = (resultSaveMean != nullptr && resultSaveInvVariance != nullptr);
            resultRunning = (resultRunningMean != nullptr && resultRunningVariance != nullptr);
        }

        const InOutDataType* in_dev_;
        InOutDataType* out_dev_;
        const AccDataType* bnScale_;
        const AccDataType* bnBias_;
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
        float Run(const Argument& arg, int nrepeat = 1)
        {
            (void)nrepeat;

            auto thread_reduce_func = [&](auto iC) {
                index_t offset_C       = iC;
                AccDataType mean       = type_convert<AccDataType>(0.0f);
                AccDataType meansquare = type_convert<AccDataType>(0.0f);

                // compute mean, meanquare, variance, invVariance
                for(int iN = 0; iN < arg.n; iN++)
                {
                    index_t offset_N = iN * arg.h * arg.w * arg.c;
                    for(int iH = 0; iH < arg.h; iH++)
                    {
                        index_t offset_H = iH * arg.w * arg.c;
                        for(int iW = 0; iW < arg.w; iW++)
                        {
                            index_t offset_W = iW * arg.c;

                            auto offset = offset_N + offset_H + offset_W + offset_C;

                            AccDataType curr_value = type_convert<AccDataType>(arg.in_dev_[offset]);

                            mean += curr_value;
                            meansquare += curr_value * curr_value;
                        };
                    }
                };

                mean       = mean / (arg.n * arg.h * arg.w);
                meansquare = meansquare / (arg.n * arg.h * arg.w);

                AccDataType variance    = meansquare - mean * mean;
                AccDataType invVariance = type_convert<AccDataType>(1.0f) /
                                          sqrt(type_convert<AccDataType>(arg.epsilon_) + variance);

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
                for(int iN = 0; iN < arg.n; iN++)
                {
                    index_t offset_N = iN * arg.h * arg.w * arg.c;
                    for(int iH = 0; iH < arg.h; iH++)
                    {
                        index_t offset_H = iH * arg.w * arg.c;
                        for(int iW = 0; iW < arg.w; iW++)
                        {
                            index_t offset_W = iW * arg.c;

                            auto offset = offset_N + offset_H + offset_W + offset_C;

                            AccDataType curr_value = type_convert<AccDataType>(arg.in_dev_[offset]);

                            arg.out_dev_[offset] =
                                arg.bnScale_[iC] * (curr_value - mean) * invVariance +
                                arg.bnBias_[iC];
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

        float Run(const device::BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        };
    };

    bool IsSupportedArgument(const device::BaseArgument* p_arg) override
    {
        (void)p_arg;

        return (true);
    };

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> inLengths,
                        const std::vector<index_t> inStrides,
                        const std::vector<index_t> outLengths,
                        const std::vector<index_t> outStrides,
                        const std::vector<index_t> bnScaleBiasMeanVarLengths,
                        const std::vector<index_t> bnScaleBiasMeanVarStrides,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        void* out_dev,
                        void* workspace_dev,
                        const void* bnScale,
                        const void* bnBias,
                        double exponentialAverageFactor,
                        void* resultRunningMean,
                        void* resultRunningVariance,
                        double epsilon,
                        void* resultSaveMean,
                        void* resultSaveInvVariance) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleBiasMeanVarStrides,
                                          alpha,
                                          beta,
                                          static_cast<const InOutDataType*>(in_dev),
                                          static_cast<InOutDataType*>(out_dev),
                                          static_cast<AccDataType*>(workspace_dev),
                                          static_cast<const AccDataType*>(bnScale),
                                          static_cast<const AccDataType*>(bnBias),
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
        str << "Reference_BatchNorm_NHWC_C<" << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
#endif
